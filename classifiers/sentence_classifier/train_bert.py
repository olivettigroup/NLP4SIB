"""
Script for training NLP models to analyzing sodium-ion batteries
using Optuna and Transformers.
"""

import logging
from functools import partial
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import optuna
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
)
from datasets import load_metric, Dataset
import os

logging.getLogger("transformers").setLevel(logging.ERROR)
# Disable parallelism to avoid deadlocks in DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"

root_dir = os.getcwd()

DATASET_NAME = f"{root_dir}/classifiers/sentence_classifier/datasets/mitigation_dataset.csv"
MODEL_NAME = "m3rg-iitd/matscibert"
STUDY_NAME = f"challenge_dataset"

DEVICE = torch.device("cuda") if torch.cuda.is_available()\
      else torch.device("cpu")
N_TRIALS = 100

def get_weighted_random_sampler(y_train):
    """Create a weighted random sampler for the imbalanced dataset."""
    class_sample_count = np.array([len(np.where(y_train == t)[0])\
                                    for t in np.unique(y_train)])
    weights = 1. / class_sample_count
    samples_weights = np.array([weights[t] for t in y_train])
    samples_weights = torch.from_numpy(samples_weights).double()
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))
    return sampler


def tokenize_function(tokenizer, examples):
    """Tokenize the text using the tokenizer."""
    return tokenizer(examples['text'], padding=True, \
                     truncation=True, max_length=512)


def load_datasets(tokenizer, is_final_run=False):
    """Get the datasets, preprocess, and tokenize them.
    Args:
        tokenizer: Tokenizer instance to tokenize the data.
        is_final_run: Boolean indicating if this is the
        final run (returns the test set if True).
    Returns:
        tokenized_train_datasets: Tokenized training dataset.
        tokenized_dev_datasets: Tokenized development dataset.
        y_train: Labels.
    """
    try:
        labeled_data = pd.read_csv(DATASET_NAME)
    except FileNotFoundError as exc:
        raise Exception(f"Dataset file {DATASET_NAME} not found.") from exc

    labeled_data['label'] = (~labeled_data['label'].isin(['no'])).astype(int)

    # Split the data
    train_condition = labeled_data['split'] == 'train'
    dev_condition = labeled_data['split'] == 'dev' if not is_final_run \
        else labeled_data['split'] == 'test'
    df_train = labeled_data[train_condition | (is_final_run & dev_condition)]\
        [['text', 'label']]
    df_dev = labeled_data[dev_condition][['text', 'label']]

    print('Number of total datapoints (train/dev/test):', len(labeled_data))
    print('Number of train datapoints:', len(df_train))
    print('Number of dev/test datapoints:', len(df_dev))

    y_train = df_train['label'].to_numpy()
    train_dataset = Dataset.from_pandas(df_train)
    dev_dataset = Dataset.from_pandas(df_dev)

    tokenize_with_tokenizer = partial(tokenize_function, tokenizer)
    tokenized_train_datasets = \
        train_dataset.map(tokenize_with_tokenizer, batched=True)
    tokenized_dev_datasets = \
        dev_dataset.map(tokenize_with_tokenizer, batched=True)

    tokenized_train_datasets = tokenized_train_datasets.remove_columns(
        ["text", "__index_level_0__"]
        )
    tokenized_dev_datasets = tokenized_dev_datasets.remove_columns(
        ["text", "__index_level_0__"]
        )

    tokenized_train_datasets = \
        tokenized_train_datasets.rename_column("label", "labels")
    tokenized_dev_datasets = \
        tokenized_dev_datasets.rename_column("label", "labels")

    tokenized_train_datasets.set_format("torch")
    tokenized_dev_datasets.set_format("torch")

    return tokenized_train_datasets, tokenized_dev_datasets, y_train


def training_loop(train_dataloader, eval_dataloader, model, optimizer,
                  num_training_steps, num_epochs, lr_scheduler, device,
                  is_final_run, study_name, tokenizer):
    model.train()
    eval_metrics = train_metrics = {'f1': load_metric('f1'),
                                    'precision': load_metric('precision'),
                                    'recall': load_metric('recall')}
    progress_bar = tqdm(total=num_training_steps, desc='Training')

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            predictions = torch.argmax(outputs.logits, dim=-1)
            for metric in train_metrics.values():
                metric.add_batch(predictions=predictions, \
                                 references=batch["labels"])
            progress_bar.update(1)

        progress_desc = (
            f"Epoch {epoch+1}/{num_epochs}"
            f" -- train f1: {train_metrics['f1'].compute(average='macro')['f1']:.4f}"
        )
        progress_bar.set_description(progress_desc)

    progress_bar.close()
    test_score = evaluate_model(eval_metrics, eval_dataloader, model, device)

    if is_final_run:
        save_model_and_tokenizer(model, tokenizer, study_name, test_score)
        return test_score
    print(f'\033[1;32mThis run has ended with a '
      f'development F1-score of: {test_score:.4f}\033[0m')
    return test_score


def evaluate_model(metrics, eval_dataloader, model, device):
    """Evaluate the model on the development set."""
    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            for metric in metrics.values():
                metric.add_batch(predictions=predictions, references=batch["labels"])
    return metrics['f1'].compute(average='macro')['f1']


def save_model_and_tokenizer(model, tokenizer, study_name, score):
    """Save the model and tokenizer to the models folder."""
    model.save_pretrained(f"models/{study_name}")
    tokenizer.save_pretrained(f"models/{study_name}")
    print(f"The test F1 score is: {round(score, 3)}")
    """for key, values in score.items():
        print(f'{key.capitalize()} - Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}')"""
    
    print(f'Model and tokenizer saved to models/{study_name}')


def objective(trial, tokenizer, device, model_name, is_final_run=False):
    """Objective function for Optuna."""
    # Define hyperparameter space using trial object
    num_epochs = trial.suggest_int('epochs', 1, 5)
    batch_size_exponent = trial.suggest_int('batch_size_exponent', 0, 3)
    batch_size = int(2 ** batch_size_exponent)
    learning_rate = trial.suggest_float('learning_rate', 5e-6, 1e-4)

    # Load the data once if it doesn't depend on trial parameters
    tokenized_train_datasets, tokenized_dev_datasets, y_train = load_datasets(tokenizer, is_final_run=is_final_run)
    sampler = get_weighted_random_sampler(y_train)
    eval_dataloader = DataLoader(tokenized_dev_datasets, batch_size=8)

    f1_scores = []

    for _ in range(3):
        # Initialize model and related components for each trial,
        # repeat 3 times for statistical significance
        model = AutoModelForSequenceClassification.\
            from_pretrained(model_name, num_labels=np.unique(y_train).shape[0])
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        train_dataloader = DataLoader(tokenized_train_datasets,\
                                      batch_size=batch_size, sampler=sampler)
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler("linear", optimizer=optimizer,\
                                     num_warmup_steps=0, num_training_steps=num_training_steps)
        model.to(device)
        # Run the training loop
        score = training_loop(train_dataloader, eval_dataloader, model,\
                              optimizer, num_training_steps, num_epochs,\
                                lr_scheduler, device, is_final_run, STUDY_NAME, tokenizer)
        f1_scores.append(score)
        torch.cuda.empty_cache()  # Only if using GPU

    # Return the average F1 score from the trials
    return np.mean(f1_scores)

def main():
    """Main function."""
    # Initialize the tokenizer and seeds
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    torch.manual_seed(0)
    np.random.seed(0)

    # Start Optuna study
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(),
                                load_if_exists=True, study_name=STUDY_NAME, \
                                    storage="sqlite:///bert.db")
    #study.optimize(lambda trial: objective(trial, tokenizer, DEVICE, MODEL_NAME), n_trials=N_TRIALS)
    
    study = optuna.load_study(study_name=STUDY_NAME, 
                                storage="sqlite:///bert.db")
    print(f"Completed trials: {len([trial for trial in study.get_trials() if trial.state == optuna.trial.TrialState.COMPLETE])}")
    # Final run with best parameters
    print(f"The study has {len(study.get_trials())} trials!")
    print("Performing the final RUN!")

    best_trial = study.best_trial
    print(f"f1: {best_trial.value}")
    print(f"Best hyperparameters: {best_trial.params}")
    print(f'The final score is: {objective(best_trial, tokenizer, DEVICE, MODEL_NAME, is_final_run=True)}')


if __name__ == "__main__":
    main()
