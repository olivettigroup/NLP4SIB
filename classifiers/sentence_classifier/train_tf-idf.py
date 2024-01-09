"""
This script trains a TF-IDF based text classifier.
It reads a dataset from a CSV file, preprocesses the data, 
and splits it into train, dev, and test sets.
The script then performs hyperparameter optimization using Optuna 
and evaluates the model's performance.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import evaluate
import optuna

# Use absolute paths and os.path.join for OS-independent file paths
root_dir = os.getcwd()
parent_dir = os.path.dirname(root_dir)
dataset_path = os.path.join(parent_dir, 'datasets', 'challenge_dataset.csv')

# Read dataset
df = pd.read_csv(dataset_path)

# Convert 'label' to a categorical type and get its codes
df['challenge_type'] = df['label'].astype('category')
df['categorical'] = df['challenge_type'].cat.codes

# Prepare data
sentences = df['text'].tolist()
split = df['split'].tolist()
targets = df['categorical'].to_numpy()

# Vectorize text data
tfidf_vectorizer = TfidfVectorizer(encoding='utf-8', decode_error='strict',
                                   lowercase=True, analyzer='word', stop_words=None,
                                   norm='l2', use_idf=True, smooth_idf=True)
X = tfidf_vectorizer.fit_transform(sentences)

# Split data into train, dev, and test sets
X_train = X[np.array(split) == 'train']
X_dev = X[np.array(split) == 'dev']
X_test = X[np.array(split) == 'test']
y_train = targets[np.array(split) == 'train']
y_dev = targets[np.array(split) == 'dev']
y_test = targets[np.array(split) == 'test']

def plot_confusion_matrix(plot_title, y_pred, y_true, class_names):
    """
    Plots a confusion matrix using seaborn and matplotlib.

    Args:
        plot_title (str): Title of the plot.
        y_pred (array-like): Predicted labels.
        y_true (array-like): True labels.
        class_names (list): List of class names.

    Returns:
        None
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    plt.figure(figsize=(10,7))
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.title(plot_title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png')

# Metrics computation functions
def compute_f1(predictions, labels):
    """
    Computes the F1 score.

    Args:
        predictions (array-like): Predicted labels.
        labels (array-like): True labels.

    Returns:
        float: F1 score.
    """
    f1_metric = evaluate.load("f1")
    return f1_metric.compute(predictions=predictions, references=labels, average='macro')['f1']

def compute_precision(predictions, labels):
    """
    Computes the precision score.

    Args:
        predictions (array-like): Predicted labels.
        labels (array-like): True labels.

    Returns:
        float: Precision score.
    """
    precision_metric = evaluate.load("precision")
    return precision_metric.compute(predictions=predictions, references=labels,
                                    average='macro')['precision']

def compute_recall(predictions, labels):
    """
    Computes the recall score.

    Args:
        predictions (array-like): Predicted labels.
        labels (array-like): True labels.

    Returns:
        float: Recall score.
    """
    recall_metric = evaluate.load("recall")
    return recall_metric.compute(predictions=predictions, references=labels,
                                 average='macro')['recall']

def objective(trial):
    """
    Objective function for hyperparameter optimization.

    Args:
        trial (optuna.Trial): Optuna trial object.

    Returns:
        float: F1 score on the dev set.
    """
    # Hyperparameter optimization
    max_iter = trial.suggest_int('iter', 1, 12)
    alpha = trial.suggest_float('alpha', 1e-4, 1e-2)
    random_seed = trial.suggest_int('random_seed', 1, 12)

    text_clf = Pipeline([
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=alpha, random_state=random_seed,
                              max_iter=max_iter, tol=None)),
    ])

    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_dev)

    print(f'Precision: {compute_precision(predicted, y_dev)}')
    print(f'Recall: {compute_recall(predicted, y_dev)}')
    print(f'F1 Score: {compute_f1(predicted, y_dev)}')

    if TEST:
        predicted_test = text_clf.predict(X_test)
        print(f'Test Precision: {compute_precision(predicted_test, y_test)}')
        print(f'Test Recall: {compute_recall(predicted_test, y_test)}')
        plot_confusion_matrix('TF-IDF Confusion Matrix',
                              predicted_test, y_test, ['Both', 'Intro', 'Macro', 'Micro', 'None'])

    return compute_f1(predicted, y_dev)


TEST = False
STUDY_NAME = '100p_svm_x1'
DB_PATH = "sqlite:///tf-idf.db"

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(),
                           load_if_exists=True, study_name=STUDY_NAME, storage=DB_PATH)
study.optimize(objective, n_trials=500)

# Load existing studies after completion
TEST = True
study = optuna.load_study(study_name=STUDY_NAME, storage=DB_PATH)
objective(study.best_trial)

test = False
study_name = '100p_svm_x1'
db_path = "sqlite:///tf-idf.db"

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(),
                           load_if_exists=True, study_name=study_name, storage=db_path)
study.optimize(objective, n_trials=500)

# Load existing studies after completion
test = True
study = optuna.load_study(study_name=study_name, storage=db_path)
objective(study.best_trial)
