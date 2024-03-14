### 🔋🔍 **NLP4SIB**: Extracting a Database of Challenges and Mitigation Strategies for Sodium-ion Battery Development 

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)


---
### 🌟 Key Highlights
- **Database of Challenges**: A detailed compilation of the challenges faced in the performance and synthesis of SIB CAMs: `database/challenge_database.xlsx`
- **Mitigation Strategies**: Identification and pairing of challenges with potential mitigation strategies: `database/mitigation_database.xlsx`

### 🖥️ NLP Pipeline
![NLP Methods](figures/nlpmethods.png)
*The sequential filtering and visualization pipeline utilizing sentence classification, phrase-level classification, and relationship extraction.*

### 📩 Contact
For inquiries or further information, please contact: *mrigi@mit.edu*

### 🙌 Acknowledgements
Special thanks to Vineeth Venugopal, Elsa Olivetti, Kevin J. Huang, Ryan Stephens, and MIT for their support.

---

## Reproduce the Results
#### 🤖 **Train the Sentence Classifier** 

### Step 1: Create a new Env
Create a new environment & activate** named `nlp4sib` using Python 3.8:
```bash
conda create -n nlp4sib-sentence python==3.8
conda activate nlp4sib-sentence
```

### Step 2: Environment Setup
Install the necessary dependencies** from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
### Step 3: Start Training
Train the Sentence Classifier**:
```bash
python classifiers/sentence_classifier/train_bert.py 
```

## 🔍 Running Entity & Relation Extraction with DYGIE++

### Step 1: Initialize DyGIE++
First, initialize the DyGIE++ submodule with the following commands:
```bash
git submodule init
git submodule update
```

### Step 2: Environment Setup
Create and activate a new Conda environment for the project:
```bash
conda create -n nlp_4sib-phrase python==3.8
conda activate nlp_4sib-phrase
```
Navigate to the DyGIE++ directory and install the required dependencies:
```bash
cd classifiers/phrase_classifier/dygiepp
pip install -r requirements.txt
pip install transformers==4.2.1 # being able to use recent models
pip install numpy==1.19.0 # version conflict
conda develop .
cd ../..
```

### Step 3: Training DyGIE++
Train the DyGIE++ model using the labeled sentences. Two training configurations are provided below:

- **Improvement Configuration:**
    ```bash
    allennlp train "classifiers/phrase_classifier/best_parameter_study_improvement.json" \
        --serialization-dir "classifiers/phrase_classifier/dygiepp/models/improvement" \
        --include-package dygie
    ```

- **Challenge Configuration:**
    ```bash
    allennlp train "classifiers/phrase_classifier/best_parameter_study_challenge.json" \
        --serialization-dir "classifiers/phrase_classifier/dygiepp/models/challenge" \
        --include-package dygie
    ```


