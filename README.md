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

### **Reproduce the Paper's Results**
#### 🤖 **Train the Sentence Classifier** 

1. **Create a new environment & activate** named `nlp4sib` using Python 3.8:
   ```bash
   conda create -n nlp4sib python==3.8
   conda activate nlp4sib
   ```
2. **Install the necessary dependencies** from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```
3. **Train the Sentence Classifier**:
   ```bash
   python classifiers/sentence_classifier/train_bert.py 
   ```

#### 🔍 Run Entity & Relation Extraction with DYGGIE++

1. **Initialize DyGIE++**:
   ```bash
    git submodule init
    git submodule update
   ```
2. **Train DyGIE++** using the labeled sentences:
    ```bash
    allennlp train "phrase_classifier/best_parameter_study_improvement.json" \
        --serialization-dir "phrase_classifier/dygiepp/models/improvement" \
        --include-package dygie
    
    allennlp train "phrase_classifier/best_parameter_study_challenge.json" \
        --serialization-dir "phrase_classifier/dygiepp/models/challenge" \
        --include-package dygie 

