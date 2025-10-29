# SentimentAnalysis_CoreMLDemo_EmotionClassifier_iOS_SwiftUI

This is an **iOS demo application** for **emotion classification** using **CoreML** and **SwiftUI**.
The project follows **MVVM architecture** with **async/await** and **actor-based model management**.

The **CoreML model** was converted from a **Scikit-learn pipeline** using **coremltools Python package**, including:

* `DictVectorizer` for feature extraction
* `LinearSVC` for classification

<p align="center">
 <img src="https://github.com/user-attachments/assets/dbb7ede8-a229-4cfe-9a94-1a3873fcbd92" width="400" alt="Emotion Classifier Demo" />
</p>


---

## Dataset

The training dataset is from [Kaggle: Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp/data). But I have added the preprocessed and formatted CSV data in the ***converter*** folder for direct use in training.

### Steps:

1. Original text files converted to **CSV** with two columns:

   * `text` → user sentence or phrase
   * `emotion` → label (e.g., joy, sadness, anger, fear, love, surprise)

2. Python example to convert `.txt` → `.csv`:

```python
import csv
from pathlib import Path

# Input and output file paths
input_file = Path("val.txt")
output_file = Path("val.csv")

# Ensure input exists
if not input_file.exists():
    raise FileNotFoundError(f"❌ Input file not found: {input_file}")

# Open the input text file and create the output CSV file
with input_file.open("r", encoding="utf-8") as infile, output_file.open("w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["text", "emotion"])  # CSV header
    
    count = 0
    for line in infile:
        line = line.strip()
        if not line:
            continue
        if ";" in line:
            text, emotion = line.split(";", 1)
            writer.writerow([text.strip(), emotion.strip()])
            count += 1
        else:
            print(f"⚠️ Skipping malformed line: {line}")
    
print(f"✅ Conversion complete! {count} entries saved to '{output_file.name}'")

```

---

## Python Setup

1. Install **Python 3.9+**
2. Install dependencies:

```bash
pip install pandas numpy nltk scikit-learn coremltools
```

3. Download NLTK data:

Step 1: Open Python 3 in your terminal

```python
python3
```

Step 2: Run the following commands in the Python prompt (>>>)
```python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('stopwords')
```

Step 3: Exit Python
```python
>>> exit()
```
---

## Training the Model

Run:

```bash
python3 train_emotion_model.py
```

**Steps performed:**

* Load CSV data
* Preprocess text
* Extract bag-of-words features
* Train **LinearSVC** using GridSearchCV
* Save **CoreML model** as `EmotionClassifier.mlmodel`

---

# How the Model Learns (Step by Step Example)

This section explains how the emotion classification model learns from training data using a step-by-step example with multiple iterations.

---

## Example Training Dataset

| Text                         | Emotion  |
| ---------------------------- | -------- |
| "I am happy today"           | joy      |
| "I feel sad"                 | sadness  |
| "I am really angry"          | anger    |
| "I love my family"           | love     |
| "I am scared of the dark"    | fear     |
| "Wow, I didn’t expect that!" | surprise |
| "I am ecstatic and joyful!"  | joy      |
| "I am very upset and angry"  | anger    |
| "I adore my friends"         | love     |
| "The night makes me anxious" | fear     |
| "I am shocked by the news"   | surprise |
| "Feeling sad and lonely"     | sadness  |

---

## Step 1: Convert Text → Features (Bag-of-Words)

Take `"I am happy today"` → tokenize, remove stopwords:

```
['happy', 'today']
```

Feature dict (all words from dataset considered):

```
{'happy':1, 'today':1, 'sad':0, 'angry':0, 'love':0, 'scared':0, 'wow':0, 'expect':0, 'ecstatic':0, 'joyful':0, 'upset':0, 'family':0, 'friends':0, 'anxious':0, 'shocked':0, 'lonely':0}
```

---

## Step 2: Initialize Weights

Each emotion has a weight vector for all words (initialized to 0):

```
weights_joy = [happy=0, today=0, sad=0, angry=0, love=0, scared=0, wow=0, expect=0, ecstatic=0, joyful=0, upset=0, family=0, friends=0, anxious=0, shocked=0, lonely=0]
weights_sadness = same...
weights_anger = same...
weights_love = same...
weights_fear = same...
weights_surprise = same...
```

---

## Step 3: Training Iterations

### **Iteration 1**: "I am happy today" → joy

**Update weights:**
* Increase weights for the true emotion for the words present.
* Decrease weights for all other emotions for the words present.

```
weights_joy      = [happy=2, today=1, ...]
weights_sadness  = [happy=-1, today=-0.5, ...]
weights_anger    = [happy=-1, today=-0.5, ...]
weights_love     = [happy=-0.5, today=-0.2, ...]
weights_fear     = [happy=-0.5, today=-0.2, ...]
weights_surprise = [happy=-0.5, today=-0.2, ...]
```

### **Iteration 2**: "I feel sad" → sadness

```
weights_sadness  = [happy=-1, today=-0.5, sad=2, ...]
weights_joy      = [happy=2, today=1, sad=-1, ...]
weights_anger    = [happy=-1, today=-0.5, sad=-1, ...]
weights_love     = [happy=-0.5, today=-0.2, sad=-0.5, ...]
weights_fear     = [happy=-0.5, today=-0.2, sad=-0.5, ...]
weights_surprise = [happy=-0.5, today=-0.2, sad=-0.5, ...]
```

### **Iteration 3**: "I am really angry" → anger

```
weights_anger   = [angry=2, really=1, ...]
weights_joy     = decrease weights for angry, really
weights_sadness = decrease weights for angry, really
weights_love    = decrease weights for angry, really
weights_fear    = decrease weights for angry, really
weights_surprise= decrease weights for angry, really
```

### **Iteration 4**: "I love my family" → love

```
weights_love    = [love=2, family=1, ...]
```

### **Iteration 5**: "I am scared of the dark" → fear

```
weights_fear    = [scared=2, dark=1, ...]
```

### **Iteration 6**: "Wow, I didn’t expect that!" → surprise

```
weights_surprise = [wow=2, expect=1, ...]
```

### **Iteration 7**: "I am ecstatic and joyful!" → joy

```
weights_joy      = [happy=2, today=1, ecstatic=1, joyful=1, ...]
weights_other   = decrease weights for ecstatic, joyful
```

### **Iteration 8**: "I am very upset and angry" → anger

```
weights_anger    = [angry=3, upset=2, ...]
weights_other    = decrease weights for angry, upset
```

### **Iteration 9**: "I adore my friends" → love

```
weights_love     = [love=3, family=1, friends=1, ...]
```

### **Iteration 10**: "The night makes me anxious" → fear

```
weights_fear     = [scared=2, dark=1, anxious=1, ...]
```

### **Iteration 11**: "I am shocked by the news" → surprise

```
weights_surprise = [wow=2, expect=1, shocked=1, ...]
```

### **Iteration 12**: "Feeling sad and lonely" → sadness

```
weights_sadness  = [sad=3, lonely=1, ...]
```


## Final Weights Table (Word → Emotion)

| Word      | Joy | Sadness | Anger | Love | Fear | Surprise |
| --------- | --- | ------- | ----- | ---- | ---- | -------- |
| happy     | 2   | -1      | -1    | -0.5 | -0.5 | -0.5     |
| today     | 1   | -0.5    | -0.5  | -0.2 | -0.2 | -0.2     |
| sad       | -1  | 3       | -1    | -0.5 | -0.5 | -0.5     |
| angry     | -1  | -1      | 3     | -0.5 | -0.5 | -0.5     |
| love      | -0.5| -0.5    | -0.5  | 3    | -0.5 | -0.5     |
| scared    | -0.5| -0.5    | -0.5  | -0.5 | 2    | -0.5     |
| wow       | -0.5| -0.5    | -0.5  | -0.5 | -0.5 | 2        |
| expect    | -0.5| -0.5    | -0.5  | -0.5 | -0.5 | 1        |
| ecstatic  | 1   | -0.5    | -0.5  | -0.5 | -0.5 | -0.5     |
| joyful    | 1   | -0.5    | -0.5  | -0.5 | -0.5 | -0.5     |
| upset     | -0.5| -0.5    | 2     | -0.5 | -0.5 | -0.5     |
| family    | -0.5| -0.5    | -0.5  | 1    | -0.5 | -0.5     |
| friends   | -0.5| -0.5    | -0.5  | 1    | -0.5 | -0.5     |
| anxious   | -0.5| -0.5    | -0.5  | -0.5 | 1    | -0.5     |
| shocked   | -0.5| -0.5    | -0.5  | -0.5 | -0.5 | 1        |
| lonely    | -0.5| 1       | -0.5  | -0.5 | -0.5 | -0.5     |

> **Note:** Positive values indicate the strength of association between a word and an emotion. Negative values reduce association with other emotions. This table is used for predicting emotions of new sentences.

---

## Step 4: Prediction Formula

When a new sentence comes in, the model computes:

```
score_emotion = sum(word_count * weight_for_that_emotion)
```

### Example: Emotion Prediction for a Sentence

Sentence: "I feel so happy and excited today!"

Steps:

1. Tokenize
   ['i', 'feel', 'so', 'happy', 'and', 'excited', 'today']

2. Remove stopwords
   ['happy', 'excited', 'today']

3. Create feature dictionary
   {'happy': 1, 'excited': 1, 'today': 1}

4. Compute `score_emotion` for each class

| Emotion  | Score Calculation                         | Score |
| -------- | ----------------------------------------- | ----- |
| Joy      | 2(happy) + 1(excited) + 1(today)          | 4     |
| Sadness  | -1(happy) + -0.5(excited) + -0.5(today)   | -2    |
| Anger    | -1(happy) + -0.5(excited) + -0.5(today)   | -2    |
| Love     | -0.5(happy) + -0.5(excited) + -0.5(today) | -1.5  |
| Fear     | -0.5(happy) + -0.5(excited) + -0.5(today) | -1.5  |
| Surprise | -0.5(happy) + -0.5(excited) + -0.5(today) | -1.5  |

Predicted Emotion: Joy (highest score = 4)

---
# Evaluation Metrics Explained

This section provides detailed explanations of all the evaluation metrics used in the training report of the emotion classification model.

---

## 1. Precision

* **Definition:** Measures how many of the predicted instances for a class are actually correct.
* **Formula:**

```math
Precision = True Positives / (True Positives + False Positives)
```

* **Example:**

  * Model predicts 10 texts as `joy`
  * 8 of them are actually `joy`
  * Precision = 8 / 10 = 0.8
* **Interpretation:** High precision means fewer **false positives**.

---

## 2. Recall (Sensitivity or True Positive Rate)

* **Definition:** Measures how many of the actual instances of a class were correctly predicted.
* **Formula:**

```math
Recall = True Positives / (True Positives + False Negatives)
```

* **Example:**

  * There are 12 actual `joy` texts
  * Model correctly predicted 8 of them
  * Recall = 8 / 12 ≈ 0.667
* **Interpretation:** High recall means fewer **false negatives**.

---

## 3. F1-Score

* **Definition:** Harmonic mean of precision and recall, balancing both metrics.
* **Formula:**

```math
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

* **Example:**

  * Precision = 0.8, Recall = 0.667
  * F1 = 2 * (0.8 * 0.667) / (0.8 + 0.667) ≈ 0.727
* **Interpretation:** High F1-score → model is both accurate and complete for that class.

---

## 4. Support

* **Definition:** Number of actual instances of each class in the dataset.
* **Example:**

  * If there are 432 `anger` texts in validation set → support = 432
* **Interpretation:** Helps understand class distribution and performance per class.

---

## 5. Accuracy

* **Definition:** Overall ratio of correct predictions to total predictions.
* **Formula:**

```math
Accuracy = Total Correct Predictions / Total Predictions
```

* **Interpretation:** Overall measure of model correctness.

---

## 6. Macro Average (Macro Avg)

* **Definition:** Average of precision, recall, and F1-score across all classes equally, ignoring class imbalance.
* **Interpretation:** Treats all classes as equally important.

---

## 7. Weighted Average (Weighted Avg)

* **Definition:** Average of precision, recall, and F1-score weighted by number of instances (support) per class.
* **Interpretation:** Reflects performance considering class distribution.

---

## 8. Confusion Matrix

* **Definition:** Table showing the counts of actual vs predicted classes.
* **Structure:** Rows = actual classes, Columns = predicted classes
* **Example:**

| Actual \ Predicted | joy | sadness | anger | love | fear | surprise |
| ------------------ | --- | ------- | ----- | ---- | ---- | -------- |
| joy                | 8   | 3       | 1     | 0    | 0    | 0        |

* **Interpretation:** Helps identify which classes the model confuses. Each cell indicates how many instances of an actual class were predicted as a particular class.

---
# Actual Test Report Discussion

This section explains the training and validation results obtained from the emotion classification model.

---

## Training and Validation Accuracy

* **Training Accuracy:** 0.9817

  * Indicates that the model correctly predicted 98.17% of training examples.
  * High accuracy shows the model has learned the patterns in the training data well.

* **Validation Accuracy:** 0.8978

  * Indicates 89.78% correct predictions on unseen validation data.
  * Slightly lower than training accuracy, showing a good balance and not much overfitting.

---

## Classification Report

| Emotion  | Precision | Recall | F1-Score | Support |
| -------- | --------- | ------ | -------- | ------- |
| anger    | 0.889     | 0.873  | 0.881    | 432     |
| fear     | 0.869     | 0.889  | 0.879    | 387     |
| joy      | 0.901     | 0.927  | 0.914    | 1072    |
| love     | 0.804     | 0.785  | 0.795    | 261     |
| sadness  | 0.937     | 0.929  | 0.933    | 933     |
| surprise | 0.887     | 0.748  | 0.811    | 115     |

### Observations

* **Best performing classes:** `joy` and `sadness` with high precision, recall, and F1-score. The model easily recognizes these due to larger support.
* **Lower performing class:** `surprise` with lower recall (0.748), likely due to fewer examples and overlapping features with other classes.
* **Balanced classes:** `anger` and `fear` show balanced performance (precision ~0.87-0.89, recall ~0.87-0.89).
* **Love:** Moderate performance, possibly due to limited vocabulary and overlap with joy/love sentiments.

---

## Confusion Matrix

```
[[377   7  21   3  24   0]
 [ 11 344   9   2  14   7]
 [ 18   3 994  39  15   3]
 [  2   1  50 205   3   0]
 [ 16  19  24   6 867   1]
 [  0  22   5   0   2  86]]
```

Confusion Matrix with Emotion Labels

```
| Actual \ Predicted | joy | sadness | anger | love | fear | surprise |
| ------------------ | --- | ------- | ----- | ---- | ---- | -------- |
| joy                | 377 | 7       | 21    | 3    | 24   | 0        |
| sadness            | 11  | 344     | 9     | 2    | 14   | 7        |
| anger              | 18  | 3       | 994   | 39   | 15   | 3        |
| love               | 2   | 1       | 50    | 205  | 3    | 0        |
| fear               | 16  | 19      | 24    | 6    | 867  | 1        |
| surprise           | 0   | 22      | 5     | 0    | 2    | 86       |

```


### Interpretation

* **Rows** = Actual emotions
* **Columns** = Predicted emotions
* **Diagonal values** = Correct predictions
* **Off-diagonal values** = Misclassifications

### Example Insights

* 21 `joy` texts misclassified as `anger`
* 22 `surprise` texts misclassified as `sadness`
* Diagonal values indicate how well each class is predicted

---

## Key Takeaways

1. The model is **highly accurate** for common emotions like joy and sadness.
2. Performance drops for **rare or subtle emotions** like surprise and love.
3. Confusion matrix provides insight for **future improvements** such as adding more training examples for underrepresented classes.
4. Overall, the model generalizes well to unseen data with 89.78% validation accuracy.

---

## iOS App Usage

1. Open `EmotionClassifier.xcodeproj` in Xcode 14+
2. Run on simulator/device
3. Enter text → tap **Analyze** → shows predicted emotion & confidence

---

## Tech Stack

* Swift 5 / SwiftUI
* CoreML
* MVVM (async/await, actor)
* Python (Scikit-learn → CoreML)

---

## License

MIT License

---

## Author

* Muhammad Haroon
* Email: mianmharoon72@gmail.com
* LinkedIn: https://www.linkedin.com/in/mian-haroon

---

## References

* [CoreML Tools Documentation](https://coremltools.readme.io/)
* [Emotions Dataset on Kaggle](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp/data)
* [Scikit-learn Documentation](https://scikit-learn.org/)
* [SwiftUI Documentation](https://developer.apple.com/documentation/swiftui)
