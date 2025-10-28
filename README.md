# SentimentAnalysis_CoreMLDemo_EmotionClassifier_iOS_SwiftUI

This is an **iOS demo application** for **emotion classification** using **CoreML** and **SwiftUI**.
The project follows **MVVM architecture** with **async/await** and **actor-based model management**.

The **CoreML model** was converted from a **Scikit-learn pipeline** using **coremltools Python package**, including:

* `DictVectorizer` for feature extraction
* `LinearSVC` for classification

![WhatsApp Image 2025-10-28 at 8 17 33 PM (4)](https://github.com/user-attachments/assets/dbb7ede8-a229-4cfe-9a94-1a3873fcbd92)


---

## Dataset

The training dataset is from [Kaggle: Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp/data).

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

This document explains how the emotion classification model learns from training data using a step-by-step example with multiple iterations.

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

---

## Step 4: Prediction Formula

When a new sentence comes in, the model computes:

```
score_emotion = sum(word_count * weight_for_that_emotion)
```

## Example Prediction with Iterations

Input:

```
I feel so happy and excited today!
```

**Processing steps:**

1. Tokenize: `['i', 'feel', 'so', 'happy', 'and', 'excited', 'today']`
2. Remove stopwords: `['happy', 'excited', 'today']`
3. Feature dictionary: `{'happy':1, 'excited':1, 'today':1}`

**Compute `score_emotion` for each class:**

* joy: 2(happy)+1(excited)+1(today) = 4 → joy score = 4
* sadness: -1 + -0.5 + -0.5 = -2
* anger: -1 + -0.5 + -0.5 = -2
* love: -0.5 + -0.2 + -0.2 = -0.9
* fear: -0.5 + -0.2 + -0.2 = -0.9
* surprise: -0.5 + -0.2 + -0.2 = -0.9

**Predicted emotion → joy** ✅

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
