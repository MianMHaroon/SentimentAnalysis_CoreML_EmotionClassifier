#"""
#Emotion Classification Model Trainer
#------------------------------------
#Author: Muhammad Haroon
#Version: 1.1
#License: MIT
#
#Description:
#This script trains a simple emotion classifier (joy, sadness, anger, fear)
#using a Linear Support Vector Classifier (LinearSVC) with a DictVectorizer.
#It outputs a CoreML-compatible model (`EmotionClassifier.mlmodel`)
#that can be directly used in iOS apps.
#"""
#
#import re
#import coremltools
#import pandas as pd
#import numpy as np
#import nltk
#from nltk.corpus import stopwords
#from nltk import word_tokenize
#from string import punctuation
#from sklearn.feature_extraction import DictVectorizer
#from sklearn.pipeline import Pipeline
#from sklearn.svm import LinearSVC
#from sklearn.model_selection import GridSearchCV, train_test_split
#
## Ensure NLTK data is available
#nltk.download('punkt')
#nltk.download('stopwords')
#
## ===============================
## 1. Load Dataset
## ===============================
#def load_dataset(path: str):
#    """Load emotion dataset with auto separator detection."""
#    try:
#        # Try comma-separated first
#        data = pd.read_csv(path)
#    except Exception:
#        # Fallback to tab-separated
#        data = pd.read_csv(path, sep="\t", header=None, names=["text", "emotion"])
#
#    # If columns are not found, try to fix automatically
#    if not {"text", "emotion"}.issubset(data.columns):
#        if data.shape[1] == 2:
#            data.columns = ["text", "emotion"]
#        else:
#            raise ValueError("CSV must contain 'text' and 'emotion' columns or have exactly 2 columns.")
#    return data
#
#data = load_dataset("train.csv")
#print(f"✅ Loaded {len(data)} records")
#print(data.head())
#
## ===============================
## 2. Feature Extraction
## ===============================
#def extract_features(sentence: str):
#    """Extract simple bag-of-words features."""
#    stop_words = set(stopwords.words("english")) | set(punctuation)
#    words = word_tokenize(str(sentence))
#    words = [w.lower() for w in words]
#    filtered = [w for w in words if w not in stop_words and not w.isdigit()]
#    
#    word_counts = {}
#    for word in filtered:
#        word_counts[word] = word_counts.get(word, 0.0) + 1.0
#    return word_counts
#
## Split into features/labels
#X = np.array([extract_features(text) for text in data["text"]])
#y = data["emotion"].values
#
## Split into training/testing for validation
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.2, random_state=42, stratify=y
#)
#
## ===============================
## 3. Model Pipeline + Grid Search
## ===============================
#print("\n🔧 Training Model... (this may take a few minutes)")
#
#pipeline = Pipeline([
#    ("vectorizer", DictVectorizer()),
#    ("classifier", LinearSVC())
#])
#
#params = {
#    "classifier__C": [1e3, 1e2, 1e1, 1, 1e-1]
#}
#
#grid = GridSearchCV(pipeline, params, cv=5, verbose=2, n_jobs=-1)
#grid.fit(X_train, y_train)
#
#model = grid.best_estimator_
#
#print("\n✅ Model Training Completed!")
#print(f"Best Parameters: {grid.best_params_}")
#print(f"Training Accuracy: {model.score(X_train, y_train):.4f}")
#print(f"Validation Accuracy: {model.score(X_test, y_test):.4f}")
#
## ===============================
## 4. Convert to CoreML
## ===============================
#print("\n📦 Converting to CoreML format...")
#
#coreml_model = coremltools.converters.sklearn.convert(model)
#coreml_model.author = "Muhammad Haroon"
#coreml_model.license = "MIT"
#coreml_model.short_description = "Emotion classification model (joy, sadness, anger, fear)"
#coreml_model.version = "1.0"
#coreml_model.input_description["input"] = "Features extracted from input text."
#coreml_model.output_description["classLabel"] = "Predicted emotion label."
#
#coreml_model.save("EmotionClassifier.mlmodel")
#
#print("\n🎉 Model successfully saved as EmotionClassifier.mlmodel")
#


"""
Emotion Classification Model Trainer
------------------------------------
Author: Muhammad Haroon
Version: 1.2
License: MIT

Description:
This script trains a simple emotion classifier (joy, sadness, anger, fear)
using a Linear Support Vector Classifier (LinearSVC) with a DictVectorizer.
It outputs:
- EmotionClassifier.mlmodel (for iOS apps)
- training_report.txt (for evaluation summary)
"""

import re
import coremltools
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from string import punctuation
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Ensure NLTK data is available
nltk.download('punkt')
nltk.download('stopwords')

# ===============================
# 1. Load Dataset
# ===============================
def load_dataset(path: str):
    """Load emotion dataset with auto separator detection."""
    try:
        data = pd.read_csv(path)
    except Exception:
        data = pd.read_csv(path, sep="\t", header=None, names=["text", "emotion"])

    if not {"text", "emotion"}.issubset(data.columns):
        if data.shape[1] == 2:
            data.columns = ["text", "emotion"]
        else:
            raise ValueError("CSV must contain 'text' and 'emotion' columns or have exactly 2 columns.")
    return data

data = load_dataset("train.csv")
print(f"✅ Loaded {len(data)} records")
print(data.head())

# ===============================
# 2. Feature Extraction
# ===============================
def extract_features(sentence: str):
    """Extract simple bag-of-words features."""
    stop_words = set(stopwords.words("english")) | set(punctuation)
    words = word_tokenize(str(sentence))
    words = [w.lower() for w in words]
    filtered = [w for w in words if w not in stop_words and not w.isdigit()]
    
    word_counts = {}
    for word in filtered:
        word_counts[word] = word_counts.get(word, 0.0) + 1.0
    return word_counts

# Prepare features
X = np.array([extract_features(text) for text in data["text"]])
y = data["emotion"].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 3. Model Pipeline + Grid Search
# ===============================
print("\n🔧 Training Model... (this may take a few minutes)")

pipeline = Pipeline([
    ("vectorizer", DictVectorizer()),
    ("classifier", LinearSVC())
])

params = {
    "classifier__C": [1e3, 1e2, 1e1, 1, 1e-1]
}

grid = GridSearchCV(pipeline, params, cv=5, verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

model = grid.best_estimator_

# ===============================
# 4. Evaluation & Reporting
# ===============================
print("\n✅ Model Training Completed!")
train_acc = model.score(X_train, y_train)
val_acc = model.score(X_test, y_test)

print(f"Best Parameters: {grid.best_params_}")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

# Detailed classification report
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, digits=3)
matrix = confusion_matrix(y_test, y_pred)

print("\n📊 Classification Report:\n", report)
print("🧩 Confusion Matrix:\n", matrix)

# Save report
with open("training_report.txt", "w") as f:
    f.write("Emotion Classification Report\n")
    f.write("----------------------------------\n")
    f.write(f"Training Accuracy: {train_acc:.4f}\n")
    f.write(f"Validation Accuracy: {val_acc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(matrix))
print("\n📝 Saved training report to training_report.txt")

# ===============================
# 5. Convert to CoreML
# ===============================
print("\n📦 Converting to CoreML format...")

coreml_model = coremltools.converters.sklearn.convert(model)
coreml_model.author = "Muhammad Haroon"
coreml_model.license = "MIT"
coreml_model.short_description = "Emotion classification model (joy, sadness, anger, fear)"
coreml_model.version = "1.0"
coreml_model.input_description["input"] = "Features extracted from input text."
coreml_model.output_description["classLabel"] = "Predicted emotion label."

coreml_model.save("EmotionClassifier.mlmodel")

print("\n🎉 Model successfully saved as EmotionClassifier.mlmodel")
