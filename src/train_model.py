import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------
# PROJECT PATHS
# ---------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vscode_processed_issues.csv"
)


# ---------------------------------
# LOAD DATASET
# ---------------------------------

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ---------------------------------
# DEFINE INPUT AND TARGET
# ---------------------------------

X = df["clean_text"]
y = df["label"]

print("\nClass distribution:")
print(y.value_counts())


# ---------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:")
print(len(X_train))

print("\nTesting samples:")
print(len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# ---------------------------------
# TF-IDF
# ---------------------------------

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)


# ---------------------------------
# CHECK TF-IDF OUTPUT
# ---------------------------------

print("\nTF-IDF completed.")

print("\nTraining TF-IDF shape:")
print(X_train_tfidf.shape)

print("\nTesting TF-IDF shape:")
print(X_test_tfidf.shape)

print("\nNumber of TF-IDF features:")
print(len(tfidf.get_feature_names_out()))