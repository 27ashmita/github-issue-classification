import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


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

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
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


# ---------------------------------
# BASELINE MODEL: MAJORITY CLASSIFIER
# ---------------------------------

baseline_model = DummyClassifier(
    strategy="most_frequent"
)

baseline_model.fit(
    X_train_tfidf,
    y_train
)

baseline_predictions = baseline_model.predict(
    X_test_tfidf
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

print("\nMajority Classifier Baseline")

print("Accuracy:")
print(f"{baseline_accuracy:.2f}")


# ---------------------------------
# LOGISTIC REGRESSION MODEL
# ---------------------------------

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_tfidf,
    y_train
)

logistic_predictions = logistic_model.predict(
    X_test_tfidf
)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

print("\nLogistic Regression")
print("Accuracy:")
print(f"{logistic_accuracy:.2f}")

# ---------------------------------
# MODEL EVALUATION
# ---------------------------------

precision = precision_score(
    y_test,
    logistic_predictions,
    pos_label="bug"
)

recall = recall_score(
    y_test,
    logistic_predictions,
    pos_label="bug"
)

f1 = f1_score(
    y_test,
    logistic_predictions,
    pos_label="bug"
)

print("\nModel Evaluation")
print("-------------------------")

print(f"Accuracy:  {logistic_accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-score:  {f1:.2f}")


# ---------------------------------
# CONFUSION MATRIX
# ---------------------------------

cm = confusion_matrix(
    y_test,
    logistic_predictions,
    labels=["bug", "feature_request"]
)

print("\nConfusion Matrix:")
print(cm)

# ---------------------------------
# SAVE CONFUSION MATRIX
# ---------------------------------

fig, ax = plt.subplots(figsize=(6, 5))

image = ax.imshow(cm)

ax.set_title("Logistic Regression Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")

ax.set_xticks([0, 1])
ax.set_xticklabels(["Bug", "Feature Request"])

ax.set_yticks([0, 1])
ax.set_yticklabels(["Bug", "Feature Request"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()


# ---------------------------------
# CLASSIFICATION REPORT
# ---------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        logistic_predictions,
        digits=2
    )
)

# ---------------------------------
# ERROR ANALYSIS
# ---------------------------------

test_results = df.loc[X_test.index].copy()

test_results["predicted_label"] = logistic_predictions

# Select only incorrectly classified issues
misclassified = test_results[
    test_results["label"] != test_results["predicted_label"]
].copy()


# ---------------------------------
# SAVE MISCLASSIFIED ISSUES
# ---------------------------------

ERROR_FILE = (
    PROJECT_ROOT
    / "results"
    / "misclassified_issues.csv"
)

ERROR_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

misclassified[
    [
        "issue_number",
        "title",
        "label",
        "predicted_label",
        "url"
    ]
].to_csv(
    ERROR_FILE,
    index=False
)


# ---------------------------------
# PRINT ERROR ANALYSIS
# ---------------------------------

print("\nError Analysis")
print("-------------------------")

print("Number of misclassified issues:")
print(len(misclassified))

print("\nMisclassified Issues:")

for _, row in misclassified.iterrows():

    print("\n" + "=" * 60)

    print("Issue Number:")
    print(row["issue_number"])

    print("\nTitle:")
    print(row["title"])

    print("\nActual Label:")
    print(row["label"])

    print("\nPredicted Label:")
    print(row["predicted_label"])


print("\nMisclassified issues saved to:")
print(ERROR_FILE)

# ---------------------------------
# SAVE MODEL METRICS
# ---------------------------------

METRICS_FILE = RESULTS_DIR / "model_metrics.txt"

with open(METRICS_FILE, "w") as file:

    file.write("MODEL EVALUATION RESULTS\n")
    file.write("========================\n\n")

    file.write(f"Total dataset size: {len(df)}\n")
    file.write(f"Training samples: {len(X_train)}\n")
    file.write(f"Testing samples: {len(X_test)}\n")
    file.write(f"TF-IDF features: {X_train_tfidf.shape[1]}\n\n")

    file.write("Majority Classifier Baseline\n")
    file.write("----------------------------\n")
    file.write(f"Accuracy: {baseline_accuracy:.2f}\n\n")

    file.write("Logistic Regression\n")
    file.write("-------------------\n")
    file.write(f"Accuracy: {logistic_accuracy:.2f}\n")
    file.write(f"Precision: {precision:.2f}\n")
    file.write(f"Recall: {recall:.2f}\n")
    file.write(f"F1-score: {f1:.2f}\n\n")

    file.write("Confusion Matrix\n")
    file.write("----------------\n")
    file.write(str(cm))
    file.write("\n\n")

    file.write("Classification Report\n")
    file.write("---------------------\n")

    file.write(
        classification_report(
            y_test,
            logistic_predictions,
            digits=2
        )
    )

    file.write("\n")
    file.write(
        f"Number of misclassified issues: {len(misclassified)}\n"
    )


print("\nModel metrics saved to:")
print(METRICS_FILE)

print("\nConfusion matrix saved to:")
print(RESULTS_DIR / "confusion_matrix.png")