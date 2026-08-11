import re
import pandas as pd
from pathlib import Path


# Find the project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input and output files
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "vscode_closed_issues.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "vscode_processed_issues.csv"

# Create output directory if it doesn't exist
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


def clean_text(text):
    """
    Clean GitHub issue text by removing URLs,
    HTML tags, markdown symbols, and extra spaces.
    """

    # Convert missing values to empty string
    if pd.isna(text):
        return ""

    # Convert to string
    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove markdown symbols
    text = re.sub(r"[#>*_`~\-]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove spaces at beginning and end
    text = text.strip()

    return text


# -----------------------------
# Load raw dataset
# -----------------------------

df = pd.read_csv(INPUT_FILE)

print("Original dataset size:")
print(df.shape)


# -----------------------------
# Remove conflicting target labels
# -----------------------------

labels = df["all_labels"].fillna("").str.lower()

conflicting_labels = (
    labels.str.contains(
        r"(?:^|,)bug(?:,|$)",
        regex=True
    )
    & labels.str.contains(
        r"(?:^|,)feature-request(?:,|$)",
        regex=True
    )
)

conflict_count = conflicting_labels.sum()

print("\nIssues with both bug and feature-request labels:")
print(conflict_count)

df = df[
    ~conflicting_labels
].reset_index(drop=True)

# -----------------------------
# Check duplicates
# -----------------------------

duplicate_count = df["issue_number"].duplicated().sum()

print("\nNumber of duplicate issues:")
print(duplicate_count)

# Remove duplicate issue numbers
df = df.drop_duplicates(
    subset="issue_number"
).reset_index(drop=True)


# -----------------------------
# Handle missing descriptions
# -----------------------------

missing_descriptions = df["description"].isna().sum()

print("\nMissing descriptions:")
print(missing_descriptions)

# Replace missing descriptions with empty text
df["description"] = df["description"].fillna("")


# -----------------------------
# Combine title and description
# -----------------------------

df["text"] = (
    df["title"].fillna("")
    + " "
    + df["description"]
)


# -----------------------------
# Clean combined text
# -----------------------------

df["clean_text"] = df["text"].apply(clean_text)


# -----------------------------
# Check empty cleaned text
# -----------------------------

empty_text_count = (
    df["clean_text"]
    .str.strip()
    .eq("")
    .sum()
)

print("\nEmpty text rows after cleaning:")
print(empty_text_count)


# Remove rows with no usable text
df = df[
    df["clean_text"].str.strip() != ""
].reset_index(drop=True)


# -----------------------------
# Save processed dataset
# -----------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# -----------------------------
# Final summary
# -----------------------------

print("\nData cleaning completed.")

print("\nFinal dataset size:")
print(df.shape)

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nProcessed dataset saved to:")
print(OUTPUT_FILE)