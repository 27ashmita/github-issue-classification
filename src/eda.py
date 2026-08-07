import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


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


# ---------------------------------
# DATASET OVERVIEW
# ---------------------------------

print("=" * 50)
print("Dataset Overview")
print("=" * 50)

print("\nRows and Columns:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ---------------------------------
# CLASS DISTRIBUTION
# ---------------------------------

print("\nClass Distribution:")
print(df["label"].value_counts())


# ---------------------------------
# ISSUE TEXT LENGTH
# ---------------------------------

# Count the number of words in each cleaned issue
df["text_length"] = df["clean_text"].apply(
    lambda text: len(str(text).split())
)

average_length = df["text_length"].mean()
maximum_length = df["text_length"].max()
minimum_length = df["text_length"].min()

print("\nAverage Issue Length:")
print(f"{average_length:.2f} words")

print("\nLongest Issue:")
print(f"{maximum_length} words")

print("\nShortest Issue:")
print(f"{minimum_length} words")


# ---------------------------------
# AVERAGE LENGTH BY CLASS
# ---------------------------------

average_by_class = (
    df.groupby("label")["text_length"]
    .mean()
)

print("\nAverage Issue Length by Class:")
print(average_by_class.round(2))


# ---------------------------------
# CLASS DISTRIBUTION GRAPH
# ---------------------------------

class_counts = df["label"].value_counts()

plt.figure(figsize=(6, 4))

plt.bar(
    class_counts.index,
    class_counts.values
)

plt.title("Issue Class Distribution")
plt.xlabel("Issue Type")
plt.ylabel("Number of Issues")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "class_distribution.png"
)

plt.close()


# ---------------------------------
# TEXT LENGTH HISTOGRAM
# ---------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["text_length"],
    bins=20
)

plt.title("Issue Text Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Issues")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "text_length_distribution.png"
)

plt.close()


# ---------------------------------
# SAVE EDA SUMMARY
# ---------------------------------

SUMMARY_FILE = RESULTS_DIR / "eda_summary.txt"

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write("EDA SUMMARY\n")
    file.write("=" * 40 + "\n\n")

    file.write(
        f"Dataset Shape: {df.shape}\n\n"
    )

    file.write("Class Distribution:\n")
    file.write(
        df["label"].value_counts().to_string()
    )

    file.write("\n\n")

    file.write(
        f"Average Issue Length: "
        f"{average_length:.2f} words\n"
    )

    file.write(
        f"Maximum Issue Length: "
        f"{maximum_length} words\n"
    )

    file.write(
        f"Minimum Issue Length: "
        f"{minimum_length} words\n\n"
    )

    file.write(
        "Average Issue Length by Class:\n"
    )

    file.write(
        average_by_class.round(2).to_string()
    )


# ---------------------------------
# COMPLETION MESSAGE
# ---------------------------------

print("\nEDA completed successfully.")

print("\nGenerated files:")
print("- class_distribution.png")
print("- text_length_distribution.png")
print("- eda_summary.txt")

print("\nResults saved in:")
print(RESULTS_DIR)