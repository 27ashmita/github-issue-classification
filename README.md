# GitHub Issue Classification

This project is part of a prospective MSc research assessment.

## Research Question

How accurately can a simple machine learning model distinguish bug reports
from feature requests using the text of GitHub issues?

## Selected Project

- GitHub repository: `microsoft/vscode`
- Project category: Developer tool
- Target labels:
  - `bug`
  - `feature-request`

## Dataset

Closed GitHub issues were collected from the Microsoft Visual Studio Code
(VS Code) repository using the GitHub REST API.

The initial dataset contained:

- 100 bug issues
- 100 feature-request issues
- 200 issues in total

For each issue, the following information was collected:

- Issue number
- Issue title
- Issue description
- Target label
- All GitHub labels
- Creation date
- Closing date
- Issue URL

Pull requests returned by the GitHub Issues API were excluded.

During label-quality checking, one issue was found to contain both
`bug` and `feature-request` labels. This issue was removed because its
target class was ambiguous.

The final processed dataset contains:

- 100 bug issues
- 99 feature-request issues
- 199 issues in total

## Data Preparation

The issue title and description were combined into one text field.

The preprocessing procedure included:

- Checking and removing duplicate issue numbers
- Replacing missing descriptions with empty text
- Checking issues with multiple GitHub labels
- Removing the issue containing both target labels
- Removing URLs
- Removing HTML tags
- Removing selected Markdown symbols
- Removing unnecessary whitespace
- Checking for empty text after cleaning

Five issues had missing descriptions. Their titles were retained so that
the issues could still be used.

No duplicate issue numbers were found.

## Exploratory Data Analysis

The final dataset contains 199 issues.

Average issue length:

- Overall: 140.61 words
- Bug: 154.44 words
- Feature request: 126.64 words

The longest issue contained 1,046 words and the shortest contained 2 words.

EDA outputs are stored in the `results/` directory.

## Machine Learning Method

The processed dataset was divided using an 80/20 stratified train-test split
with `random_state=42`.

This produced:

- 159 training issues
- 40 testing issues

Training distribution:

- 80 bugs
- 79 feature requests

Testing distribution:

- 20 bugs
- 20 feature requests

### TF-IDF

TF-IDF was used to convert the cleaned issue text into numerical features.

The TF-IDF vectorizer was fitted only on the training data and then used
to transform the test data.

The final training representation contained 3,649 TF-IDF features.

## Models

Two models were evaluated.

### Baseline 1: Majority-Class Classifier

The majority-class classifier always predicts the most frequent class in
the training dataset.

Accuracy:

`0.50`

### Baseline 2: TF-IDF + Logistic Regression

Logistic Regression was trained using the TF-IDF features.

Results:

- Accuracy: 0.70
- Precision: 0.70
- Recall: 0.70
- F1-score: 0.70

Per-class results:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Bug | 0.70 | 0.70 | 0.70 |
| Feature Request | 0.70 | 0.70 | 0.70 |

Confusion matrix:

    [[14, 6],
     [6, 14]]

The Logistic Regression model correctly classified 28 of the 40 test
issues and incorrectly classified 12.

## Error Analysis

All 12 incorrectly classified test issues were manually examined.

The analysis identified several possible reasons for classification errors:

- Similar language between bugs and feature requests
- Short or incomplete descriptions
- Important information contained in screenshots
- Ambiguous GitHub labels
- GitHub issue-template and formatting noise
- Small training dataset

The incorrectly classified issues are stored in:

`results/misclassified_issues.csv`

## Project Structure

    github-issue-classification/
    ├── data/
    │   ├── raw/
    │   │   └── vscode_closed_issues.csv
    │   └── processed/
    │       └── vscode_processed_issues.csv
    ├── results/
    │   ├── class_distribution.png
    │   ├── text_length_distribution.png
    │   ├── confusion_matrix.png
    │   ├── misclassified_issues.csv
    │   ├── model_metrics.txt
    │   └── eda_summary.txt
    ├── src/
    │   ├── collect_issues.py
    │   ├── clean_data.py
    │   ├── eda.py
    │   └── train_model.py
    ├── report/
    ├── slides/
    └── README.md

## Requirements

The project was developed using Python 3.

Main Python libraries:

- pandas
- requests
- scikit-learn
- matplotlib

Install the required libraries using:

    pip install pandas requests scikit-learn matplotlib

## Reproducing the Experiment

Run the following commands from the project root directory.

### 1. Collect GitHub issues

    python src/collect_issues.py

This creates:

    data/raw/vscode_closed_issues.csv

### 2. Clean and prepare the dataset

    python src/clean_data.py

This creates:

    data/processed/vscode_processed_issues.csv

### 3. Run exploratory data analysis

    python src/eda.py

This generates the EDA results and figures in:

    results/

### 4. Train and evaluate the classifier

    python src/train_model.py

This trains the majority baseline and Logistic Regression classifier and
generates the evaluation results, confusion matrix, and misclassified
issues.

## Limitations

This is a small empirical study using issues from only one GitHub repository.

The final dataset contains 199 issues, so the results should not be used
to make strong general conclusions.

GitHub labels may also contain ambiguity or inconsistency.

Future work could include:

- A larger dataset
- Multiple GitHub repositories
- Improved preprocessing
- Word and character n-grams
- Cross-validation
- Additional machine learning classifiers
- Contextual language models
- Cross-project evaluation

## AI Disclosure

Generative AI (ChatGPT) was used as a supporting tool during the project.

It was used for:

- Explaining Python code and machine learning concepts
- Discussing data preparation decisions
- Explaining TF-IDF and Logistic Regression
- Interpreting evaluation metrics
- Supporting manual error analysis
- Assisting with the organization and wording of documentation and the report

The code was executed by the author, and the datasets, model outputs,
evaluation results, and generated explanations were reviewed and verified
before submission.

## Status

The empirical study, model evaluation, and error analysis are complete.

Final report preparation and presentation preparation are in progress.