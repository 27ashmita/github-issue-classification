# GitHub Issue Classification

This project was developed as part of a prospective MSc research assessment.

The study investigates whether a simple machine learning approach can distinguish **bug reports** from **feature requests** using the textual content of GitHub issues.

## Research Question

**How accurately can a simple machine learning model distinguish bug reports from feature requests using the text of GitHub issues?**

## Selected Project

The issues used in this study were collected from the Microsoft Visual Studio Code (VS Code) GitHub repository.

- GitHub repository: `microsoft/vscode`
- Project category: Developer tool
- Target labels:
  - `bug`
  - `feature-request`

## Dataset

Closed GitHub issues were collected using the GitHub REST API.

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

The GitHub Issues API can also return pull requests. Since this study focuses only on GitHub issues, records containing the `pull_request` field were excluded during data collection.

### Label Quality Check

GitHub issues can contain multiple labels. Therefore, all labels associated with each collected issue were stored in the `all_labels` field.

Additional labels, such as component or release-related labels, were preserved but were not used as classification targets.

A specific check was performed to identify issues containing both target labels:

- `bug`
- `feature-request`

One issue contained both target labels. This issue was removed because its target class was ambiguous for a binary classification task.

The final processed dataset therefore contains:

- 100 bug issues
- 99 feature-request issues
- **199 issues in total**

The final dataset is nearly balanced between the two target classes.

## Data Preparation

The issue title and description were combined into a single text field for classification.

The preprocessing procedure included:

- Checking for duplicate issue numbers
- Removing duplicate issues if present
- Checking for missing descriptions
- Replacing missing descriptions with empty text
- Checking issues with multiple GitHub labels
- Removing the issue containing both target labels
- Combining the title and description
- Removing URLs
- Removing HTML tags
- Removing selected Markdown symbols
- Removing repeated whitespace
- Removing unnecessary spaces at the beginning and end of text
- Checking for empty text after cleaning

Five issues had missing descriptions. These issues were retained because their titles still contained usable textual information.

No duplicate issue numbers were found.

After preprocessing, no issue had an empty cleaned text field.

URLs, HTML tags, and selected Markdown formatting symbols were removed during preprocessing. Complete code blocks and GitHub issue-template content were not explicitly detected or removed as complete structures. Therefore, some code, template text, or technical boilerplate may remain in the cleaned issue text.

## Exploratory Data Analysis

Exploratory data analysis was performed on the processed dataset before model development.

### Dataset Distribution

- Total issues: **199**
- Bug reports: **100**
- Feature requests: **99**

### Issue Length

The average issue length was:

- Overall: **140.61 words**
- Bug reports: **154.44 words**
- Feature requests: **126.64 words**

The longest issue contained **1,046 words**.

The shortest issue contained **2 words**.

The EDA script generates:

- `class_distribution.png`
- `text_length_distribution.png`
- `eda_summary.txt`

These outputs are stored in the `results/` directory.

## Machine Learning Method

A clearly separated training and testing approach was used.

The processed dataset was divided using an **80/20 stratified train-test split** with `random_state=42`.

Stratification was used to maintain a similar class distribution in the training and testing datasets.

### Training Set

- Total: **159 issues**
- Bug: **80**
- Feature request: **79**

### Test Set

- Total: **40 issues**
- Bug: **20**
- Feature request: **20**

The test set was kept separate from model training and was used for final evaluation.

Because the final dataset contains 100 bugs and 99 feature requests, and the test set contains 20 examples from each class, class imbalance is minimal. Therefore, the 50% majority-class baseline provides a meaningful comparison, and overall accuracy is less likely to be misleading because of class imbalance.

## TF-IDF Feature Extraction

TF-IDF was used to convert the cleaned GitHub issue text into numerical features that could be used by the machine learning classifier.

English stop words were removed during TF-IDF processing.

The TF-IDF vectorizer was fitted **only on the training data**. The fitted vectorizer was then used to transform the test data.

This prevents the test data from influencing the vocabulary learned during training.

The resulting TF-IDF representations were:

- Training TF-IDF shape: `(159, 3649)`
- Testing TF-IDF shape: `(40, 3649)`
- Number of TF-IDF features: **3,649**

## Models

Two classification approaches were evaluated.

### Baseline 1: Majority-Class Classifier

A majority-class classifier was used as the simplest baseline.

The classifier predicts the most frequent class in the training dataset for every test example.

The training dataset contained:

- 80 bugs
- 79 feature requests

Therefore, `bug` was the majority class.

The majority-class baseline achieved:

- **Accuracy: 0.50**

### Baseline 2: TF-IDF + Logistic Regression

The second approach used TF-IDF text features with a Logistic Regression classifier.

Logistic Regression was selected because it is a standard linear classification algorithm that works well with high-dimensional sparse text representations such as TF-IDF. It also provides a simple and interpretable baseline for text classification.

The model was configured with:

- `max_iter=1000`
- `random_state=42`

## Model Evaluation

The Logistic Regression classifier achieved:

- **Accuracy: 0.70**
- **Precision for bug: 0.70**
- **Recall for bug: 0.70**
- **F1-score for bug: 0.70**
- **Precision for feature request: 0.70**
- **Recall for feature request: 0.70**
- **F1-score for feature request: 0.70**

Because both classes obtained the same values on this test set, the macro precision, recall, and F1-score were also **0.70**.

### Model Accuracy

| Model | Accuracy |
|---|---:|
| Majority-Class Baseline | 0.50 |
| TF-IDF + Logistic Regression | 0.70 |

The Logistic Regression classifier improved accuracy by **20 percentage points** compared with the majority-class baseline.

### Per-Class Performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Bug | 0.70 | 0.70 | 0.70 | 20 |
| Feature Request | 0.70 | 0.70 | 0.70 | 20 |

Performance was identical for the two classes on this particular test set.

### Confusion Matrix

```text
[[14, 6],
 [ 6, 14]]
```

Using the class order `bug`, `feature_request`, this means:

- 14 bug reports were correctly classified as bugs.
- 6 bug reports were incorrectly classified as feature requests.
- 14 feature requests were correctly classified as feature requests.
- 6 feature requests were incorrectly classified as bugs.

Overall, the classifier correctly classified **28 of the 40 test issues** and incorrectly classified **12**.

## Error Analysis

All **12 incorrectly classified test issues** were manually examined.

The error analysis identified several possible reasons for incorrect predictions:

- Similar language between bugs and feature requests
- Ambiguous issue descriptions
- Short or incomplete descriptions
- Important information contained mainly in screenshots
- Potential ambiguity or inconsistency in GitHub labels
- GitHub issue-template and formatting noise
- Limited training data

Some feature requests described problems with existing software behavior and therefore appeared similar to bug reports.

Similarly, some bug reports used words such as *should* and *show*, which may also appear frequently in feature requests.

Some issues contained very little textual information, while others relied heavily on screenshots. Since the classifier uses only text, information contained in images was unavailable to the model.

The manual review also considered whether the original GitHub label appeared ambiguous, which words or phrases may have influenced the prediction, and how the classification approach could potentially be improved.

The incorrectly classified issues are stored in:

`results/misclassified_issues.csv`

A detailed manual analysis of the errors is included in the research report.

## Project Structure

```text
github-issue-classification/
├── data/
│   ├── raw/
│   │   └── vscode_closed_issues.csv
│   └── processed/
│       └── vscode_processed_issues.csv
│
├── results/
│   ├── class_distribution.png
│   ├── text_length_distribution.png
│   ├── confusion_matrix.png
│   ├── misclassified_issues.csv
│   ├── model_metrics.txt
│   └── eda_summary.txt
│
├── src/
│   ├── collect_issues.py
│   ├── clean_data.py
│   ├── eda.py
│   └── train_model.py
│
├── README.md
└── requirements.txt
```

## Requirements

The project was developed using Python 3.

The main Python libraries are:

- `pandas`
- `requests`
- `scikit-learn`
- `matplotlib`

The required Python packages are listed in `requirements.txt`.

A Python virtual environment is recommended.

### Create a Virtual Environment

From the project root:

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### Install Dependencies

Install the required libraries using:

```bash
pip install -r requirements.txt
```

## Reproducing the Reported Results

The exact raw dataset used for the reported experiment is included in:

```text
data/raw/vscode_closed_issues.csv
```

To reproduce the results reported in this study, use the supplied raw dataset rather than collecting new GitHub issues.

Run the following commands from the **root directory of the project**.

### Step 1: Clean and Prepare the Dataset

```bash
python src/clean_data.py
```

This script:

- Loads the supplied raw dataset
- Checks for issues containing both target labels
- Removes the ambiguous issue containing both `bug` and `feature-request`
- Checks and removes duplicate issue numbers
- Handles missing descriptions
- Combines issue titles and descriptions
- Cleans the combined text
- Checks for empty text after cleaning
- Saves the processed dataset

The processed dataset is saved to:

```text
data/processed/vscode_processed_issues.csv
```

The expected final dataset contains:

- 199 issues
- 100 bugs
- 99 feature requests

### Step 2: Run Exploratory Data Analysis

```bash
python src/eda.py
```

This script examines:

- Dataset size
- Class distribution
- Issue text length
- Average issue length
- Average issue length by class

It generates:

```text
results/class_distribution.png
results/text_length_distribution.png
results/eda_summary.txt
```

Expected descriptive results include:

- Average issue length: **140.61 words**
- Average bug length: **154.44 words**
- Average feature-request length: **126.64 words**
- Longest issue: **1,046 words**
- Shortest issue: **2 words**

### Step 3: Train and Evaluate the Models

```bash
python src/train_model.py
```

This script:

- Creates the stratified 80/20 train-test split
- Fits TF-IDF only on the training data
- Transforms the test data using the fitted vectorizer
- Evaluates the majority-class baseline
- Trains the Logistic Regression classifier
- Calculates accuracy, precision, recall, and F1-score
- Generates the confusion matrix
- Identifies incorrectly classified issues
- Saves the evaluation outputs

The expected results are:

```text
Final dataset size: 199

Training samples: 159
Testing samples: 40

Training class distribution:
bug                80
feature_request    79

Testing class distribution:
bug                20
feature_request    20

Number of TF-IDF features: 3649

Majority Classifier Baseline
Accuracy: 0.50

Logistic Regression
Accuracy: 0.70

Bug:
Precision: 0.70
Recall: 0.70
F1-score: 0.70

Feature Request:
Precision: 0.70
Recall: 0.70
F1-score: 0.70

Confusion Matrix:
[[14  6]
 [ 6 14]]

Number of misclassified issues: 12
```

The main model outputs are:

```text
results/model_metrics.txt
results/confusion_matrix.png
results/misclassified_issues.csv
```

Using the supplied dataset and these scripts reproduces the reported experimental pipeline.

## Optional: Collect a New Dataset

The repository also includes `src/collect_issues.py`, which can be used to collect closed issues from the live GitHub REST API.

To collect a new dataset:

```bash
python src/collect_issues.py
```

The script:

- Connects to the GitHub REST API
- Collects closed issues labelled `bug`
- Collects closed issues labelled `feature-request`
- Excludes pull requests
- Stores the relevant issue information
- Preserves all GitHub labels
- Saves the raw dataset

The newly collected raw dataset is saved to:

```text
data/raw/vscode_closed_issues.csv
```

> **Reproducibility note:** The GitHub repository changes over time, so running `collect_issues.py` again may collect a different set of issues. The supplied `data/raw/vscode_closed_issues.csv` contains the exact dataset used for the reported experiment.

## Main Findings

The majority-class baseline achieved **50% accuracy**.

The TF-IDF with Logistic Regression classifier achieved **70% accuracy**.

Both bug reports and feature requests achieved:

- Precision: **0.70**
- Recall: **0.70**
- F1-score: **0.70**

The model therefore improved accuracy by **20 percentage points** compared with the majority-class baseline.

The confusion matrix showed that the model correctly classified:

- 14 of 20 bug reports
- 14 of 20 feature requests

The model incorrectly classified:

- 6 bug reports as feature requests
- 6 feature requests as bugs

These results suggest that GitHub issue text contains useful information for distinguishing bug reports from feature requests.

However, the dataset and test set are small. Therefore, the results should be interpreted as preliminary evidence rather than as a strong general conclusion about GitHub issue classification.

## Limitations

This is a small empirical study using issues from only one GitHub repository.

Important limitations include:

- The final dataset contains only 199 issues.
- Only the VS Code repository was studied.
- The test set contains only 40 issues.
- GitHub labels may contain ambiguity or inconsistency.
- Some issues contain very little textual information.
- Some issues rely on screenshots or other non-text information.
- The classifier cannot interpret information contained in images.
- Basic preprocessing may leave GitHub template, code, or technical boilerplate in the text.
- Only one standard text classifier was evaluated in addition to the majority-class baseline.
- A single train-test split was used instead of cross-validation.

Future work could investigate:

- Larger datasets
- Multiple GitHub repositories
- Improved removal of templates, code, and boilerplate
- Word and character n-grams
- Five-fold cross-validation
- Additional machine learning classifiers
- Contextual language models
- Cross-project evaluation
- Temporal evaluation
- More systematic validation of GitHub labels

## Generative AI Disclosure

Generative AI was used as a supporting tool during this assessment.

ChatGPT was used to assist with:

- Explaining Python and machine learning concepts
- Reviewing code structure and debugging
- Discussing preprocessing decisions
- Explaining TF-IDF and Logistic Regression
- Interpreting evaluation metrics
- Supporting the organization and interpretation of manual error analysis
- Organizing and revising project documentation
- Organizing and revising the research report and presentation materials

I personally executed the data collection, preprocessing, exploratory analysis, model training, and evaluation scripts. I reviewed the collected and processed datasets, generated outputs, evaluation results, and all 12 misclassified test issues.

I also reviewed, verified, and modified AI-assisted explanations and documentation before including them in the final submission. I remain responsible for the correctness of the submitted code, analysis, report, and supporting materials.

A detailed Generative AI disclosure, including representative prompts, the tasks for which AI was used, and the parts personally reviewed, verified, or modified, is provided in the research report appendix.

## Reproducibility

The experiment was rerun using the supplied raw dataset before finalization.

The rerun reproduced:

- 199 final issues
- 159 training issues
- 40 testing issues
- 3,649 TF-IDF features
- 0.50 majority-class baseline accuracy
- 0.70 Logistic Regression accuracy
- 0.70 precision for each class
- 0.70 recall for each class
- 0.70 F1-score for each class
- 12 incorrectly classified test issues

This confirms that the reported experimental results can be reproduced from the supplied dataset and source code.

