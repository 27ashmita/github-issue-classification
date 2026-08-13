# GitHub Issue Classification

This project was developed as part of a prospective MSc research assessment.

The study investigates whether a simple machine learning approach can distinguish **bug reports** from **feature requests** using the textual content of GitHub issues.

## Research Question

**How accurately can a simple machine learning model distinguish bug reports from feature requests using the text of GitHub issues?**

## Selected Project

The issues used in this study were collected from the Microsoft Visual Studio Code (VS Code) GitHub repository.

* GitHub repository: `microsoft/vscode`
* Project category: Developer tool
* Target labels:

  * `bug`
  * `feature-request`

## Dataset

Closed GitHub issues were collected using the GitHub REST API.

The initial dataset contained:

* 100 bug issues
* 100 feature-request issues
* 200 issues in total

For each issue, the following information was collected:

* Issue number
* Issue title
* Issue description
* Target label
* All GitHub labels
* Creation date
* Closing date
* Issue URL

The GitHub Issues API can also return pull requests. Since this study focuses only on GitHub issues, records containing the `pull_request` field were excluded during data collection.

### Label Quality Check

GitHub issues can contain multiple labels. Therefore, all labels associated with each collected issue were stored in the `all_labels` field.

Additional labels such as component or release-related labels were preserved but were not used as classification targets.

A specific check was performed to identify issues containing both target labels:

* `bug`
* `feature-request`

One issue contained both target labels. This issue was removed because its target class was ambiguous for a binary classification task.

The final processed dataset therefore contains:

* 100 bug issues
* 99 feature-request issues
* **199 issues in total**

The final dataset is nearly balanced between the two target classes.

## Data Preparation

The issue title and description were combined into a single text field for classification.

The preprocessing procedure included:

* Checking for duplicate issue numbers
* Removing duplicate issues if present
* Checking for missing descriptions
* Replacing missing descriptions with empty text
* Checking issues with multiple GitHub labels
* Removing the issue containing both target labels
* Combining the title and description
* Removing URLs
* Removing HTML tags
* Removing selected Markdown symbols
* Removing repeated whitespace
* Removing unnecessary spaces at the beginning and end of text
* Checking for empty text after cleaning

Five issues had missing descriptions. These issues were retained because their titles still contained usable textual information.

No duplicate issue numbers were found.

After preprocessing, no issue had an empty cleaned text field.

## Exploratory Data Analysis

Exploratory data analysis was performed on the processed dataset before model development.

### Dataset Distribution

* Total issues: 199
* Bug reports: 100
* Feature requests: 99

### Issue Length

The average issue length was:

* Overall: **140.61 words**
* Bug reports: **154.44 words**
* Feature requests: **126.64 words**

The longest issue contained **1,046 words**.

The shortest issue contained **2 words**.

The EDA script generates:

* `class_distribution.png`
* `text_length_distribution.png`
* `eda_summary.txt`

These outputs are stored in the `results/` directory.

## Machine Learning Method

A clearly separated training and testing approach was used.

The processed dataset was divided using an **80/20 stratified train-test split** with `random_state=42`.

Stratification was used to maintain a similar class distribution in the training and testing datasets.

### Training Set

* Total: 159 issues
* Bug: 80
* Feature request: 79

### Test Set

* Total: 40 issues
* Bug: 20
* Feature request: 20

The test set was kept separate from model training and was used for final evaluation.

## TF-IDF Feature Extraction

TF-IDF was used to convert the cleaned GitHub issue text into numerical features that could be used by the machine learning classifier.

English stop words were removed during TF-IDF processing.

The TF-IDF vectorizer was fitted **only on the training data**. The fitted vectorizer was then used to transform the test data.

This prevents the test data from influencing the vocabulary learned during training.

The resulting TF-IDF representations were:

* Training TF-IDF shape: `(159, 3649)`
* Testing TF-IDF shape: `(40, 3649)`
* Number of TF-IDF features: **3,649**

## Models

Two classification approaches were evaluated.

### Baseline 1: Majority-Class Classifier

A majority-class classifier was used as the simplest baseline.

The classifier predicts the most frequent class in the training dataset for every test example.

The training dataset contained:

* 80 bugs
* 79 feature requests

Therefore, `bug` was the majority class.

The majority-class baseline achieved:

* **Accuracy: 0.50**

### Baseline 2: TF-IDF + Logistic Regression

The second approach used TF-IDF text features with a Logistic Regression classifier.

Logistic Regression was selected because it is a standard linear classification algorithm that can work effectively with high-dimensional sparse text representations such as TF-IDF.

The model was configured with:

* `max_iter=1000`
* `random_state=42`

## Model Evaluation

The Logistic Regression classifier achieved:

* **Accuracy: 0.70**
* **Macro Precision: 0.70**
* **Macro Recall: 0.70**
* **Macro F1-score: 0.70**

### Per-Class Performance

| Class           | Precision | Recall | F1-score | Support |
| --------------- | --------: | -----: | -------: | ------: |
| Bug             |      0.70 |   0.70 |     0.70 |      20 |
| Feature Request |      0.70 |   0.70 |     0.70 |      20 |

The model showed the same precision, recall, and F1-score for both classes on this test set.

### Confusion Matrix

```text
[[14, 6],
 [ 6, 14]]
```

This means:

* 14 bug reports were correctly classified as bugs.
* 6 bug reports were incorrectly classified as feature requests.
* 14 feature requests were correctly classified as feature requests.
* 6 feature requests were incorrectly classified as bugs.

Overall, the classifier correctly classified **28 of the 40 test issues** and incorrectly classified **12**.

The Logistic Regression classifier improved accuracy from **50% to 70%**, an absolute improvement of **20 percentage points** compared with the majority-class baseline.

## Error Analysis

All **12 incorrectly classified test issues** were manually examined.

The error analysis identified several possible reasons for incorrect predictions:

* Similar language between bugs and feature requests
* Ambiguous issue descriptions
* Short or incomplete descriptions
* Important information contained mainly in screenshots
* Potential ambiguity or inconsistency in GitHub labels
* GitHub issue-template and formatting noise
* Limited training data

Some feature requests described problems with existing software behavior and therefore appeared similar to bug reports.

Similarly, some bug reports used words such as *should*, *show*, or other language that could also appear in feature requests.

Some issues contained very little textual information, while others relied heavily on screenshots. Since the classifier uses only text, information contained in images was unavailable to the model.

The incorrectly classified issues are stored in:

`results/misclassified_issues.csv`

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
└── README.md
```

## Requirements

The project was developed using Python 3.

The main Python libraries are:

* `pandas`
* `requests`
* `scikit-learn`
* `matplotlib`

Install the required libraries using:

```bash
pip install pandas requests scikit-learn matplotlib
```

A Python virtual environment is recommended.

For example:

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Then install the required libraries:

```bash
pip install pandas requests scikit-learn matplotlib
```

## Reproducing the Experiment

## Reproducing the Reported Results

The exact dataset used for the reported experiment is included in:

`data/raw/vscode_closed_issues.csv`

To reproduce the reported preprocessing, analysis, and model results,
run the following commands from the project root:

```bash
python src/clean_data.py
python src/eda.py
python src/train_model.py

Run the following commands from the **root directory of the project**.

### Step 1: Collect GitHub Issues

```bash
python src/collect_issues.py
```

This script:

* Connects to the GitHub REST API
* Collects closed bug issues
* Collects closed feature-request issues
* Excludes pull requests
* Stores the relevant issue information
* Preserves all GitHub labels

The raw dataset is saved to:

```text
data/raw/vscode_closed_issues.csv
```

### Step 2: Clean and Prepare the Dataset

```bash
python src/clean_data.py
```

This script:

* Checks conflicting target labels
* Removes the issue containing both target labels
* Checks and removes duplicate issue numbers
* Handles missing descriptions
* Combines titles and descriptions
* Cleans the text
* Checks for empty text after cleaning

The processed dataset is saved to:

```text
data/processed/vscode_processed_issues.csv
```

### Step 3: Run Exploratory Data Analysis

```bash
python src/eda.py
```

This script examines:

* Dataset size
* Class distribution
* Issue text length
* Average issue length
* Average issue length by class

It generates the EDA figures and summary in:

```text
results/
```

### Step 4: Train and Evaluate the Models

```bash
python src/train_model.py
```

This script:

* Creates the stratified train-test split
* Fits TF-IDF on the training data
* Transforms the test data
* Evaluates the majority-class baseline
* Trains Logistic Regression
* Calculates accuracy
* Calculates precision
* Calculates recall
* Calculates F1-score
* Generates the confusion matrix
* Identifies incorrectly classified issues
* Saves the evaluation results

The main model outputs are:

```text
results/model_metrics.txt
results/confusion_matrix.png
results/misclassified_issues.csv
```

## Main Findings

The majority-class baseline achieved **50% accuracy**.

The TF-IDF with Logistic Regression classifier achieved **70% accuracy**.

Both bug reports and feature requests achieved:

* Precision: 0.70
* Recall: 0.70
* F1-score: 0.70

The results suggest that GitHub issue text contains useful information for distinguishing bug reports from feature requests.

However, the dataset and test set are small. Therefore, the results should be interpreted as preliminary evidence rather than a strong general conclusion about GitHub issue classification.

## Limitations

This is a small empirical study using issues from only one GitHub repository.

Important limitations include:

* The final dataset contains only 199 issues.
* Only the VS Code repository was studied.
* The test set contains only 40 issues.
* GitHub labels may contain ambiguity or inconsistency.
* Some issues contain very little textual information.
* Some issues rely on screenshots or other non-text information.
* The classifier cannot interpret information contained in images.
* Basic preprocessing may leave some GitHub template or technical boilerplate.
* Only one standard text classifier was evaluated in addition to the majority-class baseline.
* A single train-test split was used instead of cross-validation.

Future work could investigate:

* Larger datasets
* Multiple GitHub repositories
* Improved removal of templates and boilerplate
* Word and character n-grams
* Five-fold cross-validation
* Additional machine learning classifiers
* Contextual language models
* Cross-project evaluation
* Temporal evaluation
* More systematic validation of GitHub labels

## Generative AI Disclosure

Generative AI was used as a supporting tool during this assessment.

A detailed disclosure of the AI tool used, the tasks for which it was used, the main prompts submitted, and the components personally reviewed or modified is provided in the research report appendix.

## Status

The following parts of the empirical study have been completed:

* Data collection
* Label-quality checking
* Data preparation
* Exploratory data analysis
* Train-test split
* TF-IDF feature extraction
* Majority-class baseline
* Logistic Regression classifier
* Model evaluation
* Confusion matrix
* Manual error analysis

The research report and presentation are being finalized.
