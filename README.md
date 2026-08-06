# GitHub Issue Classification

This project is part of a prospective MSc research assessment.

## Research Question

How accurately can a simple machine learning model distinguish bug reports from feature requests using the text of GitHub issues?

## Selected Project

- GitHub repository: `microsoft/vscode`
- Project category: Developer tool
- Target labels:
  - `bug`
  - `feature-request`

## Planned Dataset

- 100 closed bug issues
- 100 closed feature-request issues
- Total: 200 closed issues

For every issue, the dataset will include:

- Issue number
- Issue title
- Issue description
- Issue label
- Creation date
- Closing date

## Planned Methods

- GitHub REST API for reproducible data collection
- Text cleaning and quality checking
- TF-IDF feature extraction
- Majority-class baseline
- Logistic Regression text classifier
- Accuracy, precision, recall, F1-score, and confusion matrix
- Manual analysis of at least ten classification errors

## Project Structure

- `data/raw/`: original collected issue data
- `data/processed/`: cleaned data used for modelling
- `src/`: reusable Python scripts
- `notebooks/`: analysis and modelling notebooks
- `results/`: figures and evaluation outputs
- `report/`: research report materials
- `slides/`: presentation materials

## Status

Project setup completed. Data collection is the next step.
