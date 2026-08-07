# Import libraries
import requests
import pandas as pd
from pathlib import Path


# GitHub repository information
OWNER = "microsoft"
REPO = "vscode"

# Number of issues we want from each class
ISSUES_PER_CLASS = 100

# GitHub API URL
API_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/issues"


def collect_issues(label, class_name):
    """
    Collect closed GitHub issues for one label.

    label:
        GitHub label such as "bug" or "feature-request"

    class_name:
        The class name that will be stored in our dataset
    """

    collected_issues = []
    page = 1

    # Continue until we collect 100 valid issues
    while len(collected_issues) < ISSUES_PER_CLASS:

        # Parameters sent to the GitHub API
        params = {
            "state": "closed",
            "labels": label,
            "per_page": 100,
            "page": page
        }

        print(f"Collecting {label} issues - page {page}")

        # Send request to GitHub
        response = requests.get(
            API_URL,
            params=params,
            timeout=30
        )

        # Check whether the API request worked
        if response.status_code != 200:
            print("Error while accessing GitHub API")
            print("Status code:", response.status_code)
            break

        # Convert GitHub response from JSON into Python data
        issues = response.json()
        print(f"Number of issues returned: {len(issues)}")

        # If GitHub returns nothing, there are no more results
        if len(issues) == 0:
            break

        # Read each returned issue
        for issue in issues:

            # GitHub's Issues API can also return pull requests.
            # We only want actual issues.
            if "pull_request" in issue:
                continue

            issue_data = {
                "issue_number": issue["number"],
                "title": issue.get("title", ""),
                "description": issue.get("body") or "",
                "label": class_name,
                "created_at": issue.get("created_at"),
                "closed_at": issue.get("closed_at"),
                "url": issue.get("html_url")
            }

            collected_issues.append(issue_data)

            # Stop after reaching 100 issues
            if len(collected_issues) >= ISSUES_PER_CLASS:
                break

        page += 1

    print(
        f"Collected {len(collected_issues)} "
        f"{class_name} issues"
    )

    return collected_issues


# ------------------------------
# Collect bug issues
# ------------------------------

bug_issues = collect_issues(
    label="bug",
    class_name="bug"
)


# ------------------------------
# Collect feature-request issues
# ------------------------------

feature_issues = collect_issues(
    label="feature-request",
    class_name="feature_request"
)


# Combine the two classes
all_issues = bug_issues + feature_issues


# Convert the Python list into a pandas DataFrame
df = pd.DataFrame(all_issues)


# Remove duplicate issue numbers if any exist
df = df.drop_duplicates(
    subset="issue_number"
)


# Save the dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent

output_file = PROJECT_ROOT / "data" / "raw" / "vscode_closed_issues.csv"

df.to_csv(
    output_file,
    index=False
)


# Display basic information
print("\nData collection completed.")

print("\nTotal number of issues:")
print(len(df))

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nDataset saved to:")
print(output_file)