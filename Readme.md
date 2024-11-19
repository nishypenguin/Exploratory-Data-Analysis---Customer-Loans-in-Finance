# Loan Payment Analysis Project

## Overview

This project focuses on analyzing loan payment data extracted from an AWS RDS database. The workflow includes data cleaning, exploratory data analysis (EDA), and advanced insights generation to inform business decisions.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Key Scripts](#key-scripts)
- [Tasks Performed](#tasks-performed)
  - [Data Cleaning](#data-cleaning)
  - [EDA and Advanced Analysis](#eda-and-advanced-analysis)
- [Results and Insights](#results-and-insights)
- [Future Improvements](#future-improvements)

---

## Project Structure

```plaintext
.
├── db_utils.py             # Script for data extraction from AWS RDS and CSV operations
├── db_cleaner.py           # Script for cleaning and formatting data
├── db_analysis.py          # Script for EDA and analysis utilities
├── analysis.ipynb          # Notebook for EDA and data transformations
├── final_analysis.ipynb    # Notebook for advanced analysis and insights
├── requirements.txt        # Python dependencies
├── credentials.yaml        # Database credentials (not included for security reasons)
└── README.md               # Project documentation

```
---

## Key Scripts

### `db_utils.py`
- **Fetch Data**: Pull data from the AWS RDS database.
- **Save to CSV**: Save data to a local CSV file.
- **Load Data**: Load data from a CSV file into a pandas DataFrame.

### `db_cleaner.py`
- Contains the `DataTransform` class for cleaning and standardizing the data:
  - Converts dates to datetime format.
  - Converts categorical data to appropriate types.
  - Handles missing values and standardizes column formats.

### `db_analysis.py`
- Contains the `DataFrameInfo` class for EDA tasks:
  - Generates statistical summaries (mean, median, standard deviation).
  - Handles null values (detection, removal, or imputation).
  - Identifies and transforms skewed columns.
  - Removes highly correlated columns.

---

## Tasks Performed

### Data Cleaning
- **Column Formatting**:
  - Converted dates to datetime format.
  - Standardized column names and removed excess symbols.
- **Null Value Handling**:
  - Identified columns with high null percentages and dropped them.
  - Imputed missing values using the median (numeric) or mode (categorical).
- **Skewness Reduction**:
  - Identified skewed columns and applied transformations (log1p, boxcox, yeojohnson) to reduce skewness.
- **Outlier Removal**:
  - Visualized and removed outliers to improve data quality.

### EDA and Advanced Analysis
- **Loan Recovery Analysis**:
  - Calculated loan recovery rates.
  - Projected recovery amounts over the next 6 months.
- **Loan Loss Analysis**:
  - Determined percentages of "Charged Off" loans and calculated expected losses.
- **Late Payments Risk Assessment**:
  - Analyzed overdue payments and estimated risks of default.
- **Default Indicators**:
  - Identified factors like loan grade and credit scores affecting default risks.

---

## Results and Insights

- **Loan Recovery**:
  - X% of loans have been successfully recovered.
  - An additional £X is projected to be recovered in the next 6 months.
- **Loan Loss**:
  - X% of loans were marked as "Charged Off," resulting in £Y losses.
  - Late payments could lead to £Z additional losses.
- **Default Indicators**:
  - Loan grade and purpose were significant predictors of default risk.
  - Customers with lower credit scores or higher debt-to-income ratios were more likely to default.

---

## Future Improvements

- Automate the data pipeline for real-time data updates and analysis.
- Use machine learning models to predict loan defaults based on historical data.
- Create interactive dashboards for visualizing key metrics and insights