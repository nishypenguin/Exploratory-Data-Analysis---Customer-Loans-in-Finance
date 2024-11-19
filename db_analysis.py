import pandas as pd
import numpy as np
from db_cleaner import DataTransform
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox, yeojohnson


class DataFrameInfo:
    """
    A class for performing various analyses and transformations on a DataFrame.
    """

    def __init__(self, transformed_column_data):
        """
        Initializes the class with the transformed DataFrame.

        Args:
            transformed_column_data (pd.DataFrame): The DataFrame to analyze and transform.
        """
        self.transformed_column_data = transformed_column_data

    def extract_statistical_values(self, column_name):
        """
        Extracts statistical values (mean, median, standard deviation) for a given column.

        Args:
            column_name (str): The name of the column.

        Returns:
            str: A string summary of the statistical values.
        """
        mean_of_column = np.mean(self.transformed_column_data[column_name])
        median_of_column = np.median(self.transformed_column_data[column_name])
        stdev_of_column = np.std(self.transformed_column_data[column_name])

        return (
            f"Mean of column '{column_name}' = {mean_of_column:.2f}, "
            f"Median = {median_of_column:.2f}, "
            f"Standard Deviation = {stdev_of_column:.2f}"
        )

    def count_distinct_values(self, column_name):
        """
        Counts distinct values for a given column if it is categorical.

        Args:
            column_name (str): The name of the column.

        Returns:
            str: Summary of distinct values or a message if the column is not categorical.
        """
        if self.transformed_column_data[column_name].dtype.name == 'category':
            distinct_count = self.transformed_column_data[column_name].nunique()
            return f"Column '{column_name}' has {distinct_count} distinct categorical values."
        else:
            return f"Column '{column_name}' is not categorical."

    def null_value_summary(self):
        """
        Summarizes null values in the DataFrame.

        Returns:
            pd.DataFrame: A DataFrame summarizing null counts, percentages, and non-null values.
        """
        null_counts = self.transformed_column_data.isnull().sum()
        null_percentages = (null_counts / len(self.transformed_column_data)) * 100
        non_null_values = self.transformed_column_data.count()

        return pd.DataFrame({
            'Null Count': null_counts,
            'Null Percentage': null_percentages,
            'Non-Null Values': non_null_values
        })

    def drop_columns_with_high_nulls(self, threshold=50):
        """
        Drops columns with null percentages greater than a specified threshold.

        Args:
            threshold (float): The percentage threshold for null values.

        Returns:
            pd.DataFrame: Updated DataFrame after dropping columns.
        """
        null_summary = self.null_value_summary()
        high_null_cols = null_summary[null_summary['Null Percentage'] > threshold].index
        self.transformed_column_data.drop(columns=high_null_cols, inplace=True)
        print(f"Dropped columns: {list(high_null_cols)}")

        return self.transformed_column_data

    def input_missing_values(self):
        """
        Imputes missing values in the DataFrame:
        - Numeric columns: Median
        - Categorical columns: Mode

        Returns:
            pd.DataFrame: Updated DataFrame with missing values imputed.
        """
        for col in self.transformed_column_data.columns:
            if self.transformed_column_data[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(self.transformed_column_data[col]):
                    self.transformed_column_data[col].fillna(self.transformed_column_data[col].median(), inplace=True)
                    print(f"Imputed missing values in numeric column '{col}' with median.")
                elif pd.api.types.is_categorical_dtype(self.transformed_column_data[col]) or self.transformed_column_data[col].dtype == 'object':
                    self.transformed_column_data[col].fillna(self.transformed_column_data[col].mode()[0], inplace=True)
                    print(f"Imputed missing values in categorical column '{col}' with mode.")

        return self.transformed_column_data

    def identify_skewed_columns(self, threshold=0.75):
        """
        Identifies numeric columns with skewness exceeding the given threshold.

        Args:
            threshold (float): The skewness threshold.

        Returns:
            list: A list of skewed column names.
        """
        numeric_cols = self.transformed_column_data.select_dtypes(include=[np.number])
        skewness = numeric_cols.skew()
        skewed_cols = skewness[skewness.abs() > threshold].index.tolist()
        print(f"Skewed Columns (Threshold {threshold}): {skewed_cols}")
        return skewed_cols

    def transform_skewed_columns(self, skewed_columns):
        """
        Transforms skewed columns to reduce skewness using Log, Box-Cox, and Yeo-Johnson transformations.

        Args:
            skewed_columns (list): List of skewed column names.

        Returns:
            dict: Dictionary of the best transformations applied.
        """
        best_transformations = {}

        for col in skewed_columns:
            original_skew = self.transformed_column_data[col].skew()
            best_skew = abs(original_skew)
            best_method = None

            if (self.transformed_column_data[col] > 0).all():
                # Log Transformation
                log_transformed = np.log1p(self.transformed_column_data[col])
                log_skew = abs(log_transformed.skew())
                if log_skew < best_skew:
                    best_skew = log_skew
                    best_method = "log1p"
                    self.transformed_column_data[col] = log_transformed

                # Box-Cox Transformation
                boxcox_transformed, _ = boxcox(self.transformed_column_data[col])
                boxcox_skew = abs(pd.Series(boxcox_transformed).skew())
                if boxcox_skew < best_skew:
                    best_skew = boxcox_skew
                    best_method = "boxcox"
                    self.transformed_column_data[col] = boxcox_transformed

            # Yeo-Johnson Transformation
            yeojohnson_transformed, _ = yeojohnson(self.transformed_column_data[col])
            yeojohnson_skew = abs(pd.Series(yeojohnson_transformed).skew())
            if yeojohnson_skew < best_skew:
                best_skew = yeojohnson_skew
                best_method = "yeojohnson"
                self.transformed_column_data[col] = yeojohnson_transformed

            if best_method:
                best_transformations[col] = {"method": best_method, "skewness": best_skew}
                print(f"Applied {best_method} transformation to '{col}' (Skewness: {best_skew:.4f})")
            else:
                print(f"No suitable transformation applied to '{col}' (Original Skewness: {original_skew:.4f})")

        return best_transformations


if __name__ == "__main__":
    # Example usage
    cleaner = DataTransform('loan_payments_data.csv')
    transformed_data = cleaner.transform_columns()

    executor = DataFrameInfo(transformed_data)
    stats = executor.extract_statistical_values('loan_amount')
    print(stats)

    distinct_values = executor.count_distinct_values('loan_status')
    print(distinct_values)

    null_summary = executor.null_value_summary()
    print(null_summary)
