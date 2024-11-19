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
    
    def identify_highly_correlated_columns(self, numeric_data, threshold=0.9):
        """
        Identifies columns with high correlation above a specified threshold in numeric columns.

        Args:
            numeric_data (pd.DataFrame): DataFrame containing only numeric columns.
            threshold (float): Correlation coefficient threshold for identifying high correlations.

        Returns:
            list: List of column names to be removed due to high correlation.
        """
        # Compute the correlation matrix
        correlation_matrix = numeric_data.corr()

        # Identify pairs of columns with high correlation
        high_corr_pairs = correlation_matrix.abs().unstack().sort_values(ascending=False)

        # Filter out self-correlation
        high_corr_pairs = high_corr_pairs[high_corr_pairs < 1]

        # Find columns to remove
        columns_to_remove = set()
        for (col1, col2), corr_value in high_corr_pairs.items():
            if corr_value > threshold:
                if col1 not in columns_to_remove and col2 not in columns_to_remove:
                    columns_to_remove.add(col2)  # Keep one column, remove the other

        print(f"Columns to remove due to high correlation: {columns_to_remove}")
        return list(columns_to_remove)
    
    def remove_highly_correlated_columns(self, threshold=0.9):
        """
        Removes highly correlated columns based on a specified threshold, processing only numeric columns.

        Args:
            threshold (float): Correlation coefficient threshold for identifying high correlations.

        Returns:
            pd.DataFrame: The updated DataFrame with highly correlated columns removed.
        """
        # Filter to only numerical columns
        numeric_data = self.transformed_column_data.select_dtypes(include=[np.number])

        # Identify highly correlated columns
        columns_to_remove = self.identify_highly_correlated_columns(numeric_data, threshold=threshold)

        # Remove the columns from the dataset
        self.transformed_column_data.drop(columns=columns_to_remove, inplace=True)

        print(f"Removed columns: {columns_to_remove}")
        return self.transformed_column_data

class Plotter:
    """
    A class for visualizing insights and patterns in a DataFrame.
    """

    def __init__(self, transformed_column_data):
        """
        Initializes the Plotter class with the transformed DataFrame.

        Args:
            transformed_column_data (pd.DataFrame): The DataFrame to visualize.
        """
        self.transformed_column_data = transformed_column_data

    def plot_null_values(self, null_summary_before, null_summary_after):
        """
        Plots null value percentages before and after handling transformations.

        Args:
            null_summary_before (pd.DataFrame): DataFrame summarizing null values before transformations.
            null_summary_after (pd.DataFrame): DataFrame summarizing null values after transformations.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot null values before transformations
        sns.barplot(x=null_summary_before.index, y=null_summary_before['Null Percentage'], ax=ax[0])
        ax[0].set_title("Null Values Before Transformation")
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
        ax[0].set_ylabel("Null Percentage")

        # Plot null values after transformations
        sns.barplot(x=null_summary_after.index, y=null_summary_after['Null Percentage'], ax=ax[1])
        ax[1].set_title("Null Values After Transformation")
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)
        ax[1].set_ylabel("Null Percentage")

        plt.tight_layout()
        plt.show()

    def visualize_skewness(self, dataframe, columns):
        """
        Plots histograms for selected columns to visualize their skewness.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data to visualize.
            columns (list): List of column names to plot.
        """
        for col in columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(dataframe[col], kde=True, bins=30)
            plt.title(f"Distribution of {col} (Skew: {dataframe[col].skew():.2f})")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()

    def visualize_outliers(self):
        """
        Visualizes potential outliers using boxplots for numeric columns in the DataFrame.
        """
        numeric_cols = self.transformed_column_data.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            print("No numeric columns found for outlier visualization.")
            return

        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=self.transformed_column_data[col])
            plt.title(f"Boxplot of {col}")
            plt.xlabel(col)
            plt.show()

    def visualize_correlation_matrix(self):
        """
        Computes and visualizes the correlation matrix for numeric columns in the dataset.

        Returns:
            pd.DataFrame: The correlation matrix as a DataFrame.
        """
        numeric_data = self.transformed_column_data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            print("No numeric columns found to compute the correlation matrix.")
            return None

        correlation_matrix = numeric_data.corr()

        # Visualize the correlation matrix as a heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Matrix")
        plt.show()

        return correlation_matrix


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
