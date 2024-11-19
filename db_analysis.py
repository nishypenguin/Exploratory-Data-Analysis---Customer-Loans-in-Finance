import pandas as pd
import numpy as np
from db_cleaner import DataTransform
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox, yeojohnson

class DataFrameInfo():

    def __init__(self, transformed_column_data):
        self.transformed_column_data = transformed_column_data

    def extract_statistical_values(self, column_name):
        mean_of_column = np.mean( self.transformed_column_data[f'{column_name}'])
        median_of_column = np.median( self.transformed_column_data[f'{column_name}'])
        stdev_of_column = np.std( self.transformed_column_data[f'{column_name}'])
        return (f"Mean of column '{column_name}' = {mean_of_column}, "
                f"Median = {median_of_column}, "
                f"Standard Deviation = {stdev_of_column}")

    def count_distinct_values(self, column_name):
         if self.transformed_column_data[column_name].dtype.name == 'category':
            distinct_count = self.transformed_column_data[column_name].nunique()
            return f"Column '{column_name}' has {distinct_count} distinct categorical values."
         else:
            return f"Column '{column_name}' is not categorical."

    def null_value_summary(self):
        null_counts = self.transformed_column_data.isnull().sum()
        null_percentages = (null_counts / len(self.transformed_column_data)) * 100
        non_null_values = self.transformed_column_data.count()
        summary = pd.DataFrame({
            'Null Count': null_counts,
            'Null Percentage': null_percentages,
            'Non-Null Values': non_null_values
         })
        
        return summary
    
    def drop_columns_with_high_nulls(self, threshold=50):
        """
        Drops columns with null percentages greater than the given threshold.

        Args:
            threshold (float): The percentage threshold for null values.
        """
        null_summary = self.null_value_summary()
        high_null_cols = null_summary[null_summary['Null Percentage'] > threshold].index
        self.transformed_column_data.drop(columns=high_null_cols, inplace=True)
        print(f"Dropped columns: {list(high_null_cols)}")

        return self.transformed_column_data

    def input_missing_values(self):
        """
        Imputes missing values in the DataFrame:
        - Numeric columns: Impute with median
        - Categorical columns: Impute with mode
        """
        for col in self.transformed_column_data.columns:
            if self.transformed_column_data[col].isnull().sum() > 0:  # Check if there are null values
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
            threshold (float): The skewness threshold for identifying skewed columns.

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
        Chooses the transformation that reduces skewness the most.

        Args:
            skewed_columns (list): List of skewed column names.

        Returns:
            dict: A dictionary containing the best transformation applied for each column and the resulting skewness.
        """
        best_transformations = {}

        for col in skewed_columns:
            original_skew = self.transformed_column_data[col].skew()
            best_skew = abs(original_skew)
            best_method = None

            # Log Transformation
            if (self.transformed_column_data[col] > 0).all():  # Log requires all positive values
                log_transformed = np.log1p(self.transformed_column_data[col])
                log_skew = abs(log_transformed.skew())
                if log_skew < best_skew:
                    best_skew = log_skew
                    best_method = "log1p"
                    self.transformed_column_data[col] = log_transformed

            # Box-Cox Transformation
            if (self.transformed_column_data[col] > 0).all():  # Box-Cox also requires all positive values
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

            # Record the best transformation
            if best_method:
                best_transformations[col] = {"method": best_method, "skewness": best_skew}
                print(f"Applied {best_method} transformation to '{col}' (Skewness: {best_skew:.4f})")
            else:
                print(f"No suitable transformation applied to '{col}' (Original Skewness: {original_skew:.4f})")

        print(best_transformations)

    def handle_outliers(self, method="remove", factor=1.5):
        """
        Identifies and handles outliers using the IQR method for numeric columns.

        Args:
            method (str): How to handle outliers. Options: "remove" or "transform".
            factor (float): The IQR factor to determine outliers (default is 1.5).

        Returns:
            dict: A dictionary with the number of outliers handled for each column.
        """
        # Identify numeric columns
        numeric_cols = self.transformed_column_data.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            print("No numeric columns found for outlier handling.")
            return {}

        outlier_summary = {}

        for col in numeric_cols:
            # Calculate IQR
            Q1 = self.transformed_column_data[col].quantile(0.25)
            Q3 = self.transformed_column_data[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define bounds for outliers
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR

            # Identify outliers
            outliers = self.transformed_column_data[
                (self.transformed_column_data[col] < lower_bound) | 
                (self.transformed_column_data[col] > upper_bound)
            ]
            outlier_count = len(outliers)

            # Handle outliers based on method
            if method == "remove":
                self.transformed_column_data = self.transformed_column_data[
                    (self.transformed_column_data[col] >= lower_bound) &
                    (self.transformed_column_data[col] <= upper_bound)
                ]
            elif method == "transform":
                self.transformed_column_data[col] = np.clip(
                    self.transformed_column_data[col], lower_bound, upper_bound
                )

            # Record outlier handling
            outlier_summary[col] = outlier_count
            print(f"Handled {outlier_count} outliers in column '{col}' using method '{method}'.")

        return outlier_summary
    
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
    
    
class Plotter:
    """A class for visualizing data insights."""
    
    def __init__(self, transformed_column_data):
        """
        Initializes the Plotter class.
        
        Args:
            transformed_column_data (pd.DataFrame): The DataFrame to visualize.
        """
        self.transformed_column_data = transformed_column_data

    def plot_null_values(self, null_summary_before, null_summary_after):
        """
        Plots null values before and after handling transformations.

        Args:
            null_summary_before (pd.DataFrame): Null summary before transformations.
            null_summary_after (pd.DataFrame): Null summary after transformations.
        """

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sns.barplot(x=null_summary_before.index, y=null_summary_before['Null Percentage'], ax=ax[0])
        ax[0].set_title("Null Values Before Transformation")
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)

        sns.barplot(x=null_summary_after.index, y=null_summary_after['Null Percentage'], ax=ax[1])
        ax[1].set_title("Null Values After Transformation")
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)

        plt.tight_layout()
        plt.show()

    def visualize_skewness(self, dataframe, columns):
        """
        Plots histograms for given columns to visualize their skewness.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            columns (list): List of column names to visualize.
        """
        for col in columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(dataframe[col], kde=True)
            plt.title(f"Distribution of {col} (Skew: {dataframe[col].skew():.2f})")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()

    def visualize_outliers(self):
        """
        Visualizes potential outliers using boxplots for numeric columns in the DataFrame.
        """
        # Identify numeric columns
        numeric_cols = self.transformed_column_data.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            print("No numeric columns found for visualization.")
            return

        for col in numeric_cols:
            # Plot boxplot for each numeric column
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=self.transformed_column_data[col])
            plt.title(f"Boxplot of {col}")
            plt.xlabel(col)
        plt.show()

    def visualize_correlation_matrix(self):
        """
        Computes and visualizes the correlation matrix for numeric columns in the dataset.
        """
        # Select only numeric columns
        numeric_data = self.transformed_column_data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            print("No numeric columns found to compute the correlation matrix.")
            return None

        # Compute the correlation matrix
        correlation_matrix = numeric_data.corr()

        # Visualize the correlation matrix using a heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Matrix")
        plt.show()

        return correlation_matrix

if __name__ == "__main__":
    executer = DataFrameInfo('loan_payments_data.csv')
    column_statistics = executer.extract_statistical_values('loan_amount')
    column_distinc_values = executer.count_distinct_values('loan_amount')
    null_value_summary = executer.null_value_summary()
    