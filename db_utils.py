import yaml
from sqlalchemy import create_engine
import urllib.parse
import pandas as pd

class RDSDatabaseConnector:
    """A class for connecting to an AWS RDS database, extracting data, and saving/loading it as a CSV file."""

    def __init__(self, creds_file='credentials.yaml'):
        """
        Initializes the connector by reading database credentials from the YAML file and setting up the engine.

        Args:
            creds_file (str): Path to the YAML file containing database credentials.
        """
        self._creds_file = creds_file
        self._creds = self._read_db_credentials()
        self._engine = self._initialize_db_engine()

    def _read_db_credentials(self):
        """
        Reads the database credentials from the YAML file.

        Returns:
            A dictionary containing the database credentials if successful, None otherwise.
        """
        try:
            with open(self._creds_file, 'r') as file:
                credentials = yaml.safe_load(file)
                return credentials
        except Exception as e:
            print(f"Error reading credentials file: {e}")
            return None

    def _initialize_db_engine(self):
        """
        Initializes and returns a SQLAlchemy engine using the stored database credentials.

        Returns:
            SQLAlchemy engine object for database connection.
        """
        try:
            database_type = 'postgresql'
            db_api = 'psycopg2'
            host = self._creds['RDS_HOST']
            user = urllib.parse.quote_plus(self._creds['RDS_USER'])
            password = urllib.parse.quote_plus(self._creds['RDS_PASSWORD'])
            database = self._creds['RDS_DATABASE']
            port = self._creds['RDS_PORT']

            engine = create_engine(
                f"{database_type}+{db_api}://{user}:{password}@{host}:{port}/{database}"
            )
            return engine
        except KeyError:
            print("Database credentials are missing.")
            return None

    def fetch_loan_payments_data(self):
        """
        Fetches data from the loan_payments table in the RDS database and loads it into a DataFrame.

        Returns:
            A DataFrame containing the loan payments data if successful, None otherwise.
        """
        query = "SELECT * FROM loan_payments"
        try:
            df = pd.read_sql(query, self._engine)
            return df
        except Exception as e:
            print(f"Error fetching data from database: {e}")
            return None

    def save_dataframe_to_csv(self, dataframe, file_path='loan_payments_data.csv'):
        """
        Saves the provided DataFrame to a CSV file.

        Args:
            dataframe (pd.DataFrame): The DataFrame to save.
            file_path (str): The path to the CSV file where data will be saved.
        """
        try:
            dataframe.to_csv(file_path, index=False)
            print(f"Data saved to {file_path}")
        except Exception as e:
            print(f"Error saving data to CSV: {e}")

    def load_data_from_csv(self, file_path='loan_payments_data.csv'):
        """
        Loads data from a CSV file into a DataFrame.

        Args:
            file_path (str): The path to the CSV file containing the data.

        Returns:
            A DataFrame containing the loaded data if successful, None otherwise.
        """
        try:
            dataframe = pd.read_csv(file_path)
            print(f"Data shape: {dataframe.shape}")
            print("Data sample:")
            print(dataframe.head())
            return dataframe
        except Exception as e:
            print(f"Error loading data from CSV: {e}")
            return None

# Main script execution
if __name__ == "__main__":
    connector = RDSDatabaseConnector()

    
    my_data = connector.load_data_from_csv()
    print("Data successfully loaded from CSV:")
    print(f"Loaded Data Typed: {my_data.dtypes}")

