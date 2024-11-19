import yaml
from sqlalchemy import create_engine
import urllib.parse
import pandas as pd


class RDSDatabaseConnector:
    """
    A class for connecting to an AWS RDS database, extracting data,
    and saving/loading it as a CSV file.
    """

    def __init__(self, creds_file='credentials.yaml'):
        """
        Initializes the connector by reading database credentials
        from the YAML file and setting up the engine.

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
            dict: A dictionary containing the database credentials.
        """
        try:
            with open(self._creds_file, 'r') as file:
                credentials = yaml.safe_load(file)
            return credentials
        except FileNotFoundError:
            print(f"Error: Credentials file '{self._creds_file}' not found.")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
        except Exception as e:
            print(f"Unexpected error reading credentials file: {e}")
        return None

    def _initialize_db_engine(self):
        """
        Initializes and returns a SQLAlchemy engine using the stored database credentials.

        Returns:
            sqlalchemy.engine.Engine: SQLAlchemy engine object for database connection.
        """
        try:
            database_type = 'postgresql'
            db_api = 'psycopg2'
            host = self._creds['RDS_HOST']
            user = urllib.parse.quote_plus(self._creds['RDS_USER'])
            password = urllib.parse.quote_plus(self._creds['RDS_PASSWORD'])
            database = self._creds['RDS_DATABASE']
            port = self._creds['RDS_PORT']

            return create_engine(
                f"{database_type}+{db_api}://{user}:{password}@{host}:{port}/{database}"
            )
        except KeyError as e:
            print(f"Error: Missing database credential key: {e}")
        except Exception as e:
            print(f"Unexpected error initializing database engine: {e}")
        return None

    def fetch_loan_payments_data(self):
        """
        Fetches data from the 'loan_payments' table in the RDS database.

        Returns:
            pd.DataFrame: DataFrame containing the loan payments data.
        """
        query = "SELECT * FROM loan_payments"
        try:
            return pd.read_sql(query, self._engine)
        except Exception as e:
            print(f"Error fetching data from database: {e}")
        return None

    def save_dataframe_to_csv(self, dataframe, file_path='loan_payments_data.csv'):
        """
        Saves the provided DataFrame to a CSV file.

        Args:
            dataframe (pd.DataFrame): The DataFrame to save.
            file_path (str): The path to the CSV file.
        """
        try:
            dataframe.to_csv(file_path, index=False)
            print(f"Data successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving data to CSV: {e}")

    def load_data_from_csv(self, file_path='loan_payments_data.csv'):
        """
        Loads data from a CSV file into a DataFrame.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            pd.DataFrame: DataFrame containing the loaded data.
        """
        try:
            dataframe = pd.read_csv(file_path)
            print(f"Data successfully loaded. Shape: {dataframe.shape}")
            print("Sample data:")
            print(dataframe.head())
            return dataframe
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
        except pd.errors.ParserError as e:
            print(f"Error parsing CSV file: {e}")
        except Exception as e:
            print(f"Unexpected error loading data from CSV: {e}")
        return None


# Main script execution
if __name__ == "__main__":
    connector = RDSDatabaseConnector()

    # Load data from CSV
    data = connector.load_data_from_csv()
    if data is not None:
        print("Data successfully loaded from CSV:")
        print(f"Loaded Data Types:\n{data.dtypes}")
