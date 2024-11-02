import yaml
from sqlalchemy import create_engine
import urllib.parse
import pandas as pd

class RDSDatabaseConnector:

    def __init__(self, creds_file='credentials.yaml'):
        """
        Initializes the connector by reading database credentials from the YAML file.
        """
        self.creds_file = creds_file
        self.creds = self.read_db_creds()
        self.engine = self.init_db_engine()

    def read_db_creds(self):
        try:
            with open(self.creds_file, 'r') as file:
                creds = yaml.safe_load(file)
                return creds
        except Exception as e:
            print(f"Error reading credentials file: {e}")
            return None

    def init_db_engine(self):
        """Initializes and returns an SQLAlchemy engine using the database credentials."""
        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        HOST = self.creds['RDS_HOST']
        USER = urllib.parse.quote_plus(self.creds['RDS_USER'])
        PASSWORD = urllib.parse.quote_plus(self.creds['RDS_PASSWORD'])
        DATABASE = self.creds['RDS_DATABASE']
        PORT = self.creds['RDS_PORT']

        engine = create_engine(
            f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
        )
        return engine

    def extract_data_to_df(self):
        """Extracts data from the loan_payments table and returns it as a Pandas DataFrame."""
        query = "SELECT * FROM loan_payments"
        try:
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            print(f"Error extracting data: {e}")
            return None

    def save_data_to_csv(self, df, file_path='loan_payments_data.csv'):
        """Saves the DataFrame to a CSV file."""
        try:
            df.to_csv(file_path, index=False)
            print(f"Data saved to {file_path}")
        except Exception as e:
            print(f"Error saving data to CSV: {e}")


    def load_data_from_csv(self, file_path='loan_payments_data.csv'):
        """
        Loads loan payments data from a CSV file into a Pandas DataFrame.
        
        Args:
        - file_path (str): The path to the CSV file containing the data.

        Returns:
        - pd.DataFrame: DataFrame containing the loan payments data.
        """
        try:
            # Load the data into a DataFrame
            df = pd.read_csv(file_path)
            
            # Print the shape of the DataFrame
            print(f"Data shape: {df.shape}")
            
            # Display a sample of the data
            print("Data sample:")
            print(df.head())
            
            return df
        
        except Exception as e:
            print(f"Error loading data from CSV: {e}")
            return None
    

connector = RDSDatabaseConnector()

# Step 1: Extract data from the RDS database and store it in a DataFrame
df = connector.extract_data_to_df()

# Step 2: Check if data was successfully extracted
if df is not None:
    print("Data successfully extracted from the database.")
    print(f"Extracted Data Shape: {df.shape}")
    print("Sample of Extracted Data:")
    print(df.head())

    # Step 3: Save the extracted data to a CSV file
    connector.save_data_to_csv(df)

    # Step 4: Load the data from the CSV to verify it saved correctly
    loaded_df = connector.load_data_from_csv('loan_payments_data.csv')

    # Check if loading was successful and inspect the loaded data
    if loaded_df is not None:
        print("Data loaded successfully from CSV:")
        print(f"Loaded Data Shape: {loaded_df.shape}")
        print("Sample of Loaded Data:")
        print(loaded_df.head())
else:
    print("Data extraction failed.")