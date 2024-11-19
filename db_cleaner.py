import pandas as pd
import numpy as np


class DataTransform:

    def __init__(self, file = 'loan_payments_data.csv'):
        self.untransformed_dataframe = pd.read_csv(file)

    def transform_columns(self):

        self.untransformed_dataframe['term'] = self.untransformed_dataframe['term'].str.replace('months', '')
        self.untransformed_dataframe['term'] = self.untransformed_dataframe['term'].astype('Int64')
        self.untransformed_dataframe['grade'] = self.untransformed_dataframe['grade'].astype('category')
        self.untransformed_dataframe['sub_grade'] = self.untransformed_dataframe['sub_grade'].astype('category')
        self.untransformed_dataframe['home_ownership'] = self.untransformed_dataframe['home_ownership'].astype('category')
        self.untransformed_dataframe['verification_status'] = self.untransformed_dataframe['verification_status'].astype('category')
        self.untransformed_dataframe['issue_date'] = pd.to_datetime(self.untransformed_dataframe['issue_date'], format='%b-%Y')
        self.untransformed_dataframe['loan_status'] = self.untransformed_dataframe['loan_status'].astype('category')
        self.untransformed_dataframe['payment_plan'] = self.untransformed_dataframe['payment_plan'].astype('category')
        self.untransformed_dataframe['purpose'] = self.untransformed_dataframe['purpose'].str.replace('_', ' ').astype('category')
        self.untransformed_dataframe['delinq_2yrs'] = self.untransformed_dataframe['delinq_2yrs'].astype('Int64')
        self.untransformed_dataframe['earliest_credit_line'] = pd.to_datetime(self.untransformed_dataframe['earliest_credit_line'], format='%b-%Y')
        self.untransformed_dataframe['inq_last_6mths'] = self.untransformed_dataframe['inq_last_6mths'].astype('Int64')
        self.untransformed_dataframe['mths_since_last_delinq'] = self.untransformed_dataframe['mths_since_last_delinq'].astype('Int64')
        self.untransformed_dataframe['mths_since_last_record'] = self.untransformed_dataframe['mths_since_last_record'].astype('Int64')
        self.untransformed_dataframe['last_payment_date'] = pd.to_datetime(self.untransformed_dataframe['last_payment_date'], format='%b-%Y')
        self.untransformed_dataframe['next_payment_date'] = pd.to_datetime(self.untransformed_dataframe['next_payment_date'], format='%b-%Y')
        self.untransformed_dataframe['last_credit_pull_date'] = pd.to_datetime(self.untransformed_dataframe['last_credit_pull_date'], format='%b-%Y')
        self.untransformed_dataframe['collections_12_mths_ex_med'] = self.untransformed_dataframe['collections_12_mths_ex_med'].astype('Int64')
        self.untransformed_dataframe['mths_since_last_major_derog'] = self.untransformed_dataframe['mths_since_last_major_derog'].astype('Int64')
        self.untransformed_dataframe['policy_code'] = self.untransformed_dataframe['policy_code'].astype('category')
        self.untransformed_dataframe['application_type'] = self.untransformed_dataframe['application_type'].astype('category')

        self.transformed_dataframe = self.untransformed_dataframe

        return self.transformed_dataframe
    
    

        

        
if __name__ == "__main__":
    cleaner = DataTransform()
    cleaned_file = cleaner.transform_columns()
    print(cleaned_file)