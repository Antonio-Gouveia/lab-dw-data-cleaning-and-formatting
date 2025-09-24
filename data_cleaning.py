# data_cleaning.py

import pandas as pd
import numpy as np

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the column names of a DataFrame.

    This function performs the following steps:
    1. Converts all column names to lowercase.
    2. Replaces spaces with underscores.
    3. Renames specific columns 'st' to 'state' and 'income' to 'customer_income'.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with standardized column names.
    """
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    df = df.rename(columns={'st': 'state', 'income': 'customer_income'})
    return df

def standardize_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the 'gender' column to 'F' or 'M'.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the 'gender' column standardized.
    """
    df.loc[df['gender'].str.contains(r'^[Ff]', na=False), 'gender'] = 'F'
    df.loc[df['gender'].str.contains(r'^[Mm]', na=False), 'gender'] = 'M'
    return df

def standardize_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the 'state' column to full state names.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the 'state' column standardized.
    """
    state_mapping = {
        'AL': 'ALABAMA', 'AK': 'ALASKA', 'AZ': 'ARIZONA', 'AR': 'ARKANSAS', 'CA': 'CALIFORNIA', 'CALI': 'CALIFORNIA',
        'CO': 'COLORADO', 'CT': 'CONNECTICUT', 'DE': 'DELAWARE', 'FL': 'FLORIDA', 'GA': 'GEORGIA',
        'HI': 'HAWAII', 'ID': 'IDAHO', 'IL': 'ILLINOIS', 'IN': 'INDIANA', 'IA': 'IOWA',
        'KS': 'KANSAS', 'KY': 'KENTUCKY', 'LA': 'LOUISIANA', 'ME': 'MAINE', 'MD': 'MARYLAND',
        'MA': 'MASSACHUSETTS', 'MI': 'MICHIGAN', 'MN': 'MINNESOTA', 'MS': 'MISSISSIPPI', 'MO': 'MISSOURI',
        'MT': 'MONTANA', 'NE': 'NEBRASKA', 'NV': 'NEVADA', 'NH': 'NEW HAMPSHIRE', 'NJ': 'NEW JERSEY',
        'NM': 'NEW MEXICO', 'NY': 'NEW YORK', 'NC': 'NORTH CAROLINA', 'ND': 'NORTH DAKOTA', 'OH': 'OHIO',
        'OK': 'OKLAHOMA', 'OR': 'OREGON', 'PA': 'PENNSYLVANIA', 'RI': 'RHODE ISLAND', 'SC': 'SOUTH CAROLINA',
        'SD': 'SOUTH DAKOTA', 'TN': 'TENNESSEE', 'TX': 'TEXAS', 'UT': 'UTAH', 'VT': 'VERMONT',
        'VA': 'VIRGINIA', 'WA': 'WASHINGTON', 'WV': 'WEST VIRGINIA', 'WI': 'WISCONSIN', 'WY': 'WYOMING'
    }
    df['state'] = df['state'].str.upper().replace(state_mapping)
    return df

def standardize_education(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the 'education' column.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the 'education' column standardized.
    """
    df.loc[df['education'].str.contains(r'^[Bb]', na=False), 'education'] = 'Bachelor'
    return df

def standardize_vehicle_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the 'vehicle_class' column by grouping similar values.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the 'vehicle_class' column standardized.
    """
    df.loc[df['vehicle_class'].str.contains(r'^[Lu]', na=False), 'vehicle_class'] = 'Luxury'
    df.loc[df['vehicle_class'].str.contains(r'\bSports\b', na=False), 'vehicle_class'] = 'Luxury'
    return df

def clean_and_convert_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and converts numerical columns.

    This function removes the '%' character from 'customer_lifetime_value',
    and then converts it and other specified columns to the correct numeric types.
    It also extracts and converts the 'number_of_open_complaints' column.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with cleaned and converted numerical columns.
    """
    # Remove '%' and convert to numeric
    df['customer_lifetime_value'] = df['customer_lifetime_value'].str.rstrip('%')
    df['customer_lifetime_value'] = pd.to_numeric(df['customer_lifetime_value'], errors='coerce').astype('float64')

    # Extract number of open complaints and convert to numeric
    df['number_of_open_complaints'] = df['number_of_open_complaints'].str.extract(r'/(\d+)/')
    df['number_of_open_complaints'] = pd.to_numeric(df['number_of_open_complaints'], errors='coerce').astype('Int64')
    
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills in missing values in both categorical and numerical columns.

    This function fills missing values in categorical columns with 'Unknown'
    and in numerical columns with the median. It also creates a flag column
    to identify previously missing 'number_of_open_complaints' values.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """
    # Fill categorical columns with 'Unknown'
    categorical_cols = ['customer', 'state', 'gender', 'education', 'policy_type', 'vehicle_class']
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')

    # Fill numerical columns with the median
    numerical_cols = ['total_claim_amount', 'monthly_premium_auto', 'customer_income', 'customer_lifetime_value']
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())

    # Create a flag for missing complaints
    df['complaints_missing'] = df['number_of_open_complaints'].isnull()
    
    return df

def main_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to execute the complete data cleaning and formatting pipeline.

    This function orchestrates the entire cleaning process by calling
    all the individual cleaning functions in a logical order.

    Args:
        df (pd.DataFrame): The raw input DataFrame.

    Returns:
        pd.DataFrame: The fully cleaned and formatted DataFrame.
    """
    print("Starting the data cleaning and formatting pipeline...")
    
    # Execute functions in a logical sequence
    df = clean_column_names(df)
    df = standardize_gender(df)
    df = standardize_state(df)
    df = standardize_education(df)
    df = standardize_vehicle_class(df)
    df = clean_and_convert_numerical(df)
    df = handle_missing_values(df)
    
    print("Data cleaning pipeline completed successfully.")
    return df