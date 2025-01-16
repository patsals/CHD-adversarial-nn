import numpy as np
import pandas as pd
import os

# Configurations
data_directory = 'data'
raw_data_filename = 'nhanes_data.parquet'
cleaned_data_filename = 'nhanes_data_processed.parquet'
codebook_filename = "codebook.csv"
resulting_filepath = os.path.join(data_directory, cleaned_data_filename)
original_filepath = os.path.join(data_directory, raw_data_filename)
codebook_filepath = os.path.join(data_directory, codebook_filename)

# Setup
columns_with_nulls_to_drop = [
    'Moderate-work',
    'Vigorous-work',
    'Diabetes'
]

columns_to_ignore = [
    'Blood related stroke', 
    'SEQN', 
    'Year Range'
]

columns_to_impute = [
    "Mean volume of platelets", 
    "Platelet count", 
    "Red blood cell width", 
    "Hemoglobin", 
    "Red blood cells", 
    "Basophils", 
    "White blood cells",
    "Creatinine", 
    "Uric acid", 
    "Triglycerides", 
    "Protein", 
    "Bilirubin", 
    "Phosphorus", 
    "Lactate dehydrogenase (LDH)", 
    "Iron", 
    "Glucose", 
    "Gamma-glutamyl transferase (GGT)", 
    "Alkaline phosphatase (ALP)", 
    "Aspartate aminotransferase (AST)", 
    "Alanine aminotransferase (ALT)", 
    "Albumin", 
    "High-density lipoprotein (HDL)", 
    "Cholesterol", 
    "Glycohemoglobin", 
    "Diastolic", 
    "Systolic", 
    "Body mass index", 
    "Weight", 
    "Age"
]


def map_enumerated_columns(df, codebook_df):
    """
    Map enumerated columns to categorical values
    
    Args:
        df (DataFrame): The dataset
        codebook_df (dataFrame): The codebook lookup dataset
    """

    # map all non-range attributes to actual values according to codebook
    # some categorical values also have "Range of Values" as its value for some reason
    quantitative_attributes = codebook_df[codebook_df['description'] == 'Range of Values']['attribute_name'].unique()
    qualitative_attributes = [attr for attr in \
                            codebook_df[codebook_df['description'] != 'Range of Values']['attribute_name'].unique() \
                            if attr not in quantitative_attributes]

    for q_attribute in qualitative_attributes:
        df[q_attribute] = df[q_attribute].astype('object')
        print('Mapping:', q_attribute)
        for year_range in df['Year Range'].unique():
            col_year_range_codebook = codebook_df[(codebook_df['year_range'] == year_range) &
                                                (codebook_df['attribute_name'] == q_attribute)]
            
            # codebook for year_range + column doesnt exist
            if len(col_year_range_codebook) == 0:
                print(' - Unable to locate mapping for', year_range)

            # map all rows with corresponding year range and column to each value in codebook
            for index, row in col_year_range_codebook.iterrows():
                mask = (df['Year Range'] == year_range) & (df[q_attribute] == row['single_value'])
                if row['description'] != 'Range of Values':
                    df.loc[mask, q_attribute] = row['description']
    print()

    return df


def drop_null_values(df):
    """
    Drop null values from dataset
    
    Args:
        df (DataFrame): The dataset
    """

    df = df[df['Coronary heart disease'].isin(['Yes', 'No'])]

    for column in columns_with_nulls_to_drop:
        df = df[~df[column].isna()]

    return df


def ignore_columns(df):
    """
    Remove columns from dataset
    
    Args:
        df (DataFrame): The dataset
    """
    
    df = df[[col for col in df.columns if col not in columns_to_ignore]]

    return df


def impute_columns(df):
    """
    Impute columns from dataset using mean value
    
    Args:
        df (DataFrame): The dataset
    """

    for column in columns_to_impute:
        mean_value = df[column].mean()
        num_null = df[column].isna().sum()
        print(f'"{column}": replacing {num_null} missing values with: {mean_value}')
        df.loc[df[column].isna(), column] = mean_value
    print('\nNumber of remaining nulls:', df.isna().sum().sum())
    print()

    return df


if __name__ == "__main__":
    df = pd.read_parquet(original_filepath)
    codebook_df = pd.read_csv(codebook_filepath)

    df = map_enumerated_columns(df, codebook_df)

    df = drop_null_values(df)

    df = ignore_columns(df)

    df = impute_columns(df)

    df.to_parquet(resulting_filepath)