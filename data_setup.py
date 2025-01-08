import pandas as pd
import os
import argparse
import requests
from bs4 import BeautifulSoup
import json
import csv

# Configurations
webpage_url = 'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component={}&CycleBeginYear={}'
codebook_url = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{}/DataFiles/{}.htm#Codebook'

config_directory = "config"
data_directory = 'data'

data_attributes_file_name = "data_attributes.json"
attribute_name_map_file_name = 'data_attribute_names_map.json'
dataset_save_file_name = 'nhanes_data.parquet'
file_directories_file_name = "file_directories.json"
codebook_file_name = 'codebook.csv'

file_directories_file_path = os.path.join(config_directory, file_directories_file_name)
data_attributes_file_path = os.path.join(config_directory, data_attributes_file_name)
attribute_name_map_file_path = os.path.join(config_directory, attribute_name_map_file_name)
dataset_save_file_path = os.path.join(data_directory, dataset_save_file_name)
codebook_file_path = os.path.join(data_directory, codebook_file_name)

with open(file_directories_file_path, "r") as file:
    file_directory_dictionary = json.load(file)

with open(data_attributes_file_path, "r") as file:
    data_attribute_dict = json.load(file)

with open(attribute_name_map_file_path, "r") as file:
    attribute_name_map = json.load(file)

start_years = [1999, 2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015, 2017]
codebook_file_header = ['year_range', 'attribute', 'attribute_name', 'description', 'is_range',
                        'start_range_value', 'end_range_value', 'single_value']

def download_specific_files(url, output_folder, filename):
    """
    Downloads specific files hyperlinked on a webpage.
    
    Args:
        url (str): The URL of the webpage to scrape.
        output_folder (str): The folder to save the downloaded files.
        filenames (list): A list of specific file names to download.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    links = soup.find_all('a', href=True)

    for link in links:
        href = link['href']
        
        # Resolve relative URLs to absolute URLs
        file_url = href if href.startswith('http') else requests.compat.urljoin(url, href)
        
        # Extract the file name from the URL
        file_name = file_url.split('/')[-1]

        if file_name == filename:
            file_path = os.path.join(output_folder, file_name)
            try:
                file_response = requests.get(file_url, stream=True)
                file_response.raise_for_status() 
                
                # Save the file
                with open(file_path, 'wb') as file:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        file.write(chunk)

                return
            except Exception as e:
                print(f"Failed to download {file_url}: {e}")
                raise e
    raise Exception

def download_files_by_year_range(output_directory, year_range):
    """
    Downloads all associated files by year_range
    
    Args:
        output_directory (str): The parent directory to save all the downloaded files
        year_range (str): The year range of data to download
    """
    start_year = int(year_range.split('-')[0])

    for directory, files in file_directory_dictionary[year_range].items():
        print(f'Requesting directory: "{directory}"')
        formatted_url = webpage_url.format(directory, start_year)
        for file in files:
            print(f' - Downloading file: "{file}"', end="")
            try:
                download_specific_files(formatted_url, output_directory, file)
                print(' [SUCCESS]')
            except Exception as e:
                print(' [FAIL]')



def validate_attributes(directory, year_range):
    """
    Checks for all valid attributes for the data interested in.
    
    Args:
        directory (str): The parent directory where all the data files are located.
        year_range (str): The year range of data to validate
    """
    print(f'Checking data extracted for year range')    
        
    attribute_tracker = {attribute:0 for attribute in data_attribute_dict[year_range]}

    message = ''
    for file_name in os.listdir(directory):
        attributes_checked = []
        data_file_path = os.path.join(directory, file_name)
        file_df = pd.read_sas(data_file_path, format='xport')
        
        for column in file_df.columns:
            if column in attribute_tracker:
                attributes_checked.append(column)
                attribute_tracker[column] += 1

        print(f' - Attributes validated from {file_name}: {attributes_checked}')

    for attribute, count in attribute_tracker.items():
        if count == 0:
            message += ' - Attribute: <' + attribute + '> [MISSING]\n'    
    if len(message) == 0:
        print(' - All attributes checked [SUCCESS]')
    else:
        print(message.rstrip('\n'))


def process_attributes(directory, year_range):
    """
    Processes the dataset by extracting and renaming the desired columns.
    
    Args:
        directory (str): The parent directory where all the data files are located.
        year_range (str): The year range of data to validate
    """
    print(f'Processing data extracted for year range')

    with open(attribute_name_map_file_path, "r") as file:
        attribute_name_map = json.load(file)

    with open(data_attributes_file_path, "r") as file:
        data_attribute_dict = json.load(file)

    result_df = pd.DataFrame({'SEQN': []})
    for file_name in os.listdir(directory):
        data_file_path = os.path.join(directory, file_name)
        file_df = pd.read_sas(data_file_path, format='xport')

        # Rename columns to common naming convention
        extracted_columns = file_df.columns.intersection(data_attribute_dict[year_range])
        extracted_columns = list(extracted_columns) + ['SEQN']
        print(f' - Length of dataset extracted from {file_name}: ', len(file_df))

        df_i = file_df[extracted_columns]
        df_i = df_i.rename(columns=attribute_name_map)

        result_df = result_df.merge(df_i, how='outer', on='SEQN')

        # Add year range column
        result_df['Year Range'] = [year_range] * len(result_df)

    print(' - Merging all Datasets [SUCCESS]')
    return result_df


def write_to_csv(file_path, new_row, header=False):
    try:
        if header:
            os.remove(file_path)
    except Exception:
        pass

    try:
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(new_row)
            
    except FileNotFoundError:
        # If the file doesn't exist, create it and write the header and row
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(new_row)

def extract_codebook(year_range, first_run=False):
    """
    Processes the dataset by extracting and renaming the desired columns.
    
    Args:
        directory (str): The parent directory where all the data files are located.
        year_range (str): The year range of data to validate
    """
    print(f'Extracting codebook information for year range')

    if first_run:
        write_to_csv(codebook_file_path, codebook_file_header, header=True)

    file_dictionary = file_directory_dictionary[year_range]
    year_start = year_range.split('-')[0]
    failed = False
    for file in sum(file_dictionary.values(), []):
        file_location = file.strip('.xpt')
        url = codebook_url.format(year_start, file_location)

        print(f' - Retrieving relevant attribute information from {file_location}:', end="")
        
        try:
            num_extracted = 0
            # Make a GET request to the URL
            response = requests.get(url)
            response.raise_for_status()  # Check for request errors

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all 'div' elements with the class 'pagebreak'
            divs = soup.find_all('div', class_='pagebreak')

            # Iterate over each 'div' section
            for div in divs:
                # Extract the variable name from 'dd' element with class 'info'
                variable_name = div.find('dd', class_='info').text
                
                # We only care about storing the attributes we are interested in
                if variable_name not in data_attribute_dict[year_range]:
                    continue
            
                num_extracted += 1
                # Extract the associated table
                table = div.find('table', class_='values')
                if table:
                    # Extract table rows
                    table_rows = table.find_all('tr')
                    
                    # Extract information from table rows
                    for row in table_rows[1:]:  # Skip the header row
                        columns = row.find_all('td')
                        code_or_value = columns[0].text.strip() if columns[0] else ''
                        value_description = columns[1].text.strip() if columns[1] else ''
                        attribute_name = attribute_name_map.get(variable_name, None)
                        if 'missing' in value_description.lower():
                            continue
                    
                        # year_range, atttribute, is_range, start_range_value, end_range_value, acceptable_values
                        if 'range' in value_description.lower():
                            start_range_value = float(code_or_value.split('to')[0])
                            end_range_value = float(code_or_value.split('to')[1])
                            row = [year_range, variable_name, attribute_name, value_description, True, start_range_value, end_range_value, None]
                            write_to_csv(codebook_file_path, row)
                        else:
                            row = [year_range, variable_name, attribute_name, value_description, False, None, None, code_or_value]
                            write_to_csv(codebook_file_path, row)

            print(f' ({num_extracted} attributes)')
        except Exception:
            failed = True
            print(' [FAIL]')

    if not failed:
        print(' - Extracted codebook [SUCCESS]')
    else:
        print(' - Extracted codebook [FAIL]')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Python script with a flag")

    parser.add_argument('--skip_download', action='store_true', help='Activate the flag')
    
    args = parser.parse_args()

    print(args)
    dfs = []

    for start_year in start_years:
        year_range = str(start_year) + '-' + str(start_year + 1)
        print(f'~~~~~~~~~~~~~~~~ EXTRACTING {year_range} DATA ~~~~~~~~~~~~~~~~')

        output_file_path = f'{data_directory}/{year_range}'

        if not args.skip_download:
            download_files_by_year_range(output_file_path, year_range)
        else:
            print('Skipping download...')
        print()

        validate_attributes(output_file_path, year_range)
        print()

        df_i = process_attributes(output_file_path, year_range)
        dfs.append(df_i)
        print()

        extract_codebook(year_range, first_run=start_year==start_years[0])
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~ DONE. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')


    print(f'~~~~~~~~~~~~~~~~~~~~ SAVING ENTIRE DATA ~~~~~~~~~~~~~~~~~~~~~~')
    print('Appending all dataset years ', end="")
    processed_df = pd.concat(dfs)
    print('[SUCCESS]')
    print()
    print(f'Saving total combined dataset to: {dataset_save_file_path}', end="")
    processed_df.to_parquet(dataset_save_file_path)
    print(' [SUCCESS]')

    print(f'~~~~~~~~~~~~~~~~~~~~~~~~ COMPLETE. ~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
