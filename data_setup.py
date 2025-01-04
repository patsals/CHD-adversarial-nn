import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
import json

# Configurations
webpage_url = 'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component={}&CycleBeginYear={}'
data_attributes_file_path = "data_attributes.json"
start_years = [1999, 2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015]
output_directory = 'data_test'

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
    file_directories_path = 'file_directories.json'
    with open(file_directories_path, "r") as file:
        file_directory_dictionary = json.load(file)

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
    print(f'Checking data extracted from year range: {year_range}')

    # Read the dictionary from the JSON file
    with open(data_attributes_file_path, "r") as file:
        data_attribute_dict = json.load(file)
        
    attribute_tracker = {attribute:0 for attribute in data_attribute_dict[year_range]}

    message = ''
    for file_name in os.listdir(directory):
        data_file_path = os.path.join(directory, file_name)
        file_df = pd.read_sas(data_file_path, format='xport')
        
        for column in file_df.columns:
            if column in attribute_tracker:
                attribute_tracker[column] += 1

    for attribute, count in attribute_tracker.items():
        if count == 0:
            message += ' - Attribute: <' + attribute + '> [MISSING]\n'    
    if len(message) == 0:
        print('- All attributes checked [SUCCESS]')
    else:
        print(message.rstrip('\n'))


for start_year in start_years:
    year_range = str(start_year) + '-' + str(start_year + 1)
    print(f'~~~~EXTRACTING {year_range} DATA~~~~')

    output_file_path = f'{output_directory}/{year_range}'
    download_files_by_year_range(output_file_path, year_range)
    print()
    validate_attributes(output_file_path, year_range)

    print(f'~~~~DONE.~~~~\n')

