import pandas as pd

import os
import requests
from bs4 import BeautifulSoup
import json


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


# 1999-2000
# file_dictionary = {
#     'Demographics': ["DEMO.xpt"],
#     "Examination": ["BPX.xpt", "BMX.xpt"],
#     "Laboratory": ["LAB25.xpt", "LAB18.xpt", "LAB13.xpt", "LAB18.xpt", "LAB10.xpt"],
#     "Questionnaire": ["PAQ.xpt", "DIQ.xpt", "MCQ.xpt"]
# }

# 2001-2002: B
# file_dictionary = {
#     'Demographics': ["DEMO_B.xpt"],
#     "Examination": ["BPX_B.xpt", "BMX_B.xpt"],
#     "Laboratory": ["L25_B.xpt", "L10_B.xpt", "L10_2_B.xpt", "L13_B.xpt", "L40_B.xpt", "L40_2_B.xpt"],
#     "Questionnaire": ["PAQ_B.xpt", "DIQ_B.xpt", "MCQ_B.xpt"]
# }

# 2003-2004: C
# file_dictionary = {
#     'Demographics': ["DEMO_C.xpt"],
#     "Examination": ["BPX_C.xpt", "BMX_C.xpt"],
#     "Laboratory": ["L25_C.xpt", "L10_C.xpt", "L13_C.xpt", "L40_C.xpt"],
#     "Questionnaire": ["PAQ_C.xpt", "DIQ_C.xpt", "MCQ_C.xpt"]
# }


# # 2005-2006: D
# # 2007-2008: E
# # 2009-2010: F
# # 2011-2012: G
# # 2013-2014: H
# # 2015-2016: I
# file_dictionary = {
#     'Demographics': ["DEMO_I.xpt"],
#     "Examination": ["BPX_I.xpt", "BMX_I.xpt"],
#     "Laboratory": ["BIOPRO_I.xpt", "CBC_I.xpt", "GHB_I.xpt", "HDL_I.xpt", 'TCHOL_I.xpt'],
#     "Questionnaire": ["PAQ_I.xpt", "DIQ_I.xpt", "MCQ_I.xpt"]
# }


def download_files_by_year_range(output_directory, year_range):
    webpage_url = 'https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component={}&CycleBeginYear={}'
    start_year = int(year_range.split('-')[0])
    file_directories_path = 'file_directories.json'
    with open(file_directories_path, "r") as file:
        file_directory_dictionary = json.load(file)

    for directory, files in file_directory_dictionary[year_range].items():
        print(f'Reading directory: "{directory}"')
        formatted_url = webpage_url.format(directory, start_year)
        for file in files:
            print(f' - Reading in file: "{file}"', end="")
            try:
                download_specific_files(formatted_url, output_directory, file)
                print(' [SUCCESS]')
            except Exception as e:
                print(' [FAIL]')



def check_attributes(directory, year_key):
    file_path = "data_attributes.json"

    print(f'Checking data extracted from year range: {year_key}')
    # Read the dictionary from the JSON file
    with open(file_path, "r") as file:
        data_attribute_dict = json.load(file)
        
    attribute_tracker = {attribute:0 for attribute in data_attribute_dict[year_key]}

    for file_name in os.listdir(directory):
        try:
            file_path = os.path.join(directory, file_name)

            file_df = pd.read_sas(file_path, format='xport')
            
            for column in file_df.columns:
                if column in attribute_tracker:
                    attribute_tracker[column] += 1

        except Exception as e:
            print(' [FAILED]\n')

    for attribute, count in attribute_tracker.items():
        if count == 0:
            print(f' - Attribute: {attribute} [MISSING]')
    
    print(f'Checking done.')



output_directory = 'data_test'

for start_year in [1999, 2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015]:
    year_range = str(start_year) + '-' + str(start_year + 1)
    print(f'EXTRACTING {year_range} DATA')

    output_file_path = f'{output_directory}/{year_range}'
    download_files_by_year_range(output_file_path, year_range)
    check_attributes(output_file_path, year_range)

    print(f'DONE.\n')

# check_attributes('data/2001_2002/', '2001-2002')
# check_attributes('data/2003_2004/', '2003-2004')
# check_attributes('data/2005_2006/', '2005-2006')
# check_attributes('data/2007_2008/', '2007-2008')
# check_attributes('data/2009_2010/', '2009-2010')
# check_attributes('data/2011_2012/', '2011-2012')
# check_attributes('data/2013_2014/', '2013-2014')
# check_attributes('data/2015_2016/', '2015-2016')
