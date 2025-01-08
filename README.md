# DSC180A - Capstone Project

This repository contains all of the code that was utilized for the 2024-2025 UCSD Data Science Capstone Project.

---
## Introduction


---
## How to get Started

##### 1) Getting the code
- clone repository into your working directory:
    `git clone https://github.com`

##### 2.a) Setting up environment (Windows Users)
- create a working environment:
    `python -m venv venv`
- activate the working environment:
    `venv/scripts/activate`
- download the dependencies:
    `pip install -r requirements.txt`

##### 2.b) Setting up environment (Linux/Mac OS users)
- create a working environment:
    `python3 -m venv venv`
- activate the working environment:
    `source venv/bin/activate`
- download the dependencies:
    `pip install -r requirements.txt`

##### 3) Downloading the Data
- Run the data setup file to extract, transform, and load all of the data locally:
    - `python data_setup.py` for windows users
    - `python3 data_setup.py` for Linux/Mac OS users
    - adding the `--skip_download` flag skips the requesting/downloading of files
- Note: The script makes requests to multiple url endpoints at www.sciencedirect.com to download the files. It is important to lookout for any message logs that dont show `[SUCCESS]` as an output — this indicates an error that needs to be resolved.
