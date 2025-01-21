# DSC180A - Capstone Project

This repository contains all of the code that was utilized for the 2024-2025 UCSD Data Science Capstone Project.

---
## Introduction


---
## How to get Started

##### 1) Getting the code
- Clone repository into your working directory:
    `git clone https://github.com/patsals/DSC-180-Capstone.git`
- Enter the repository directory `DSC 180 Capstone`

##### 2.a) Setting up environment (Windows Users)
- Install Linux using Windows Subsystem for Linux (WSL) in powershell:
    `wsl --install`
- Install Linux Distribution System through Microsoft store:
    `Ubuntu 22.04.5 LTS`
- Open Ubuntu and wait for download to complete
- Once download complete enter a Username and Password as prompted
- Open VS Code then click on the two caret mark icon on the bottom left of window:
- Select the "Connect to WSL Using Distro...""
- You should see `Ubuntu 22.04.5 LTS` and select that option
- Proceed to 2.b instructions

##### 2.b) Setting up environment (Linux/Mac OS users)
- Create a working environment:
    `python3 -m venv venv`
- Activate the working environment:
    `source venv/bin/activate`
- Download the dependencies:
    `pip install -r requirements.txt`

##### 3) Downloading the Data
- Run the data setup file to extract, transform, and load all of the data locally:
    - `python data_setup.py` for windows users
    - `python3 data_setup.py` for Linux/Mac OS users
    - adding the `--skip_download` flag skips the requesting/downloading of files
- Note: The script makes requests to multiple url endpoints at www.sciencedirect.com to download the files. It is important to lookout for any message logs that do not show `[SUCCESS]` as an output — this indicates an error that needs to be resolved.
