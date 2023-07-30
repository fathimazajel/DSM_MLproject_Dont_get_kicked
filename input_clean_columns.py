#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 19:48:34 2023

@author: fathimazajel
"""

import pandas as pd
import os

def process_data():
    # Define file paths for input and temp folders
    input_folder = 'input'
    output_folder = 'temp'
    

    # Import train.csv and test.csv into dataframes
    train_df = pd.read_csv(os.path.join(input_folder, 'training.csv'))
    test_df = pd.read_csv(os.path.join(input_folder, 'test.csv'))
    
    # Convert all columns to lowercase
    train_df.columns = train_df.columns.str.lower()
    test_df.columns = test_df.columns.str.lower()
    
    # Export train and test dataframes as csv files in the temp folder
    train_df.to_csv(os.path.join(output_folder, 'training.csv'), index=False)
    test_df.to_csv(os.path.join(output_folder, 'test.csv'), index=False)
    
    # Print success message
    print("Files successfully imported, converted to lowercase, and exported to the temp folder.")
