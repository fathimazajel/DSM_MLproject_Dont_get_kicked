#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 19:40:24 2023

@author: fathimazajel
"""
import preprocessing
import input_clean_columns

#Execute the function from the script 
input_clean_columns.process_data()

# Preprocessing 
train_path = 'temp/training.csv'
test_path = 'temp/test.csv'
output_train_path = 'temp/preprocessed_train.csv'
output_test_path = 'temp/preprocessed_test.csv'

train_df, test_df = preprocessing.preprocess_data(train_path, test_path)

preprocessing.write_data(train_df, output_train_path)
preprocessing.write_data(test_df, output_test_path)