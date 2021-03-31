#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 22:11:07 2021

@author: umreenimam
"""

"""""""""""""""
IMPORTING PACKAGES
"""""""""""""""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA 

"""""""""""""""
FUNCTIONS
"""""""""""""""
# Function to read and load data
def load_read_data(data):
    loaded_data = pd.read_excel(data)
    
    return loaded_data

# Function to remove columns
def remove_cols(data):
    proteins = data.iloc[:, 2]
    data = data.iloc[:, 9:31]
    data_t = data.transpose()
    renamed_df = data_t.rename(columns = proteins)
    
    return renamed_df

# Remove co-linearity
def remove_colinearity(data, neg_threshold):
    corr_mat = data.corr()
    row = corr_mat.shape[0]
    column = corr_mat.shape[1]
    
    correlated_features = []
    
    for x in range(row): 
        for y in range(column):
            if x == y:
                break
            if corr_mat.iloc[x, y] > abs(neg_threshold) or corr_mat.iloc[x, y] < neg_threshold:
                correlated_features.append(corr_mat.columns[x])
                break

    return correlated_features

# Normalize data using min max scaler
def min_max(data): 
    scaler = MinMaxScaler()
    normalized_corr = scaler.fit_transform(data)
    normalied_corr_df = pd.DataFrame(normalized_corr)
    
    return normalied_corr_df

def create_lineplot(data, col_nums, figure_name):
    sns.set_theme(style = 'white')
    
    plt.plot(data.iloc[:, 0], label = 'First Protein', color = 'red')
    plt.plot(data.iloc[:, 1:col_nums].mean(axis = 1), label = 'Avg. of Other Proteins', color = 'blue')
    plt.legend(loc = 'upper right')
    plt.title('Line Plot of Experimental Data', fontsize = 18)
    plt.xlabel('Row', fontsize = 12)
    plt.ylabel('Protein Value', fontsize = 12)
    plt.savefig(figure_name)
    plt.show

"""""""""""""""
Q1-Q3: Load data into dataframe and remove unecessary columns
"""""""""""""""
# Load data and read excel file
os.chdir('/Users/umreenimam/Documents/Masters/Masters_Classes/CS_5310/chapter06/lab/part1')
filename = 'Qsparse_SubjectStripTearSamples-Dec2020_Report.xls'
data_df = load_read_data(filename)

# Create new transposed dataframe
new_protein_df = remove_cols(data_df)

"""""""""""""""
Q4: Remove co-linearity
"""""""""""""""
# Call function to remove linearity
cols_to_remove = remove_colinearity(new_protein_df, -0.8)

# Remove co-linear columns 
df_remain = new_protein_df.drop(columns = cols_to_remove, axis = 1)

"""""""""""""""
Q5: Normalize remaining columns
"""""""""""""""
df_normalized = min_max(df_remain)
protein_names = list(df_remain.columns)
protein_names = pd.Series(protein_names)
df_normalized = df_normalized.rename(columns = protein_names)

"""""""""""""""
Q6: Create line plot that displays experimental data
"""""""""""""""
cols = df_normalized.shape[1]
protein_plot = create_lineplot(df_normalized, cols, 'fig1.png')

"""""""""""""""
Q7-Q8: Train & print OLS model
"""""""""""""""
# Split data into dependent and independent vars
x_vars = df_normalized.iloc[:, 1:cols]
y_vars = df_normalized.iloc[:, 0]

# Train data
res = sm.OLS(y_vars, x_vars).fit()

# Print summary
res_summary = str(res.summary())
print(res_summary)

# Below is an alternative to the OLS model 
# Trying out the PCA package instead 
"""""""""""""""
Q7-Q8: Train & print PCA model
"""""""""""""""
# Initialize PCA
pca = PCA(n_components = 20)

# Train data
pca.fit(df_normalized)

# Print summary
print(pca.explained_variance_ratio_)
print(pca.singular_values_)