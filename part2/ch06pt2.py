#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 13:03:59 2021

@author: umreenimam
"""

"""""""""""""""
IMPORTING PACKAGES
"""""""""""""""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import tree

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
    data = data.iloc[:, 10:32]
    data_t = data.transpose()
    renamed_df = data_t.rename(columns = proteins)
    
    return renamed_df

# Zero fill-in rows function 
def zero_fill(data):
    filled_in_data = data.replace('Filtered', 0.0)
    
    return filled_in_data

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

    return corr_mat, correlated_features

# Normalize data using min max scaler
def min_max(data): 
    scaler = MinMaxScaler()
    normalized_corr = scaler.fit_transform(data)
    normalied_corr_df = pd.DataFrame(normalized_corr)
    
    return normalied_corr_df

# Function to create line plot
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
    
# Creation of heatmap function
def create_heatmap(data, figure_name):
    sns.set_theme(style = "white")
    
    plt.figure(figsize=(20, 20))
    color_map = sns.diverging_palette(230, 20, as_cmap = True)
    sns.heatmap(data, annot = False, cmap = color_map, vmax = 1, 
                center = 0, square = True, linewidths = 0.1,
                cbar_kws = {"shrink": 0.75})
    plt.title('Heat Map of Correlation Coefficient Matrix', fontsize = 18)
    plt.xlabel('Row Number from the Data Frame', fontsize = 12)
    plt.ylabel('Column Number from the Data Frame', fontsize = 12)
    plt.savefig(figure_name)
    plt.show

"""""""""""""""
Q1-Q2: Load data into dataframe and remove unecessary columns
"""""""""""""""
# Load data and read excel file
os.chdir('/Users/umreenimam/Documents/Masters/Masters_Classes/CS_5310/chapter06/lab/part2')
filename = 'Data_SubjectStripTearSamples-Dec2020_Report.xls'
data_df = load_read_data(filename)

# Create new transposed dataframe
new_protein_df = remove_cols(data_df)

"""""""""""""""
Q3: Zero fill-in rows with 'filtered' in the cells
"""""""""""""""
# Calling zero_fill function
zero_filled_df = zero_fill(new_protein_df)

"""""""""""""""
Q4-Q6: Remove co-linearity and normalize remaining columns
"""""""""""""""
# Calling remove_colinearity function
# Reducing threshold value to 0.60 for data modeling
# Original threshold of 0.80 resulted in a dataframe with 714 columns
correlation_matrix, cols_to_remove = remove_colinearity(zero_filled_df, -0.60)

# Dropping columns
df_remain = zero_filled_df.drop(columns = cols_to_remove, axis = 1)

# Normalize using min_max function
df_normalized = min_max(df_remain)

# Renaming columns
protein_names = list(df_remain.columns)
protein_names = pd.Series(protein_names)
df_normalized = df_normalized.rename(columns = protein_names)

"""""""""""""""
Q7: Create lineplot
"""""""""""""""
cols = df_normalized.shape[1]
protein_plot = create_lineplot(df_normalized, cols, 'fig1.png')

"""""""""""""""
Q8: Create correlation matrix of all columns
"""""""""""""""
# Creating correlations for normalized data
df_corr_normalized = df_normalized.corr()

# Creating heatmap of normalized correlation
df_corr_norm_hm = create_heatmap(df_corr_normalized, 'fig2.png')

"""""""""""""""
Q9: Train a Decision Tree Regressor
"""""""""""""""
x_vars = df_normalized.iloc[:, 1:cols]
y_vars = df_normalized.iloc[:, 0]

regressor = DecisionTreeRegressor(max_depth = 3, random_state = 42)
model = regressor.fit(x_vars, y_vars)

"""""""""""""""
Q10: Create a visualization of the tree
"""""""""""""""
fig = plt.figure(figsize = (40, 40))
tree_vis = tree.plot_tree(regressor, filled = True)
fig.savefig('fig3.png')

"""""""""""""""
Q11: Make predictions for dataset
"""""""""""""""
prediction = regressor.predict(x_vars)

"""""""""""""""
Q12: Compute correlation and mean absolute error (MAE)
"""""""""""""""
correlation, p_value = pearsonr(prediction, y_vars)
mae = mean_absolute_error(prediction, y_vars)