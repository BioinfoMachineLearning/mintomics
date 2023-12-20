import pandas as pd
import numpy as np
def GeneFilter(df):
    df = df.fillna(0.0)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    # Filter rows where the sum of values in each row is >= 10
    rows_to_keep = df[df.sum(axis=1) >= 10]

    # Create a new DataFrame with the filtered rows
    counts_df = df.loc[rows_to_keep.index]
    return counts_df

def read_counts2tpm(df):
    """
    convert read counts to TPM (transcripts per million)
    :df: a dataFrame that contains the read count with its gene length. 
    :sample_reads: read count values for all transcripts
    :gene_len: Gene length values
    :return: TPM
    """
    result = df
    sample_reads = result.loc[:, result.columns != 'length'].copy()
    gene_len = result.loc[:, ['length']]
    normalize_by_genelength = sample_reads.values / gene_len.values
    scaling_factor = (np.sum(normalize_by_genelength, axis=0).reshape(1, -1))/1e6
    normalize_sequencingdepth = normalize_by_genelength / scaling_factor
    tpm = normalize_sequencingdepth
    return tpm

from rnanorm import CPM

#dfs = pd.read_csv("Collaboration_data.csv",index_col=0,delimiter='\t')
#
#df_unique = dfs[~dfs.index.isna()]
#
#
#counts = GeneFilter(df_unique)
#counts = counts.fillna(0.0)
#
#df_control = counts.filter(regex='Finnerty')
#df_0_5 = counts.filter(regex='0.5preg')
#df_1_5 = counts.filter(regex='1.5preg')
#df_2_5 = counts.filter(regex='2.5preg')
#
#cpm_df_control = CPM().set_output(transform="pandas").fit_transform(df_control.T).T.apply(np.arcsinh)
#cpm_df_0_5preg = CPM().set_output(transform="pandas").fit_transform(df_0_5.T).T.apply(np.arcsinh)
#cpm_df_1_5preg = CPM().set_output(transform="pandas").fit_transform(df_1_5.T).T.apply(np.arcsinh)
#cpm_df_2_5preg = CPM().set_output(transform="pandas").fit_transform(df_2_5.T).T.apply(np.arcsinh)
#
#
#
#cpm_df_control.to_csv("Dataset/Data_cpm/Data_control.csv")
#cpm_df_0_5preg.to_csv("Dataset/Data_cpm/Data_0_5preg.csv")
#cpm_df_1_5preg.to_csv("Dataset/Data_cpm/Data_1_5preg.csv")
#cpm_df_2_5preg.to_csv("Dataset/Data_cpm/Data_2_5preg.csv")

import torch
import math
def z_score(exp_mat):
            exp_mat.iloc[:,1] = np.log10(exp_mat.iloc[:,1])
            
            mean_sam = np.mean(exp_mat.iloc[:,1], axis = 0)
            std_sam = np.std(exp_mat.iloc[:,1], axis = 0)
            #print(mean_sam.shape)
            exp_mat.iloc[:,1] = ((exp_mat.iloc[:,1] - mean_sam)/std_sam)
         
            return exp_mat

def min_max_normalize(df1):
    """Normalize a dataframe to the range [0,1]."""
    df = np.log10(df1.iloc[:,0])
    df1.iloc[:,0] = (df - df.min()) / (df.max() - df.min()) 
    return df1      

def min_max_normalize1(df1):
    """Normalize a dataframe to the range [0,1]."""
    df = np.log10(df1)
    df1 = (df - df.min()) / (df.max() - df.min()) 
    return df1     
import matplotlib.pyplot as plt

def sigmoid(x):
    sum1 = max(x)
    x = x / sum1
    return 1 / (1 + np.exp(-x))

df_labels = pd.read_csv("Labels_orig.csv")
df_labels.set_index(df_labels.iloc[:,0], inplace=True)
df_labels_nat = df_labels.iloc[:,1:5]
df_labels_nat= df_labels_nat.dropna(how="all")
df_labels_nat = df_labels_nat.fillna(1)

from mlxtend.preprocessing import minmax_scaling
df_labels_nat1 = np.log10(df_labels_nat)
df_labels_nat2 = minmax_scaling(df_labels_nat1, columns=['luminal protein estrus', 'luminal protein 0.5','luminal protein 1.5','luminal protein 2.5'])
#df_labels_nat = min_max_normalize1(df_labels_nat)
df_labels_nat2.to_csv("Dataset/Labels_norm_overall.csv")



#df_label_control = pd.read_csv('Dataset/Labels/Labels_control.csv',delimiter='\t')
#print(len(df_label_control))
df_label_control = pd.DataFrame(df_labels_nat['luminal protein estrus'])
#df_label_control = df_label_control.fillna(1)
df_label_control = min_max_normalize(df_label_control)
print(len(df_label_control))
#df_label_control.iloc[:,1] = sigmoid(df_label_control.iloc[:,1])
plt.plot(df_label_control.iloc[:,0])

df_label_control.to_csv('Dataset/Labels_proc_log10_minmax/Labels_control.csv')


df_label_0_5 = pd.DataFrame(df_labels_nat['luminal protein 0.5'])
#df_label_0_5 = df_label_0_5.fillna(1)
df_label_0_5 = min_max_normalize(df_label_0_5)
print(len(df_label_0_5))
plt.plot(df_label_0_5.iloc[:,0])
df_label_0_5.to_csv('Dataset/Labels_proc_log10_minmax/Labels_0_5preg.csv')

df_label_1_5 = pd.DataFrame(df_labels_nat['luminal protein 1.5'])
#df_label_0_5 = df_label_1_5.fillna(1)
df_label_0_5 = min_max_normalize(df_label_1_5)
print(len(df_label_1_5))
plt.plot(df_label_1_5.iloc[:,0])
df_label_1_5.to_csv('Dataset/Labels_proc_log10_minmax/Labels_1_5preg.csv')

df_label_2_5 = pd.DataFrame(df_labels_nat['luminal protein 2.5'])
#df_label_2_5 = df_label_2_5.fillna(1)
df_label_2_5 = min_max_normalize(df_label_2_5)
print(len(df_label_2_5))
plt.plot(df_label_2_5.iloc[:,0])
df_label_2_5.to_csv('Dataset/Labels_proc_log10_minmax/Labels_2_5preg.csv')


#df_label_0_5 = pd.read_csv('Dataset/Labels/Labels_0_5preg.csv',delimiter='\t')
#df_label_0_5 = df_label_0_5.dropna()
#df_label_0_5 = z_score(df_label_0_5)
#print(len(df_label_control))
#df_label_0_5.to_csv('Dataset/Labels_proc_log10/Labels_0_5preg.csv',index=None)
#
#df_label_1_5 = pd.read_csv('Dataset/Labels/Labels_1_5preg.csv',delimiter='\t')
#df_label_1_5 = df_label_1_5.dropna()
#df_label_1_5 = z_score(df_label_1_5)
#print(len(df_label_control))
#df_label_1_5.to_csv('Dataset/Labels_proc_log10/Labels_1_5preg.csv',index=None)
#
#df_label_2_5 = pd.read_csv('Dataset/Labels/Labels_2_5preg.csv',delimiter='\t')
#df_label_2_5 = df_label_2_5.dropna()
#df_label_2_5 = z_score(df_label_2_5)
#print(len(df_label_control))
#df_label_2_5.to_csv('Dataset/Labels_proc_log10/Labels_2_5preg.csv',index=None)