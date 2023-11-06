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

dfs = pd.read_csv("Collaboration_data.csv",index_col=0,delimiter='\t')

df_unique = dfs[~dfs.index.isna()]


counts = GeneFilter(df_unique)
counts = counts.fillna(0.0)

df_control = counts.filter(regex='Finnerty')
df_0_5 = counts.filter(regex='0.5preg')
df_1_5 = counts.filter(regex='1.5preg')
df_2_5 = counts.filter(regex='2.5preg')

cpm_df_control = CPM().set_output(transform="pandas").fit_transform(df_control.T).T.apply(np.arcsinh)
cpm_df_0_5preg = CPM().set_output(transform="pandas").fit_transform(df_0_5.T).T.apply(np.arcsinh)
cpm_df_1_5preg = CPM().set_output(transform="pandas").fit_transform(df_1_5.T).T.apply(np.arcsinh)
cpm_df_2_5preg = CPM().set_output(transform="pandas").fit_transform(df_2_5.T).T.apply(np.arcsinh)



cpm_df_control.to_csv("Dataset/Data_cpm/Data_control.csv")
cpm_df_0_5preg.to_csv("Dataset/Data_cpm/Data_0_5preg.csv")
cpm_df_1_5preg.to_csv("Dataset/Data_cpm/Data_1_5preg.csv")
cpm_df_2_5preg.to_csv("Dataset/Data_cpm/Data_2_5preg.csv")