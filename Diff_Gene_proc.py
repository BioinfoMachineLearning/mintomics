import pandas as pd
import numpy as np
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
def GeneFilter(df):
    df = df.fillna(0.0)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    # Filter rows where the sum of values in each row is >= 10
    rows_to_keep = df[df.sum(axis=1) >= 10]

    # Create a new DataFrame with the filtered rows
    counts_df = df.loc[rows_to_keep.index]
    return counts_df


dfs = pd.read_csv("JOYS_Project/mintomics/Collaboration_data.csv",index_col=0,delimiter='\t')

df_unique = dfs[~dfs.index.isna()]
#
#
counts = GeneFilter(df_unique)
counts = counts.fillna(0.0)

dfs_3pts = counts.loc[:, ~counts.columns.str.contains('3.5')]
dfs_3pts_nops1 = dfs_3pts.loc[:, ~dfs_3pts.columns.str.contains('pseudo')]
dfs_3pts_nops = dfs_3pts_nops1.T
print(dfs_3pts_nops.head())
substring_map = {
    'Finnerty': 'TC',
    '_0.5preg': 'T0.5',
    '_1.5preg': 'T1.5',
    '_2.5preg': 'T2.5'
}

# Initialize an empty list
result_list = []

# Iterate over the dictionary keys and check if the index contains the substring
#for substring, value in substring_map.items():
#    #result_list.extend([value] * dfs_3pts_nops.index.str.contains(substring).sum())
#    result_list.extend([substring_map[substring] for idx in dfs_3pts_nops.index if substring in idx])
for idx in dfs_3pts_nops.index:
    for key in substring_map.keys():
        print(key)
    
        if key in idx:
            print(idx)
            result_list.append(substring_map[key])
print(result_list)
maps = pd.read_csv("/home/aghktb/JOYS_Project/mintomics/output_n.csv",delimiter="\t",header=None,index_col=1)
print(maps)
all_prot = pd.read_csv("/home/aghktb/JOYS_Project/mintomics/Labels_orig.csv")
#


prots = all_prot['Accession'].tolist()
common_map = list(set(prots).intersection(list(maps.index)))
genes = maps.loc[common_map,0].tolist()


metadata = pd.DataFrame(zip(dfs_3pts_nops.index, result_list),
                        columns = ['Sample', 'Condition'])
metadata = metadata.set_index('Sample')
print(metadata)
dds = DeseqDataSet(counts=dfs_3pts_nops,metadata=metadata,design_factors="Condition")
dds.deseq2()
print(dds)
selected_genes=[]
for co in substring_map.values():
    if co!="TC":
        stat_res = DeseqStats(dds, contrast = ('Condition',co,'TC'))
        stat_res.summary()

        res = stat_res.results_df

        
    # Adjusted p-value threshold
        p_value_threshold = 0.05
    # Log2 fold change threshold
        log2FC_threshold = 0.1
        sel_ge = list(res[(res['padj'] < p_value_threshold) & (abs(res['log2FoldChange']) > log2FC_threshold) ].index)
    # Select genes based on adjusted p-value, log2 fold change, and base mean
        common_map = pd.DataFrame(list(set(sel_ge).intersection(list(genes))))
        #common_prot = pd.DataFrame(maps[maps[0].isin(common_map)].index.tolist())
        common_map.to_csv("/home/aghktb/JOYS_PROJECT/mintomics/Siggenebasedprotlist_TC"+co+".csv",index=None,header=None)
        
            

        selected_genes.extend(sel_ge)
        
        
        #selected_genes.to_csv("/home/aghktb/JOYS_Project/mintomics/Dataset/Diff_data/diff_ctr_"+co+".csv")
    # Print the selected genes
        
        
exit()        
        
diff_data = dfs_3pts_nops1.loc[list(set(selected_genes)),:]
for co in substring_map.values():
    cols = metadata.index[metadata['Condition']==co]
    print(cols)
    diff_data1 = diff_data.loc[:,cols]

    print(diff_data1.shape) 
   # diff_data1.to_csv("/home/aghktb/JOYS_Project/mintomics/Dataset/Diff_data/diff_ctr_"+co+".csv")
#print(diff_data.shape)
      
def min_max_normalize(df1):
    """Normalize a dataframe to the range [0,1]."""
    df = np.log10(df1.iloc[:,0])
    df1.iloc[:,0] = (df - df.min()) / (df.max() - df.min())
    print(df1) 
    return df1  
import protrank as pt
ignore_ind = ['luminal protein so estrus','luminal protein so 0.5','luminal protein so 1.5','luminal protein so 2.5']
data_p = pt.load_data("/home/aghktb/JOYS_Project/mintomics/Labels_orig.csv",ignore_cols=ignore_ind)
print(data_p.shape)

what_to_compare = [[['luminal protein estrus','luminal protein 0.5'], ['luminal protein estrus', 'luminal protein 1.5'], ['luminal protein estrus', 'luminal protein 2.5']]]
description = 'TC_vs_T0.5_sample_dataset'

pt.data_stats(data_p, what_to_compare)
significant_proteins =pt.rank_proteins(data_p, what_to_compare, description)

from mlxtend.preprocessing import minmax_scaling
significant_df = data_p.loc[significant_proteins,:]

#significant_df = np.log10(significant_df)
print(significant_df.head())
for col in significant_df.columns:    
    significant_df1 = min_max_normalize(pd.DataFrame(significant_df[col]))
    print(significant_df1)
    #significant_df1.to_csv('/home/aghktb/JOYS_Project/mintomics/Dataset/Diff_labels/'+col+'.csv')
print(significant_df1)

maps = pd.read_csv("/home/aghktb/JOYS_Project/mintomics/output_n.csv",delimiter="\t",header=None,index_col=1)

common_map = list(set(significant_proteins).intersection(list(maps.index)))
print(len(common_map))
genes = maps.loc[common_map,0].tolist()
common_genes = list(set(genes).intersection(list(dfs_3pts_nops1.index)))
print(len(common_genes))

selected_genes.extend(common_genes)
diff_data = dfs_3pts_nops1.loc[list(set(selected_genes)),:]


from rnanorm import CPM
for co in substring_map.values():
    cols = metadata.index[metadata['Condition']==co]
    #print(cols)
    diff_data1 = diff_data.loc[:,cols]
    diff_data_cpm = CPM().set_output(transform="pandas").fit_transform(diff_data1.T).T.apply(np.arcsinh)

    
    #diff_data1.to_csv("/home/aghktb/JOYS_Project/mintomics/Dataset/Diff_data/diff_ctr_"+co+".csv")



'''
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
results=[]
for i in list(data_p.index):
    control_data = data_p['luminal protein estrus']
    test_data = data_p['luminal protein 0.5']
    t_stat, p_val = ttest_ind(control_data[i], test_data[i])
    results.append({'Timepoint': 0.5, 'T-statistic': t_stat, 'P-value': p_val})

    print(results)
    # Correct for multiple testing
    p_values = [result['P-value'] for result in results]
    reject, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
    for i, result in enumerate(results):
        result['Adjusted P-value'] = p_corrected[i]
        result['Significant'] = reject[i]
    print(result)

import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FactorVector

# Activate conversion between pandas DataFrame and R data.frame
pandas2ri.activate()

# Load the limma library
limma = importr("limma")

# Read the data
df = pd.read_csv("your_data_file.txt", sep="\t", index_col=0)

# Create a design matrix
design = limma.model_matrix("~0 + factor(c(rep('Control', {0}), rep('Timepoint0.5', {1}), rep('Timepoint1.5', {2}), rep('Timepoint2.5', {3})))".format(
    df.shape[1]-1, df.shape[1]-1, df.shape[1]-1, df.shape[1]-1))

# Fit the linear model
fit = limma.lmFit(pandas2ri.py2rpy(df), design)

# Apply empirical Bayes smoothing
fit = limma.eBayes(fit)

# Perform differential expression analysis
contrast_matrix = limma.makeContrasts(
    "Timepoint0.5-Control", "Timepoint1.5-Control", "Timepoint2.5-Control", levels=design)
results = limma.contrasts_fit(fit, contrast_matrix)
results = limma.eBayes(results)

# Get the differentially expressed proteins
diff_proteins = limma.topTable(results, coef="Timepoint0.5-Control", number=n_proteins_to_keep)
'''