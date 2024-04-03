import pandas as pd
import numpy as np
Diff_genes = pd.read_csv("/home/aghktb/JOYS_Project/mintomics/Dataset/Diff_data/diff_ctr_TC.csv",index_col=0).index.tolist()

diff_prot = pd.read_csv("/home/aghktb/JOYS_Project/mintomics/Dataset/Diff_labels/luminal protein estrus.csv",index_col=0).index.tolist()
maps = pd.read_csv("/home/aghktb/JOYS_Project/mintomics/output_n.csv",delimiter="\t",header=None,index_col=1)

common_map = list(set(diff_prot).intersection(list(maps.index)))

genes = maps.loc[common_map,0].tolist()

gene_names = pd.read_csv("/home/aghktb/JOYS_Project/mintomics/Dataset/Data_cpm/Data_control.csv",index_col=0).index.tolist()
common_genes = list(set(genes).intersection(Diff_genes))
list1 = ["control","0_5","1_5","2_5"]
 # Function to replace cell value
def replace_value(cell_value):
        if cell_value not in Diff_genes:
            return np.nan  # or return '-'
        return cell_value
for i in list1:
    TF_high_prot_2_5 = pd.read_csv("/home/aghktb/JOYS_PROJECT/mintomics/Tfs_allprot_"+i+".csv")
    if i!="control":
        geneexpbasedprot = pd.read_csv("/home/aghktb/JOYS_PROJECT/mintomics/Siggenebasedprotlist_TCT"+i+".csv",header=None)[0].to_list()
        print(geneexpbasedprot)

    #print(TF_high_prot_2_5.head())
    
    col = TF_high_prot_2_5.columns.tolist()
    
    common_cols = [idx for idx in col if idx in common_genes]
    if i!="control":
        #common_pro = list(set(common_cols).intersection(geneexpbasedprot))
        common_pro=[idx for idx in col if idx in geneexpbasedprot]
    else:
         common_pro = common_cols
    
    TF_high_prot_2_5_diff = TF_high_prot_2_5[common_pro]
    print(TF_high_prot_2_5_diff.shape)

    #filtered_df = TF_high_prot_2_5_diff[[col for col in TF_high_prot_2_5_diff.columns if any(gene in Diff_genes for gene in TF_high_prot_2_5_diff[col])]]


    #filtered_d1 = TF_high_prot_2_5_diff.applymap(replace_value)
    filtered_df2 = TF_high_prot_2_5_diff.dropna(how='all')
    filtered_df3 = filtered_df2.apply(lambda x: x.sort_values().values).dropna(how='all')
    filtered_df3.to_csv("/home/aghktb/JOYS_PROJECT/mintomics/AllTfs_diffcodinggene_"+i+".csv",index=0)