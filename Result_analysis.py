import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "Dataset")

Diff_genes = pd.read_csv(os.path.join(DATA, "Diff_data", "diff_ctr_TC.csv"), index_col=0).index.tolist()
diff_prot = pd.read_csv(os.path.join(DATA, "Diff_labels", "luminal protein estrus.csv"), index_col=0).index.tolist()
maps = pd.read_csv(os.path.join(DATA, "genetoprotein.csv"), delimiter="\t", header=None, index_col=1)

common_map = list(set(diff_prot).intersection(list(maps.index)))

genes = maps.loc[common_map, 0].tolist()

gene_names = pd.read_csv(os.path.join(DATA, "Data_cpm", "Data_control.csv"), index_col=0).index.tolist()
common_genes = list(set(genes).intersection(Diff_genes))
list1 = ["control", "0_5", "1_5", "2_5"]
siggene_names = {
    "0_5": "Siggenebasedprotlist_TCT0.5.csv",
    "1_5": "Siggenebasedprotlist_TCT1.5.csv",
    "2_5": "Siggenebasedprotlist_TCT2.5.csv",
}


def replace_value(cell_value):
    if cell_value not in Diff_genes:
        return np.nan
    return cell_value


def resolve_table(name):
    for folder in (ROOT, DATA):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Missing {name}. Result_analysis expects this table in {ROOT} or {DATA}; it is not created by the dataset loaders."
    )


for i in list1:
    TF_high_prot_2_5 = pd.read_csv(resolve_table("Tfs_allprot_" + i + ".csv"))
    if i != "control":
        geneexpbasedprot = pd.read_csv(resolve_table(siggene_names[i]), header=None)[0].to_list()
        print(geneexpbasedprot)

    col = TF_high_prot_2_5.columns.tolist()

    common_cols = [idx for idx in col if idx in common_genes]
    if i != "control":
        common_pro = [idx for idx in col if idx in geneexpbasedprot]
    else:
        common_pro = common_cols

    TF_high_prot_2_5_diff = TF_high_prot_2_5[common_pro]
    print(TF_high_prot_2_5_diff.shape)

    filtered_df2 = TF_high_prot_2_5_diff.dropna(how="all")
    filtered_df3 = filtered_df2.apply(lambda x: x.sort_values().values).dropna(how="all")
    filtered_df3.to_csv(os.path.join(ROOT, "AllTfs_diffcodinggene_" + i + ".csv"), index=0)
