from Bio import SeqIO
import pandas as pd


def to_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(f'{key}\t{value}' for key, value in data.items()))


def get_mapping():
    mapping = {}
    with open("uniprot_sprot.fasta") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            acc = record.id.split("|")#[1]
            specie = acc[2].split("_")[1]
            acc = acc[1]

            if specie == "MOUSE":
                des = record.description.split(" ")
                # print(des)
                for i in des:
                    if i.startswith("GN="):
                        gene = i.split("=")[1]
                        mapping[gene] = acc
                        # print(gene)
    return mapping


dfs = pd.read_excel("Cheng_Collaboration_data.xlsx", sheet_name="Protein from luminal fluid")

# Retain only rows with at least one value
selected_rows = dfs[(~dfs['luminal protein estrus'].isnull()) & (~dfs['luminal protein 0.5'].isnull()) \
    & (~dfs['luminal protein 1.5'].isnull()) &  (~dfs['luminal protein 2.5'].isnull()) \
    & (~dfs['luminal protein so estrus'].isnull()) &  (~dfs['luminal protein so 0.5'].isnull()) \
    & (~dfs['luminal protein so 1.5'].isnull()) &  (~dfs['luminal protein so 2.5'].isnull())]

joy_proteins = set(selected_rows['Accession'].tolist())

print(joy_proteins)

mapping = get_mapping()

mapping = {i: j for i, j in mapping.items() if j in joy_proteins}


rem = set([i for i in joy_proteins if not i in set(mapping.values())])

print(rem)
print(len(rem))

exit()

to_file(mapping, 'Data/gene2protein.csv')
print("Found {} proteins, remaining {}".format(len(mapping), len(joy_proteins) - len(mapping)))

exit()
dfs = pd.read_excel("Cheng_Collaboration_data.xlsx", sheet_name="bulkRNA-seq literature")
joy_genes = set(dfs['Accession'].tolist())

yo = set(mapping.keys())

print(len(joy_genes.intersection(yo)))








