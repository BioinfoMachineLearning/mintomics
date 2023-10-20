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

mapping = get_mapping()
ls = set(mapping.values())
# to_file(mapping, 'output.csv')


dfs = pd.read_excel("Cheng_Collaboration_data.xlsx", sheet_name="Protein from luminal fluid")
joy_proteins = set(dfs['Accession'].tolist())

print(len(joy_proteins), len(ls), len(joy_proteins.intersection(ls)))








