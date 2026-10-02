import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
INTERMEDIATE = os.path.join(ROOT, "intermediate_files")
DATA = os.path.join(ROOT, "Dataset")

STAGES = ["0_5", "1_5", "2_5"]
SIGGENE_NAMES = {
    "0_5": "Siggenebasedprotlist_TCT0.5.csv",
    "1_5": "Siggenebasedprotlist_TCT1.5.csv",
    "2_5": "Siggenebasedprotlist_TCT2.5.csv",
}


def resolve_table(name):
    for folder in (INTERMEDIATE, ROOT, DATA):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Missing {name}. Looked in {INTERMEDIATE}, {ROOT}, and {DATA}."
    )


def main():
    os.makedirs(INTERMEDIATE, exist_ok=True)
    for stage in STAGES:
        ranked = pd.read_csv(resolve_table(f"Tfs_allprot_{stage}.csv"))
        significant_genes = set(
            pd.read_csv(resolve_table(SIGGENE_NAMES[stage]), header=None)[0].tolist()
        )
        kept = [column for column in ranked.columns if column in significant_genes]
        filtered = ranked[kept].dropna(how="all")
        sorted_columns = filtered.apply(lambda column: column.sort_values().values).dropna(how="all")
        out_path = os.path.join(INTERMEDIATE, f"AllTfs_diffcodinggene_{stage}.csv")
        sorted_columns.to_csv(out_path, index=False)
        print(out_path, sorted_columns.shape)


if __name__ == "__main__":
    main()
