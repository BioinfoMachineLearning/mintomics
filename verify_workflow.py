"""Record the current mintomics workflow without changing loader logic."""
import csv
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "verification_report.json"
SEED = 42


def run_cmd(args, timeout=60):
    try:
        completed = subprocess.run(
            args,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout[-20000:],
            "stderr": completed.stderr[-20000:],
        }
    except Exception as exc:
        return {"returncode": None, "error": repr(exc)}


def read_table(path):
    text = path.read_text(errors="replace").splitlines()
    if not text:
        return [], []
    sep = "\t" if "\t" in text[0] else ","
    rows = list(csv.reader(text, delimiter=sep))
    return rows[0], rows[1:]


def tf_indicator(gene_ids, tf_names):
    ordered = [str(i).upper() for i in gene_ids]
    tf_set = {str(name).upper() for name in tf_names if str(name).strip()}
    flags = [1.0 if gene in tf_set else 0.0 for gene in ordered]
    absent = sorted(name for name in tf_set if name not in set(ordered))
    return {
        "n_expression_rows": len(ordered),
        "n_tf_symbols": len(tf_set),
        "n_tf_flags": int(sum(flags)),
        "n_tf_symbols_absent_from_expression": len(absent),
        "construction": "Each expression-table row keeps its order. The TF flag is 1 when that row symbol is in Dataset/Mouse_TFs1.",
    }


def mask_behavior():
    import numpy as np

    source = np.arange(12, dtype=np.float32).reshape(4, 3)
    dat = source.copy()
    views = []
    gene_dat = dat
    masks = [np.array([0, 2]), np.array([1])]
    for gene_mask in masks:
        gene_dat[gene_mask] = 0.0
        gene_dat = np.array(gene_dat, copy=True)
        views.append(gene_dat.copy())
    fresh = []
    base = source.copy()
    for gene_mask in masks:
        copied = np.array(base, copy=True)
        copied[gene_mask] = 0.0
        fresh.append(copied)
    return {
        "source_rows_changed_by_alias_assignment": int(np.any(dat != source)),
        "view2_contains_view1_zeros": bool(np.all(views[1][0] == 0)),
        "fresh_copies_keep_unmasked_rows": bool(
            np.allclose(fresh[1][0], source[0]) and np.allclose(fresh[1][2], source[2])
        ),
        "loader_statement": "masked = np.array(dat, copy=True); masked[gene_mask] = 0.0; Gene_dat = torch.tensor(masked)",
    }


def scale_expression(values, include_last_column):
    import numpy as np

    matrix = np.array(values, dtype=np.float32)
    if include_last_column:
        block = matrix
    else:
        block = matrix[:, :-1]
    percentile = np.percentile(block, 99.9)
    scaled = np.clip(block, 0.0, percentile) / percentile
    if include_last_column:
        return scaled
    return np.concatenate((scaled, matrix[:, -1:]), axis=1)


def class_balance(label_path, thresholds):
    header, rows = read_table(label_path)
    if len(header) < 2:
        return {"error": "fewer than 2 columns", "columns": header}
    values = []
    for row in rows:
        text = row[1].strip()
        if len(row) < 2 or text == "":
            continue
        try:
            values.append(float(text))
        except ValueError:
            continue
    result = {"n_proteins": len(values), "abundance_column": header[1]}
    for threshold in thresholds:
        positives = sum(value > threshold for value in values)
        result[str(threshold)] = {
            "positive_count": positives,
            "positive_fraction": positives / len(values) if values else None,
        }
    return result


def mapping_rate(expression_path, label_path, map_path):
    gene_header, gene_rows = read_table(expression_path)
    label_header, label_rows = read_table(label_path)
    _, map_rows = read_table(map_path)
    gene_col = 0
    genes = {row[gene_col].upper() for row in gene_rows if row}
    accessions = {row[0].upper() for row in label_rows if row}
    pairs = []
    for row in map_rows:
        if len(row) < 2:
            continue
        pairs.append((row[0].upper(), row[1].upper()))
    mapped_genes = {gene for gene, _ in pairs if gene in genes}
    mapped_accessions = {acc for _, acc in pairs if acc in accessions}
    retained = [(gene, acc) for gene, acc in pairs if gene in genes and acc in accessions]
    return {
        "expression_file": str(expression_path.relative_to(ROOT)),
        "expression_columns": gene_header,
        "n_expression_genes": len(genes),
        "n_label_accessions": len(accessions),
        "n_map_rows": len(pairs),
        "mapped_gene_rate": len(mapped_genes) / len(genes) if genes else None,
        "mapped_accession_rate": len(mapped_accessions) / len(accessions) if accessions else None,
        "retained_pairs": len(retained),
        "retained_pair_rate_of_map_rows": len(retained) / len(pairs) if pairs else None,
    }


def file_status(path):
    target = ROOT / path
    return {"path": path, "exists": target.exists(), "bytes": target.stat().st_size if target.exists() else None}


def main():
    report = {
        "seed": SEED,
        "seed_locations": [
            "Training.py pl.seed_everything(42)",
            "Inference.py pl.seed_everything(42)",
        ],
        "abundance_threshold_defaults": {
            "Training.py": 0.8,
            "Inference.py": 0.8,
            "PrepareDataset.py": 0.8,
            "prepare_test_data.py": 0.8,
            "Diff_dataset.py": 0.6,
            "Diff_dataset_test.py": 0.6,
        },
        "inference": {
            "timepoint_choices": ["0.5", "1.5", "2.5", "all"],
            "default_timepoint": "2.5",
            "output_dir": "intermediate_files",
            "outputs_per_stage": [
                "Tfs_allprot_<tag>.csv",
                "inference_confusion_matrix_<tag>.png",
                "inference_attention_<tag>.png",
            ],
            "stage_tag": "timepoint with '.' replaced by '_'",
        },
        "siggene": {
            "script": "src/preprocess/Diff_Gene_proc.py",
            "output_dir": "intermediate_files",
            "rule": "DESeq2 padj < 0.05 and absolute log2 fold change > 0.1 versus Finnerty control TC, then genes mapped to Labels_orig accessions",
            "files": [
                "Siggenebasedprotlist_TCT0.5.csv",
                "Siggenebasedprotlist_TCT1.5.csv",
                "Siggenebasedprotlist_TCT2.5.csv",
            ],
        },
    }

    os_release = Path("/etc/os-release")
    report["environment"] = {
        "os_uname": " ".join(platform.uname()),
        "os_release": os_release.read_text() if os_release.exists() else None,
        "python": sys.version,
    }
    report["git"] = {
        "commit": run_cmd(["git", "rev-parse", "HEAD"]),
        "tag": run_cmd(["git", "describe", "--tags", "--always"]),
    }
    report["gpu"] = run_cmd(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"]
    )
    report["packages"] = run_cmd([sys.executable, "-m", "pip", "freeze"], timeout=120)

    dataset = ROOT / "Dataset"
    tf_path = dataset / "Mouse_TFs1"
    tf_names = [line.strip() for line in tf_path.read_text().splitlines() if line.strip()]
    cpm_control = dataset / "Data_cpm" / "Data_control.csv"
    cpm_header, cpm_rows = read_table(cpm_control)
    cpm_ids = [row[0] for row in cpm_rows if row]
    report["tf_indicator"] = {
        "file": str(cpm_control.relative_to(ROOT)),
        "columns": cpm_header,
        "loader": "PrepareDataset reads Dataset/Mouse_TFs1 and sets the flag on each expression-table row",
        "measurement": tf_indicator(cpm_ids, tf_names),
    }

    report["masking"] = mask_behavior()

    import numpy as np

    def cell(value):
        text = "" if value is None else str(value).strip()
        if text == "":
            return 0.0
        return float(text)

    numeric = np.array(
        [[cell(value) for value in row[1:7]] + [0.0] for row in cpm_rows if len(row) >= 7],
        dtype=np.float32,
    )
    numeric[:50, -1] = 1.0
    train_scaled = scale_expression(numeric, include_last_column=False)
    test_scaled = scale_expression(numeric, include_last_column=False)
    report["scaling"] = {
        "file": "Dataset/Data_cpm/Data_control.csv",
        "train_validation": "PrepareDataset scales the six RNA columns with a 99.9 percentile clip and divide, then appends the TF flag unchanged",
        "held_out": "prepare_test_data uses the same RNA-only scaling and appends the TF flag unchanged",
        "same_procedure": bool(np.allclose(train_scaled, test_scaled)),
        "max_abs_difference": float(np.max(np.abs(train_scaled - test_scaled))),
    }

    thresholds = [0.6, 0.8]
    label_dir = dataset / "Labels_proc_log10_minmax"
    report["class_balance"] = {
        path.name: class_balance(path, thresholds)
        for path in sorted(label_dir.glob("Labels_*.csv"))
    }

    map_path = dataset / "genetoprotein.csv"
    report["mapping_rate"] = {
        "train_control_cpm": mapping_rate(cpm_control, label_dir / "Labels_control.csv", map_path),
        "held_out_2_5_cpm": mapping_rate(
            dataset / "Data_cpm" / "Data_2_5preg.csv",
            label_dir / "Labels_2_5preg.csv",
            map_path,
        ),
    }

    expected = [
        "Dataset/Data_cpm/Data_control.csv",
        "Dataset/Data_cpm/Data_0_5preg.csv",
        "Dataset/Data_cpm/Data_1_5preg.csv",
        "Dataset/Data_cpm/Data_2_5preg.csv",
        "Dataset/Data/Data_control.csv",
        "Dataset/Data/Data_0_5preg.csv",
        "Dataset/Data/Data_1_5preg.csv",
        "Dataset/Data/Data_2_5preg.csv",
        "Dataset/Labels_proc_log10_minmax/Labels_control.csv",
        "Dataset/Labels_proc_log10_minmax/Labels_0_5preg.csv",
        "Dataset/Labels_proc_log10_minmax/Labels_1_5preg.csv",
        "Dataset/Labels_proc_log10_minmax/Labels_2_5preg.csv",
        "Dataset/genetoprotein.csv",
        "Dataset/Mouse_TFs1",
        "Dataset/Diff_data/diff_ctr_TC.csv",
        "Dataset/Diff_data/diff_ctr_T0.5.csv",
        "Dataset/Diff_data/diff_ctr_T1.5.csv",
        "Dataset/Diff_data/diff_ctr_T2.5.csv",
        "Dataset/Diff_labels/luminal protein estrus.csv",
        "Dataset/Diff_labels/luminal protein 0.5.csv",
        "Dataset/Diff_labels/luminal protein 1.5.csv",
        "Dataset/Diff_labels/luminal protein 2.5.csv",
        "intermediate_files/Tfs_allprot_0_5.csv",
        "intermediate_files/Tfs_allprot_1_5.csv",
        "intermediate_files/Tfs_allprot_2_5.csv",
        "intermediate_files/inference_confusion_matrix_0_5.png",
        "intermediate_files/inference_confusion_matrix_1_5.png",
        "intermediate_files/inference_confusion_matrix_2_5.png",
        "intermediate_files/inference_attention_0_5.png",
        "intermediate_files/inference_attention_1_5.png",
        "intermediate_files/inference_attention_2_5.png",
        "intermediate_files/Siggenebasedprotlist_TCT0.5.csv",
        "intermediate_files/Siggenebasedprotlist_TCT1.5.csv",
        "intermediate_files/Siggenebasedprotlist_TCT2.5.csv",
    ]
    report["paths"] = [file_status(path) for path in expected]
    diff_header, _ = read_table(dataset / "Diff_data" / "diff_ctr_TC.csv")
    prot_header, _ = read_table(dataset / "Diff_labels" / "luminal protein estrus.csv")
    report["differential_table_columns"] = {
        "Dataset/Diff_data/diff_ctr_TC.csv": diff_header,
        "Dataset/Diff_labels/luminal protein estrus.csv": prot_header,
        "Result_analysis_reads": [
            "Dataset/Diff_data/diff_ctr_TC.csv index_col=0",
            "Dataset/Diff_labels/luminal protein estrus.csv index_col=0",
            "Dataset/genetoprotein.csv tab-separated, no header, index_col=1",
            "Dataset/Data_cpm/Data_control.csv index_col=0",
            "Tfs_allprot_{control,0_5,1_5,2_5}.csv resolved from the repository root, then Dataset/",
            "Siggenebasedprotlist_TCT{0.5,1.5,2.5}.csv resolved from the repository root, then Dataset/",
        ],
        "Result_analysis_does_not_search": "intermediate_files",
        "Result_analysis_writes": "AllTfs_diffcodinggene_{control,0_5,1_5,2_5}.csv",
    }

    report["loader_attempts"] = {}
    for threshold in thresholds:
        section = {}
        for name, module_name, call in (
            ("train", "src.dataset.PrepareDataset", "gene2protein(stage='train', size=1, pertage=0.0, abundance_threshold=threshold)"),
            ("held_out", "src.dataset.prepare_test_data", "gene2protein(stage='test', size=1, pertage=0.0, abundance_threshold=threshold)"),
        ):
            try:
                module = __import__(module_name, fromlist=["gene2protein"])
                started = time.perf_counter()
                data, target, info, target_classify = module.gene2protein(
                    stage="train" if name == "train" else "test",
                    size=1,
                    pertage=0.0,
                    abundance_threshold=threshold,
                )
                section[name] = {
                    "seconds": time.perf_counter() - started,
                    "data_shape": list(data.shape),
                    "target_shape": list(target.shape),
                    "info_shape": list(info.shape),
                    "target_classify_shape": list(target_classify.shape),
                    "positive_count": int(target_classify.sum()),
                }
            except Exception:
                section[name] = {"error": traceback.format_exc()[-4000:]}
        report["loader_attempts"][str(threshold)] = section

    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    report["training_runs"] = {}
    for threshold, save_dir in ((0.6, "verify_thr_0.6"), (0.8, "verify_thr_0.8")):
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "Training.py",
                    "--num_epochs",
                    "1",
                    "--unit_test",
                    "1",
                    "--num_gpus",
                    "1",
                    "--batch_size",
                    "1",
                    "--abundance_threshold",
                    str(threshold),
                    "--save_dir",
                    save_dir,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=180,
                env=env,
            )
            outcome = {
                "returncode": completed.returncode,
                "seconds": time.perf_counter() - started,
                "stdout": completed.stdout[-20000:],
                "stderr": completed.stderr[-20000:],
            }
        except Exception as exc:
            outcome = {"returncode": None, "seconds": time.perf_counter() - started, "error": repr(exc)}
        checkpoint_dir = ROOT / "Trainings" / save_dir
        checkpoints = sorted(str(path.relative_to(ROOT)) for path in checkpoint_dir.glob("*.ckpt")) if checkpoint_dir.exists() else []
        metric_files = sorted(str(path.relative_to(ROOT)) for path in checkpoint_dir.rglob("*.csv")) if checkpoint_dir.exists() else []
        outcome["checkpoints"] = checkpoints
        outcome["metric_csv_files"] = metric_files
        report["training_runs"][str(threshold)] = outcome

    forward = {}
    try:
        import torch
        from src.dataset.prepare_test_data import gene2protein as held_out_gene2protein
        from src.model.Model import TransformerMintomics

        held = held_out_gene2protein(stage="test", size=1, pertage=0.0, abundance_threshold=0.8)
        tensor = held[0][:1]
        n_proteins = held[1].shape[1]
        model = TransformerMintomics(n_class=n_proteins)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        tensor = tensor.to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.no_grad():
            logits, attention = model(tensor)
        forward = {
            "source": "prepare_test_data.gene2protein held-out Data_2_5preg.csv, pertage 0",
            "input_shape": list(tensor.shape),
            "logit_shape": list(logits.shape),
            "attention_shape": list(attention.shape) if torch.is_tensor(attention) else str(type(attention)),
            "seconds": time.perf_counter() - started,
            "device": str(device),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
        }
    except Exception:
        forward = {"error": traceback.format_exc()[-4000:]}
    report["held_out_forward"] = forward

    def dir_bytes(path):
        total = 0
        if not path.exists():
            return 0
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
        return total

    report["storage_bytes"] = {
        "Dataset": dir_bytes(dataset),
        "Trainings": dir_bytes(ROOT / "Trainings"),
        "intermediate_files": dir_bytes(ROOT / "intermediate_files"),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
