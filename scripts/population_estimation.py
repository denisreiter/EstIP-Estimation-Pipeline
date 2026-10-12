"""Estimate population for a given year using a trained model.

Usage:
    python scripts/population_estimation.py --config <path/to/your/config/estimation_config.yaml>
"""

import argparse
import hmac
import hashlib

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd

from utils.checkpoint import Checkpoint
from utils.data_loading import load_feature_list, load_gpkg, load_model, load_run_config
from utils.train_config import ID_COL


def load_reference_table(
    reference_file, 
    ags_col, 
    municipalities_file
):
    """
    Loads the reference population (AGS, EINW) and municipality name (GEN) tables from Excel, 
    normalizes both AGS codes to 8 digits (right-padding the reference, left-padding the municipalities), 
    and returns their inner join on ags_col.
    """
    reference = pd.read_excel(reference_file, usecols=["AGS", "EINW"], dtype={"AGS": str, "EINW": float})
    municipalities = pd.read_excel(municipalities_file, usecols=["AGS", "GEN"], dtype={"AGS": str, "GEN": str})

    reference[ags_col] = reference["AGS"].str.ljust(8, "0")
    municipalities[ags_col] = municipalities["AGS"].str.zfill(8)

    merged_reference = reference.merge(municipalities, on=ags_col, how="inner")

    return merged_reference


def record_key(
    building_id, 
    person_idx: int, 
    salt: bytes
) -> float:
    """
    Derives a deterministic pseudo-random key in [0, 1) for one person record by 
    HMAC-SHA256 hashing building_id:person_idx with a secret salt and scaling 
    the first 8 digest bytes by 2^64.
    """
    msg = f"{building_id}:{person_idx}".encode()
    digest = hmac.new(salt, msg, hashlib.sha256).digest()
    n = int.from_bytes(digest[:8], "big")
    return n / 2**64


def compute_ckm_cell_stats(
    buildings_df: pd.DataFrame,
    salt: bytes,
    count_col: str = "RND_PREDICTION",
    id_col: str = "bldg_gmlid",
    grid_col: str = "MAPPED_CELLCODE",
) -> pd.DataFrame:
    """
    Expands building-level counts into individual person records, assigns each a record key, 
    and aggregates per grid cell into the true count and the cell key (the sum of record keys modulo 1). 
    Cells with no residents are kept with a count and key of 0.
    """
    counts = buildings_df[count_col].to_numpy()
    building_ids = buildings_df[id_col].to_numpy()
    grid_ids = buildings_df[grid_col].to_numpy()

    rep_building = np.repeat(building_ids, counts)
    rep_grid = np.repeat(grid_ids, counts)
    person_idx = np.concatenate([np.arange(c) for c in counts]) if counts.sum() else np.array([])

    keys = np.array([record_key(b, p, salt) for b, p in zip(rep_building, person_idx)])
    person_df = pd.DataFrame({"grid_id": rep_grid, "record_key": keys})

    cell_stats = person_df.groupby("grid_id").agg(
        true_count=("record_key", "size"),
        cell_key=("record_key", lambda s: s.sum() % 1.0),
    ).reset_index()

    all_cells = pd.Series(buildings_df[grid_col].unique(), name="grid_id")
    cell_stats = all_cells.to_frame().merge(cell_stats, on="grid_id", how="left")
    cell_stats["true_count"] = cell_stats["true_count"].fillna(0).astype(int)
    cell_stats["cell_key"] = cell_stats["cell_key"].fillna(0.0)

    return cell_stats


def apply_ptable_noise(
    cell_stats: pd.DataFrame, 
    ptab: pd.DataFrame
) -> pd.DataFrame:
    """
    Looks up the noise value v in the perturbation table for each cell, using the cell’s count 
    (capped at the table maximum) and its cell key. It adds the noise to the true count and floors 
    the result at 0 to yield published_count.
    """
    max_i = ptab["i"].max()
    cell_stats = cell_stats.copy()
    cell_stats["capped_count"] = cell_stats["true_count"].clip(upper=max_i)

    ptab_sorted = ptab.sort_values(["i", "p_int_lb"]).reset_index(drop=True)

    def lookup_noise(i: int, key: float) -> int:
        rows = ptab_sorted[ptab_sorted["i"] == i]
        match = rows[(rows["p_int_lb"] <= key) & (key < rows["p_int_ub"])]
        if match.empty:
            return 0
        return int(match.iloc[0]["v"])

    cell_stats["noise"] = [
        lookup_noise(i, k) for i, k in zip(cell_stats["capped_count"], cell_stats["cell_key"])
    ]
    cell_stats["published_count"] = (cell_stats["true_count"] + cell_stats["noise"]).clip(lower=0)
    return cell_stats


def run_ckm_pipeline(
    buildings_df: pd.DataFrame,
    salt: bytes,
    ptable_csv_path: str,
    count_col: str = "RND_PREDICTION",
    id_col: str = "bldg_gmlid",
    grid_col: str = "MAPPED_CELLCODE",
) -> pd.DataFrame:
    """
    Runs the full Cell-Key Method pipeline: it reads the perturbation table from CSV, 
    computes the per-cell true counts and cell keys from the building data, 
    and applies the noise to produce anonymised cell counts.
    """
    ptab = pd.read_csv(ptable_csv_path)
    cell_stats = compute_ckm_cell_stats(buildings_df, salt, count_col, id_col, grid_col)
    return apply_ptable_noise(cell_stats, ptab)


def run(
    config, 
    cp
):
    """
    Predicts building-level population with the trained model, optionally applies a threshold and a 
    per-municipality correction towards the reference totals, and exports 
    building-level and grid-cell estimates as CSVs. It also writes the CKM-anonymised cell counts, 
    and returns the paths of the output files.
    """
    year = config["year"]
    model_path = config["model"]
    features_path = config["features"]
    input_gpkg = config["input_gpkg"]
    ags_col = config["ags_col"]
    cell_col = config["cell_col"]
    reference_file = config.get("reference_file")
    municipalities_file = config.get("municipalities_file")
    salt_hex = config.get("salt_hex")
    ptable = config.get("ptable_file")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_correction = config.get("apply_correction", True)
    replace_threshold = config.get("replace_threshold", True)
    threshold = config.get("threshold", 0.5)
    cell_rounding = config.get("cell_rounding", True)

    feature_spec = load_feature_list(features_path)
    features = feature_spec["features"]
    best_iteration = feature_spec.get("best_iteration")
    categories = feature_spec.get("categories")

    if categories is None:
        print("WARNING: no categories in the feature artifact; categorical codes will be "
              "re-derived from this file and may not match the ones the model was trained on")

    data = load_gpkg(input_gpkg, layer="s5_spatial_assignment", categories=categories)

    id_table = data[[ID_COL, ags_col, cell_col, "REG_TYPE"]].copy()
    id_table[ags_col] = id_table[ags_col].astype(str).str.zfill(8)
    id_table["REG_TYPE"] = id_table["REG_TYPE"].astype(str)
    cp.mark("Data loaded")

    model = load_model(model_path, device=config.get("device", "cuda"))

    if best_iteration is None:
        print("WARNING: no best_iteration in the feature artifact; predicting with all trees")
        iteration_range = None
    else:
        iteration_range = (0, best_iteration + 1)

    predictions = model.predict(data[features], iteration_range=iteration_range)
    cp.mark("Predictions made")

    print(f"Total predicted population: {predictions.sum()}")

    calculation_table = id_table.copy()
    calculation_table["PREDICTION"] = predictions

    if replace_threshold:
        calculation_table["PREDICTION"] = calculation_table["PREDICTION"].mask(
            calculation_table["PREDICTION"] < threshold, 0
        )

    value_col = "PREDICTION"

    if apply_correction:
        reference = load_reference_table(reference_file, ags_col, municipalities_file)
        ref_total = reference["EINW"].sum()
        print(f"Total reference population: {ref_total}")

        non_zero = calculation_table[calculation_table["PREDICTION"] > 0]
        agg = non_zero.groupby(ags_col).agg(
            AGS_BUILDING_CNT=(ID_COL, "count"),
            AGS_PREDICTION_SUM=("PREDICTION", "sum"),
        ).reset_index()

        agg_reference = agg.merge(reference, on=ags_col, how="left", validate="1:1")

        unmatched = agg_reference["EINW"].isna().sum()
        if unmatched:
            print(
                f"WARNING: {unmatched} municipalities have predictions but no matching "
                f"reference value; those predictions are left uncorrected rather than "
                f"silently written as NaN."
            )

        agg_reference["AGS_CORR_FAC"] = (
            (agg_reference["EINW"] - agg_reference["AGS_PREDICTION_SUM"]) / agg_reference["AGS_BUILDING_CNT"]
        )
        agg_reference["AGS_CORR_FAC"] = agg_reference["AGS_CORR_FAC"].fillna(0)

        calculation_table = calculation_table.merge(
            agg_reference[[ags_col, "AGS_CORR_FAC"]], on=ags_col, how="left"
        )
        calculation_table["AGS_CORR_FAC"] = calculation_table["AGS_CORR_FAC"].fillna(0)

        calculation_table["CORR_PREDICTION"] = calculation_table["PREDICTION"].where(
            calculation_table["PREDICTION"] == 0,
            calculation_table["PREDICTION"] + calculation_table["AGS_CORR_FAC"],
        )

        calculation_table["CORR_PREDICTION"] = calculation_table["CORR_PREDICTION"].clip(lower=0)

        if replace_threshold:
            calculation_table["CORR_PREDICTION"] = calculation_table["CORR_PREDICTION"].mask(
                calculation_table["CORR_PREDICTION"] < threshold, 0
            )

        corr_total = calculation_table["CORR_PREDICTION"].sum()
        print(f"Total predicted population after correction: {corr_total}")
        print(f"Difference between corrected predictions and reference population: {corr_total - ref_total}")

        value_col = "CORR_PREDICTION"

    cp.mark("Correction applied" if apply_correction else "No correction applied")

    output_cols = [ID_COL, cell_col, ags_col, value_col]
    output_table = calculation_table[output_cols]
    output_csv = output_dir / f"{year}_population_estimates.csv"
    output_table.to_csv(output_csv, sep=";", index=False)

    estip_buildings = output_table[[ID_COL, value_col]].rename(
        columns={ID_COL: "bldg_gmlid", value_col: "estimated_population"}
    )
    estip_buildings_csv = output_dir / f"EstIP_{year}.csv"
    estip_buildings.to_csv(estip_buildings_csv, sep=";", index=False)

    cp.mark("Buildings exported")

    cell_data = calculation_table.groupby(cell_col, as_index=False).agg(
        **{value_col: (value_col, "sum"), "REG_TYPE": ("REG_TYPE", "first")}
    )
    cell_data["POPULATION_EST"] = cell_data[value_col].round(0) if cell_rounding else cell_data[value_col]

    cell_csv = output_dir / f"{year}_population_estimates_cells.csv"
    cell_data.to_csv(cell_csv, sep=";", index=False)

    cp.mark("Raw cells exported")

    output_table["RND_PREDICTION"] = output_table["CORR_PREDICTION"].round().astype(int)
    salt = bytes.fromhex(salt_hex)
    raw_estip_cells = run_ckm_pipeline(output_table, salt, ptable, "RND_PREDICTION", ID_COL, cell_col)
    raw_estip_cells_csv = output_dir / f"{year}_raw_anonymised_cells.csv"
    raw_estip_cells.to_csv(raw_estip_cells_csv, sep=";", index=False)
    
    estip_cells = raw_estip_cells[["grid_id", "published_count"]].rename(
        columns={"grid_id": "CellCode", "published_count": "estimated_population"}
    )
    estip_cells_csv = output_dir / f"EstIP_cells_{year}.csv"
    estip_cells.to_csv(estip_cells_csv, sep=";", index=False)

    cp.mark("Anonymised cells exported")

    return output_csv, cell_csv, estip_buildings_csv, estip_cells_csv


def main():
    parser = argparse.ArgumentParser(description="Predict population for a given year.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cp = Checkpoint("Start")
    config = load_run_config(args.config)
    run(config, cp)
    cp.finish()


if __name__ == "__main__":
    main()