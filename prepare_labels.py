"""
Step 1 — Extract and prepare labels from the Galaxy Zoo 2 catalogue.

Reads the raw Galaxy Zoo 2 catalogue (Hart et al. 2016) and produces a
clean CSV with:
  * Galaxy ID, RA, Dec
  * Debiased voting probabilities for Q1 and Q2
  * Hard (argmax) integer labels ready for training

Input  : data/raw/gz2_hart16.csv.gz   (≈240 k rows)
Output : data/interim/labels_q1_q2.csv

Galaxy Zoo Questions handled
-----------------------------
Q1  "Is the galaxy smooth, does it have features, or is it a star?"
     Classes: smooth (0), features or disk (1), star or artifact (2)

Q2  "Is the galaxy edge-on?"
     Classes: edge-on (0), not edge-on (1)
"""

import pandas as pd

from config import RAW_CATALOG_PATH, LABELS_PATH

# ──────────────────────────────────────────────
#  Column mapping from the raw catalogue
#  The original names are long GZ2 codes; we rename them to short,
#  human-readable names.
# ──────────────────────────────────────────────
RAW_COLUMNS = [
    "dr7objid",                                              # galaxy ID
    "ra", "dec",                                             # sky coords
    # Q1: smooth / features / star  (debiased vote fractions)
    "t01_smooth_or_features_a01_smooth_debiased",
    "t01_smooth_or_features_a02_features_or_disk_debiased",
    "t01_smooth_or_features_a03_star_or_artifact_debiased",
    # Q2: edge-on yes / no  (debiased vote fractions)
    "t02_edgeon_a04_yes_debiased",
    "t02_edgeon_a05_no_debiased",
]

RENAME_MAP = {
    "dr7objid":                                              "id",
    "t01_smooth_or_features_a01_smooth_debiased":            "prob_smooth",
    "t01_smooth_or_features_a02_features_or_disk_debiased":  "prob_features_or_disk",
    "t01_smooth_or_features_a03_star_or_artifact_debiased":  "prob_star_or_artifact",
    "t02_edgeon_a04_yes_debiased":                           "prob_edge_on",
    "t02_edgeon_a05_no_debiased":                            "prob_not_edge_on",
}


def main():
    # ── 1. Load raw catalogue ────────────────────────
    # Read galaxy ID as string to preserve the full 18-digit number
    # (avoids floating-point truncation that would break filename matching).
    df = pd.read_csv(
        RAW_CATALOG_PATH,
        usecols=RAW_COLUMNS,
        dtype={"dr7objid": "string"},
    )
    df = df.rename(columns=RENAME_MAP)

    # ── 2. Create hard labels via argmax ─────────────
    # Q1: pick the class with the highest vote fraction → 0 / 1 / 2
    q1_cols = ["prob_smooth", "prob_features_or_disk", "prob_star_or_artifact"]
    df["q1_label_name"] = df[q1_cols].idxmax(axis=1)
    q1_map = {"prob_smooth": 0, "prob_features_or_disk": 1, "prob_star_or_artifact": 2}
    df["q1_label"] = df["q1_label_name"].map(q1_map)

    # Q2: pick the class with the highest vote fraction → 0 / 1
    q2_cols = ["prob_edge_on", "prob_not_edge_on"]
    df["q2_label_name"] = df[q2_cols].idxmax(axis=1)
    q2_map = {"prob_edge_on": 0, "prob_not_edge_on": 1}
    df["q2_label"] = df["q2_label_name"].map(q2_map)

    # ── 3. Save ──────────────────────────────────────
    df.to_csv(LABELS_PATH, index=False)

    print(f"Saved labels for Q1 + Q2 → {LABELS_PATH}")
    print(f"  Total galaxies : {len(df):,}")
    print(f"  Q1 distribution: {df['q1_label'].value_counts().to_dict()}")
    print(f"  Q2 distribution: {df['q2_label'].value_counts().to_dict()}")
    print(df.head())


if __name__ == "__main__":
    main()
