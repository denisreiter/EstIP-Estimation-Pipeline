"""Model-tied feature configuration, shared across notebooks and scripts."""

# Unique building/row identifier column
ID_COL = "bldg_gmlid"

# Target column
LABEL_COL = "MAPPED_POPULATION"

# Columns used as model input features
FEATURES = [
    "CNT_PRTS",
    "RF_TYPE",
    "FNC_CODE",
    "SHPX_2D",
    "RATIO_C_A",
    "RATIO_LW_MRR",
    "RATIO_MRR_A",
    "CNT_NDS",
    "CNT_HLS",
    "A",
    "C",
    "H",
    "V",
    "RATIO_HLL_V",
    "DIST_NGHB",
    "RATIO_SW",
    "CNT_NGHB_R50",
    "DIST_NGHB_R50",
    "DIST_STRT",
    "CNT_ADD",
    "BLCK_TYPE",
    "CNT_BLCK",
    "A_SUM_BLCK",
    "A_MEAN_BLCK",
    "A_STD_BLCK",
    "V_SUM_BLCK",
    "V_MEAN_BLCK",
    "V_STD_BLCK",
    "MEAN_DIST_NGHB_BLCK",
    "RATIO_A_BLCK",
    "RATIO_A_SUM",
    "RATIO_V_SUM",
    "RATIO_A_MEAN",
    "RATIO_V_MEAN",
    "MUNICIP",
    "REG_TYPE"
]

# Subset of FEATURES that are categorical (must be numerically formatted)
CATEGORICAL_COLS = [
    "RF_TYPE",
    "FNC_CODE",
    "BLCK_TYPE",
    "MUNICIP",
    "REG_TYPE"
]

# Grouping column for spatial cross-validation (GroupKFold)
SPATIAL_CV_COL = "blck_id"