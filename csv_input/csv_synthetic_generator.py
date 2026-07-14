
import pandas as pd
import numpy as np
from copulas.multivariate import GaussianMultivariate


def compare_datasets(real_df, synthetic_df, target_col=None):
    """
    Compare real and synthetic datasets.

    Parameters
    ----------
    real_df : pd.DataFrame
    synthetic_df : pd.DataFrame
    target_col : str, optional
        Binary target column (e.g. 'stable_unstable')
    """

    print("\n" + "=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)

    print(f"Real dataset:      {real_df.shape}")
    print(f"Synthetic dataset: {synthetic_df.shape}")

    # ====================================================
    # NUMERIC VARIABLES
    # ====================================================

    numeric_cols = real_df.select_dtypes(
        include=np.number
    ).columns.intersection(synthetic_df.columns)

    print("\n" + "=" * 80)
    print("NUMERIC VARIABLES")
    print("=" * 80)

    comparison_rows = []

    for col in numeric_cols:

        comparison_rows.append({
            "Variable": col,

            "Real Mean":
                round(real_df[col].mean(), 3),

            "Synth Mean":
                round(synthetic_df[col].mean(), 3),

            "Mean Diff":
                round(
                    synthetic_df[col].mean()
                    - real_df[col].mean(),
                    3
                ),

            "Real Std":
                round(real_df[col].std(), 3),

            "Synth Std":
                round(synthetic_df[col].std(), 3),

            "Real Min":
                round(real_df[col].min(), 3),

            "Synth Min":
                round(synthetic_df[col].min(), 3),

            "Real Max":
                round(real_df[col].max(), 3),

            "Synth Max":
                round(synthetic_df[col].max(), 3),
        })

    numeric_summary = pd.DataFrame(comparison_rows)

    print(numeric_summary)

    # ====================================================
    # CATEGORICAL VARIABLES
    # ====================================================

    categorical_cols = [
        c for c in real_df.columns
        if c not in numeric_cols
        and c in synthetic_df.columns
    ]

    print("\n" + "=" * 80)
    print("CATEGORICAL VARIABLES")
    print("=" * 80)

    for col in categorical_cols:

        print(f"\n--- {col} ---")

        real_freq = (
            real_df[col]
            .value_counts(normalize=True)
            .sort_index()
        )

        synth_freq = (
            synthetic_df[col]
            .value_counts(normalize=True)
            .sort_index()
        )

        comparison = pd.concat(
            [real_freq, synth_freq],
            axis=1
        )

        comparison.columns = [
            "Real %",
            "Synthetic %"
        ]

        comparison = comparison.fillna(0)

        print(comparison.round(3))

    # ====================================================
    # BINARY VARIABLES
    # ====================================================

    binary_cols = []

    for col in numeric_cols:

        vals = set(real_df[col].dropna().unique())

        if vals.issubset({0, 1}):
            binary_cols.append(col)

    if binary_cols:

        print("\n" + "=" * 80)
        print("BINARY VARIABLES")
        print("=" * 80)

        rows = []

        for col in binary_cols:

            rows.append({
                "Variable": col,
                "Real Positive Rate":
                    round(real_df[col].mean(), 3),
                "Synthetic Positive Rate":
                    round(synthetic_df[col].mean(), 3),
            })

        print(pd.DataFrame(rows))

    # ====================================================
    # CORRELATIONS
    # ====================================================

    if len(numeric_cols) > 1:

        print("\n" + "=" * 80)
        print("CORRELATION PRESERVATION")
        print("=" * 80)

        corr_real = real_df[numeric_cols].corr()
        corr_synth = synthetic_df[numeric_cols].corr()

        diff = (corr_synth - corr_real).abs()

        avg_corr_error = diff.values.mean()

        print(
            f"Average absolute correlation difference: "
            f"{avg_corr_error:.4f}"
        )

        print("\nLargest differences:")

        diff_long = (
            diff.stack()
            .reset_index()
        )

        diff_long.columns = [
            "Var1",
            "Var2",
            "Difference"
        ]

        diff_long = diff_long[
            diff_long["Var1"] != diff_long["Var2"]
        ]

        print(
            diff_long
            .sort_values(
                "Difference",
                ascending=False
            )
            .head(10)
        )

    # ====================================================
    # TARGET STRATIFICATION
    # ====================================================

    if target_col and target_col in real_df.columns:

        print("\n" + "=" * 80)
        print(f"STRATIFIED ANALYSIS: {target_col}")
        print("=" * 80)

        numeric_for_strat = [
            c for c in numeric_cols
            if c != target_col
        ]

        for col in numeric_for_strat:

            print(f"\n--- {col} ---")

            real_grouped = (
                real_df
                .groupby(target_col)[col]
                .mean()
            )

            synth_grouped = (
                synthetic_df
                .groupby(target_col)[col]
                .mean()
            )

            comparison = pd.concat(
                [real_grouped, synth_grouped],
                axis=1
            )

            comparison.columns = [
                "Real Mean",
                "Synthetic Mean"
            ]

            print(comparison.round(3))

    real_df.groupby("Rupture")["PHASE"].mean()
    synthetic_df.groupby("Rupture")["PHASE"].mean()

    real_df.groupby("Rupture")["ELAPSS"].mean()
    synthetic_df.groupby("Rupture")["ELAPSS"].mean()

    real_df["Rupture"].mean()
    synthetic_df["Rupture"].mean()

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


def preprocess_real_dataset(df: pd.DataFrame):
    """
    Clean a real-world clinical dataset before synthetic generation.

    Returns:
        cleaned_df
        report (dict)
    """

    df = df.copy()

    report = {
        "dropped_columns": [],
        "missing_values_before": df.isna().sum().to_dict(),
    }

    # --------------------------------------------------
    # 1. Drop likely identifier columns
    # --------------------------------------------------

    id_keywords = [
        "id",
        "patientid",
        "patient_id",
        "subjectid",
        "subject_id",
        "recordid",
        "record_id",
        "studyid",
        "study_id",
        "mrn",
    ]

    cols_to_drop = []

    for col in df.columns:

        col_lower = str(col).lower().replace(" ", "").replace("-", "_")

        if col_lower in id_keywords:
            cols_to_drop.append(col)
            continue

        # Drop almost-unique identifier columns
        unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)

        if unique_ratio > 0.95:
            cols_to_drop.append(col)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    report["dropped_columns"] = cols_to_drop

    # --------------------------------------------------
    # 2. Convert booleans and common binary encodings
    # --------------------------------------------------

    bool_cols = df.select_dtypes(include=["bool"]).columns

    for col in bool_cols:
        df[col] = df[col].astype(int)
    

    # --------------------------------------------------
    # 3. Convert date columns to numeric
    # --------------------------------------------------

    for col in df.columns:

        if "date" in str(col).lower():

            try:
                dt = pd.to_datetime(df[col], errors="coerce")

                if dt.notna().sum() > 0:

                    # days since first date
                    reference = dt.min()

                    df[col] = (
                        dt - reference
                    ).dt.days

            except Exception:
                pass

    # --------------------------------------------------
    # 4. Handle missing values
    # --------------------------------------------------

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = [
        c for c in df.columns
        if c not in numeric_cols
    ]

    # Numeric -> median
    for col in numeric_cols:

        if df[col].isna().any():

            median_value = df[col].median()

            if pd.isna(median_value):
                median_value = 0

            df[col] = df[col].fillna(median_value)

    # Categorical -> mode
    for col in categorical_cols:

        if df[col].isna().any():

            mode_values = df[col].mode()

            if len(mode_values):
                fill_value = mode_values.iloc[0]
            else:
                fill_value = "Unknown"

            df[col] = df[col].fillna(fill_value)

    # --------------------------------------------------
    # 5. Remove columns that are entirely missing
    # --------------------------------------------------

    empty_cols = df.columns[df.isna().all()].tolist()

    if empty_cols:

        df = df.drop(columns=empty_cols)

        report["dropped_columns"].extend(empty_cols)

    report["missing_values_after"] = df.isna().sum().to_dict()

    return df, report


def infer_schema_from_dataframe(df: pd.DataFrame):

    schema = {"columns": {}, "constraints": []}

    for col in df.columns:

        s = df[col].dropna()

        if len(s) == 0:
            continue

        if pd.api.types.is_numeric_dtype(s):

            unique_values = set(s.unique())

            # -----------------------------------------
            # Binary variables (0/1)
            # -----------------------------------------

            if unique_values.issubset({0, 1}):

                schema["columns"][col] = {
                    "type": "binary",
                    "prob": float(s.mean())
                }

                continue

            # -----------------------------------------
            # Low-cardinality integers -> category
            # -----------------------------------------

            if (
                pd.api.types.is_integer_dtype(s)
                and len(unique_values) <= 10
            ):

                values = sorted(list(unique_values))

                probs = (
                    s.value_counts(normalize=True)
                    .reindex(values)
                    .fillna(0)
                    .tolist()
                )

                schema["columns"][col] = {
                    "type": "category",
                    "values": values,
                    "probs": probs,
                }

                continue

            # -----------------------------------------
            # Integer
            # -----------------------------------------

            if pd.api.types.is_integer_dtype(s):

                schema["columns"][col] = {
                    "type": "int",
                    "range": [int(s.min()), int(s.max())],
                }

            # -----------------------------------------
            # Float
            # -----------------------------------------

            else:

                schema["columns"][col] = {
                    "type": "float",
                    "range": [float(s.min()), float(s.max())],
                    "dist": "truncated_normal",
                    "mean": float(s.mean()),
                    "sd": float(s.std()) if float(s.std()) > 0 else 1.0,
                }

        else:

            values = list(s.astype(str).unique())

            probs = (
                s.astype(str)
                .value_counts(normalize=True)
                .reindex(values)
                .fillna(0)
                .tolist()
            )

            schema["columns"][col] = {
                "type": "category",
                "values": values,
                "probs": probs,
            }

    return schema


def generate_synthetic_from_real(real_df: pd.DataFrame, n_rows=None):
    if n_rows is None:
        n_rows = len(real_df)

    schema = infer_schema_from_dataframe(real_df)

    synthetic = pd.DataFrame(index=range(n_rows))

    for col, spec in schema["columns"].items():
        
        if spec["type"] == "binary":
            prob = spec.get("prob", 0.5)
            synthetic[col] = np.random.binomial(
                1,
                prob,
                size=n_rows
            )

        elif spec["type"] == "category":
            synthetic[col] = pd.Series(
                np.random.choice(
                    spec["values"],
                    size=n_rows,
                    p=spec.get("probs")
                )
            )

        elif spec["type"] == "int":
            vals = np.random.randint(
                spec["range"][0],
                spec["range"][1] + 1,
                size=n_rows,
            )
            synthetic[col] = vals

        elif spec["type"] == "float":
            vals = np.random.normal(
                spec.get("mean", 0),
                spec.get("sd", 1),
                size=n_rows,
            )
            vals = np.clip(vals, spec["range"][0], spec["range"][1])
            synthetic[col] = vals

    
    binary_cols = [
        c for c, spec in schema["columns"].items()
        if spec["type"] == "binary"
    ]

    copula_cols = [
        c for c, spec in schema["columns"].items()
        if spec["type"] in ["int", "float"]
    ]

    if len(copula_cols) > 1:
        copula = GaussianMultivariate()
        copula.fit(real_df[copula_cols])

        sampled_numeric = copula.sample(n_rows)

        for col in copula_cols:
            synthetic[col] = sampled_numeric[col].values

            spec = schema["columns"][col]
            synthetic[col] = synthetic[col].clip(
                spec["range"][0],
                spec["range"][1],
            )

            if spec["type"] == "int":
                synthetic[col] = synthetic[col].round().astype(int)

    return synthetic, schema
