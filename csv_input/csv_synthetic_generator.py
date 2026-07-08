
import pandas as pd
import numpy as np
from copulas.multivariate import GaussianMultivariate



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
