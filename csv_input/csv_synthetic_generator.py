
import pandas as pd
from copulas.multivariate import GaussianMultivariate


def infer_schema_from_dataframe(df: pd.DataFrame):
    schema = {"columns": {}, "constraints": []}

    for col in df.columns:
        s = df[col].dropna()

        if pd.api.types.is_numeric_dtype(s):
            if pd.api.types.is_integer_dtype(s):
                schema["columns"][col] = {
                    "type": "int",
                    "range": [int(s.min()), int(s.max())],
                }
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
        if spec["type"] == "category":
            synthetic[col] = pd.Series(
                pd.np.random.choice(
                    spec["values"],
                    size=n_rows,
                    p=spec.get("probs")
                )
            )

        elif spec["type"] == "int":
            vals = pd.np.random.randint(
                spec["range"][0],
                spec["range"][1] + 1,
                size=n_rows,
            )
            synthetic[col] = vals

        elif spec["type"] == "float":
            vals = pd.np.random.normal(
                spec.get("mean", 0),
                spec.get("sd", 1),
                size=n_rows,
            )
            vals = pd.np.clip(vals, spec["range"][0], spec["range"][1])
            synthetic[col] = vals

    numeric_cols = [
        c for c in real_df.columns
        if pd.api.types.is_numeric_dtype(real_df[c])
    ]

    if len(numeric_cols) > 1:
        copula = GaussianMultivariate()
        copula.fit(real_df[numeric_cols])

        sampled_numeric = copula.sample(n_rows)

        for col in numeric_cols:
            synthetic[col] = sampled_numeric[col].values

            spec = schema["columns"][col]
            synthetic[col] = synthetic[col].clip(
                spec["range"][0],
                spec["range"][1],
            )

            if spec["type"] == "int":
                synthetic[col] = synthetic[col].round().astype(int)

    return synthetic, schema
