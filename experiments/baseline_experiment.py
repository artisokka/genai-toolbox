import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)

import matplotlib.pyplot as plt


# ==================================================
# CONFIG
# ==================================================

DATA_PATH = "patients.csv"

COHORT_COLUMN = "Cohort_number"

#TRAIN_TARGET = "StableUnstable0stable1unstablei.e.growthorrupture"
#TEST_TARGET = "Rupture1yes0no"

TRAIN_TARGET = ""
TEST_TARGET = ""

RANDOM_STATE = 42


# ==================================================
# HELPER
# ==================================================

def clean_column_names(df):
    df = df.copy()

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    return df


def prepare_ml_data(df, target_column):

    df = df.copy()

    id_columns = [
        "Patientidentifyingnumber",
        "Aneurysmidentifyingnumber",
    ]

    existing_ids = [
        c for c in id_columns
        if c in df.columns
    ]

    df = df.drop(
        columns=existing_ids,
        errors="ignore"
    )

    columns_to_drop = (
        id_columns +
        [TRAIN_TARGET, TEST_TARGET]
    )

    y = df[target_column]

    X = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = [
        c
        for c in X.columns
        if c not in numeric_features
    ]

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            ),
        ]
    )

    return X, y, preprocessor


# ==================================================
# LOAD DATA
# ==================================================


if DATA_PATH.lower().endswith(".csv"):
    df = pd.read_csv(DATA_PATH)

elif DATA_PATH.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(DATA_PATH)

else:
    raise ValueError(
        f"Unsupported file type: {DATA_PATH}"
    )

print(df.columns.tolist())

df = clean_column_names(df)

print("Dataset shape:", df.shape)


# ==================================================
# COHORTS
# ==================================================

cohort1 = df[
    df[COHORT_COLUMN] == 1
].copy()

cohort2 = df[
    df[COHORT_COLUMN] == 2
].copy()

print("\nCohort 1")
print(cohort1.shape)

print("\nCohort 2")
print(cohort2.shape)

print("\nStable/Unstable distribution")
print(cohort1[TRAIN_TARGET].value_counts(dropna=False))

print("\nRupture distribution")
print(cohort2[TEST_TARGET].value_counts(dropna=False))


# ==================================================
# TRAIN DATA
# ==================================================

X_train, y_train, preprocessor = prepare_ml_data(
    cohort1,
    TRAIN_TARGET
)

print(X_train.columns.tolist())

X_test = cohort2.drop(
    columns=[
        TRAIN_TARGET,
        TEST_TARGET,
        "Patientidentifyingnumber",
        "Aneurysmidentifyingnumber",
    ],
    errors="ignore"
)

y_test = pd.to_numeric(
    cohort2[TEST_TARGET],
    errors="coerce"
).astype(int)


# ==================================================
# MODEL
# ==================================================

rf = RandomForestClassifier(
    n_estimators=500,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1,
)

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            rf
        ),
    ]
)


print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)


# ==================================================
# PREDICT
# ==================================================

probs = model.predict_proba(
    X_test
)[:, 1]

auc = roc_auc_score(
    y_test,
    probs
)

print("\nROC AUC")
print(f"{auc:.4f}")


# ==================================================
# ROC CURVE
# ==================================================

fpr, tpr, _ = roc_curve(
    y_test,
    probs
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(
    "Cohort 1 RF -> Cohort 2 Rupture Prediction"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "cohort1_to_cohort2_roc.png",
    dpi=300
)

plt.close()

print(
    "\nROC figure saved:"
    " cohort1_to_cohort2_roc.png"
)


# ==================================================
# FEATURE IMPORTANCE
# ==================================================

preprocessor = model.named_steps["preprocessor"]
numeric_features = preprocessor.transformers_[0][2]
cat_transformer = preprocessor.transformers_[1][1]

categorical_features = (
    cat_transformer.named_steps["onehot"]
    .get_feature_names_out(
        preprocessor.transformers_[1][2]
    )
)

feature_names = np.concatenate([
    numeric_features,
    categorical_features
])

importances = (
    model
    .named_steps["classifier"]
    .feature_importances_
)

# Sanity check

print("\nFeatures: ", len(feature_names))
print("\nImportances: ", len(importances))

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances,
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print("\nTop 20 Features")

print(
    importance_df.head(20)
)

#Save results

importance_df.to_csv(
    "feature_importance.csv",
    index=False
)

print(
    "\nFeature importance saved:"
    " feature_importance.csv"
)