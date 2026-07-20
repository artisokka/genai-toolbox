import os
import argparse
from datetime import datetime

import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

from csv_synthetic_generator import (
    compare_datasets,
    save_comparison_results,
    save_comparison_plots,
    generate_synthetic_from_real,
    preprocess_real_dataset,
)

def main():

    parser = argparse.ArgumentParser(
        description="Generate synthetic tabular datasets from a real CSV "
                    "using CTGAN, Schema+Copula, or both methods"
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to the input CSV dataset"
        )
    )

    parser.add_argument(
        "--method",
        choices=["ctgan", "copula", "both"],
        default="both",
        help=(
            "Synthetic data generation method. "
            "'ctgan' uses a CTGAN neural network, "
            "'copula' uses schema inference and copula modelling, "
            "'both' generates datasets using both methods for comparison. "
            "(default: %(default)s)"
        )
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help=(
              "Number of synthetic rows to generate. "
              "If omitted, the number of rows in the input dataset is used."
        )
    )

    parser.add_argument(
        "--target",
        default=None,
        help=(
            "Optional target column for subgroup/stratified analysis "
            "during dataset comparison."
        )
    )

    parser.add_argument(
        "--run-name",
        default="run",
        help=(
            "Name used for the output directory and generated files. "
            "(default: %default)s)"
        )
    )

    args = parser.parse_args()

    print("Loading dataset...")

    real_df = pd.read_csv(args.input)

    real_df, report = preprocess_real_dataset(real_df)

    print("\nDataset Quality Report")
    print(report)

    rows_to_generate = (
        args.rows
        if args.rows is not None
        else len(real_df)
    )

    print(
        f"\nGenerating {rows_to_generate} rows using "
        f"{args.method.upper()}"
    )

    generated_datasets = {}
    schemas = {}

    # CTGAN

    if args.method == "ctgan":

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(real_df)

        synthesizer = CTGANSynthesizer(
            metadata=metadata,
            epochs=300
        )

        synthesizer.fit(real_df)

        synthetic_df = synthesizer.sample(
            num_rows=rows_to_generate
        )

        generated_datasets["CTGAN"] = synthetic_df

        schemas = {
            "generator": "CTGAN",
            "rows_generated": rows_to_generate
        }

    # Schema + Copula
    elif args.method == "copula":

        synthetic_df, schema = generate_synthetic_from_real(
            real_df,
            n_rows=rows_to_generate
        )

        generated_datasets["Copula"] = synthetic_df
        schemas["Copula"] = schema

    # Compare methods

    elif args.method == "both":
        # CTGAN
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(real_df)

        synthesizer = CTGANSynthesizer(
            metadata=metadata,
            epochs=300
        )

        synthesizer.fit(real_df)

        ctgan_df = synthesizer.sample(
            num_rows=rows_to_generate
        )

        generated_datasets["CTGAN"] = ctgan_df

        schemas["CTGAN"] = {
            "generator": "CTGAN",
            "rows_generated": rows_to_generate
        }

        # Copula
        copula_df, copula_schema = (
            generate_synthetic_from_real(
                real_df,
                n_rows=rows_to_generate
            )
        )

        generated_datasets["Copula"] = copula_df
        schemas["Copula"] = copula_schema

    output_dir = "generated_tabular"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y-%m-%dT%H-%M"
    )

    run_name = f"{timestamp}_{args.run_name}"

    metadata_dir = os.path.join(
        output_dir,
        run_name
    )

    os.makedirs(metadata_dir, exist_ok=True)

    # Save datasets as csv

    for dataset_name, df in generated_datasets.items():

        output_path = os.path.join(
            metadata_dir,
            f"{run_name}_{dataset_name}.csv"
        )

        df.to_csv(
            output_path,
            index=False
        )

        print(
            f"{dataset_name} with {len(df)} rows saved to:\n{output_path}"
        )

    print("\nSchema")
    print(schemas)

    print("\nRunning comparison...")

    for dataset_name, df in generated_datasets.items():

        results = compare_datasets(
            real_df,
            df,
            target_col=args.target
        )

        save_comparison_results(
            results,
            output_dir=metadata_dir,
            run_name=f"{run_name}_{dataset_name}"
        )

    plot_dir = save_comparison_plots(
        real_df=real_df,
        output_dir=metadata_dir,
        synthetic_datasets=generated_datasets
    )

    print("Comparison complete.")
    print(f"Results saved to: {metadata_dir}")
    print(f"Plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()