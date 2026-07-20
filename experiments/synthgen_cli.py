import os
import argparse
from datetime import datetime

import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

from csv_synthetic_generator import (
    compare_datasets,
    save_comparison_results,
    generate_synthetic_from_real,
    preprocess_real_dataset,
)

from csv_input.csv_synthetic_generator import (
                        save_comparison_plots,
                    )


def main():

    parser = argparse.ArgumentParser(
        description="Synthetic Health Data Generator"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV file"
    )

    parser.add_argument(
        "--method",
        choices=["ctgan", "copula"],
        default="copula"
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=None
    )

    parser.add_argument(
        "--target",
        default=None,
        help="Optional target variable"
    )

    parser.add_argument(
        "--run-name",
        default="run"
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

        schema = {
            "generator": "CTGAN",
            "rows_generated": rows_to_generate
        }

    # Schema + Copula
    else:

        synthetic_df, schema = generate_synthetic_from_real(
            real_df,
            n_rows=rows_to_generate
        )

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

    output_path = os.path.join(
        metadata_dir,
        f"{run_name}.csv"
    )

    synthetic_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nGenerated {len(synthetic_df)} rows"
    )

    print(
        f"Saved dataset to:\n{output_path}"
    )

    print("\nSchema")
    print(schema)

    print("\nRunning comparison...")

    results = compare_datasets(
        real_df,
        synthetic_df,
        target_col=args.target
    )

    save_comparison_results(
        results,
        output_dir=metadata_dir,
        run_name=run_name
    )

    print("Comparison complete.")
    print(f"Results saved to: {metadata_dir}")


if __name__ == "__main__":
    main()