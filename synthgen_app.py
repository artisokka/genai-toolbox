import streamlit as st
#st.set_page_config(layout="wide")

import os
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from datetime import datetime

# --- Import CSV generator components ---
from csv_input.csv_synthetic_generator import (
    compare_datasets,
    save_comparison_results,
    generate_synthetic_from_real,
    preprocess_real_dataset,
)

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Streamlit App UI ---
st.title("GenAI Toolbox: Synthetic Health Data Generator: T3 deployment version")

tab1, tab2 = st.tabs(["📊 Advanced Tabular Data Generation", "Disabled"])

# --- Advanced Tabular Data Generation Tab ---
with tab1:
    st.header("Generate Synthetic Tabular Data")
    
    st.markdown("""
        ### Generation Modes

        **➡️ Patient Dataset Generation**
        - Upload a real tabular dataset (CSV).
        - The toolbox automatically infers:
        - variables and data types
        - value ranges and categories
        - statistical relationships between variables
        - Generate a synthetic dataset that preserves the structure of the original data while containing no real patient records.

        **➡️ Knowledge-Based Generation (RAG)**
        - Upload documents and build a knowledge base.
        - The model infers a dataset schema from the indexed literature and generates synthetic tabular data based on the available information.
        - Useful when structured patient data is not available.

        ### Data Synthesis
        - Select a generation mode.
        - Specify the number of rows to generate.
        - Review the inferred schema before generation.
        - Download the generated synthetic dataset for analysis or model development.
    """)

    st.info(
        "For synthetic-data validation studies, Patient Dataset Generation "
        "is the recommended mode."
    )


    generation_mode = st.radio(
        "Generation mode",
        [
            "Patient Dataset Generation (CSV)",
            "Knowledge-based generation (RAG)"
        ]
    )

    if generation_mode == "Knowledge-based generation (RAG)":

        st.markdown("RAG mode is disabled in T3-deployment")
    
    elif generation_mode == "Patient Dataset Generation (CSV)":

        st.markdown("""
        ### Validation Workflow
        1. Upload a real cohort (CSV).
        2. Generate a synthetic version of the cohort.
        3. Compare descriptive statistics and variable relationships.
        4. Use the synthetic data for downstream machine learning experiments.
        """)

        uploaded_csv = st.file_uploader(
            "Upload patient dataset",
            type=["csv"]
        )

        synth_method = st.radio(
            "Synthetic data generation method",
            [
                "CTGAN",
                "Schema + Copula"
            ]
        )

        if uploaded_csv is not None:
            real_df = pd.read_csv(uploaded_csv)
            real_df, report = preprocess_real_dataset(real_df)

            st.subheader("Dataset Quality Report")
            st.json(report)

            st.subheader("Dataset Preview")
            st.dataframe(real_df.head())

            rows_to_generate = st.number_input(
                "Rows to generate",
                min_value=10,
                value=len(real_df)
            )

            target_col = st.selectbox(
                "Optional target variable for subgroup analysis",
                options=["None"] + real_df.columns.tolist()
            )

            run_name = st.text_input(
                "Optional run name",
                value=None
            )
            
            if target_col == "None":
                target_col = None


            if st.button("Generate Synthetic Dataset"):
                try:

                    # Method 1: CTGAN

                    if synth_method == "CTGAN":

                        with st.spinner (
                            "Training CTGAN and generating synthetic data..."
                        ):
                            metadata = SingleTableMetadata()
                            metadata.detect_from_dataframe(
                                data = real_df
                            )
                            synthesizer = CTGANSynthesizer(
                                metadata = metadata,
                                epochs = 300
                            )

                            synthesizer.fit(real_df)

                            synthetic_df = synthesizer.sample(
                                num_rows = rows_to_generate
                            )

                            schema = {
                                "generator": "CTGAN",
                                "rows_generated": rows_to_generate
                            }

                    # Method 2: Schema + Copula
                    elif synth_method == "Schema + Copula":

                        synthetic_df, schema = generate_synthetic_from_real(
                            real_df,
                            n_rows=rows_to_generate
                        )
                    
                    # Saving output

                    output_dir = "generated_tabular"

                    os.makedirs(output_dir, exist_ok=True)

                    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M")
                    run_name = f"{timestamp}_{run_name}"

                    metadata_dir = os.path.join(
                        output_dir,
                        f"{run_name}"
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

                    st.session_state.final_synthetic_df = synthetic_df

                    st.success(
                        f"Generated {len(synthetic_df)} synthetic rows."
                    )

                    st.info(
                        f"Saved to: {output_path}"
                    )

                    with st.expander("Inferred Schema"):
                        st.json(schema)

                    st.subheader("Generated data")
                    st.write(synthetic_df.head())

                    st.subheader("Comparing real dataset to synthetic")

                    results = compare_datasets(
                        real_df,
                        synthetic_df,
                        target_col=target_col
                    )

                    save_comparison_results(
                        results,
                        output_dir=metadata_dir,
                        run_name=run_name,
                    )

                    from csv_input.csv_synthetic_generator import (
                        save_comparison_plots,
                    )

                    plot_dir = save_comparison_plots(
                        real_df,
                        synthetic_df,
                        metadata_dir,
                    )

                    st.success(f"Plots saved to: {plot_dir}")

                except Exception as e:
                    st.error(
                        f"Generation failed: {str(e)}"
                    )
with tab2:
    st.markdown("Disabled in T3-deployment")
