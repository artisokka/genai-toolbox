import streamlit as st
#st.set_page_config(layout="wide")

import numpy as np
import torch
import json
import pickle
import os
import sys
import matplotlib.pyplot as plt
import re
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from io import StringIO
from app_utils import _build_index_to_name, _humanize_diag
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
try:
    from copulas.multivariate import GaussianMultivariate
except Exception:
    GaussianMultivariate = None

# --- Import CSV generator components ---
from csv_input.csv_synthetic_generator import (
    infer_schema_from_dataframe,
    generate_synthetic_from_real,
)

# --- Import RAG components ---
# Assuming these files are in a 'rag' subdirectory
from rag.main import create_rag_system, get_answer
from rag.pdf_to_text import convert_pdfs_to_text
from rag.txt_to_index import create_faiss_index

script_dir = os.path.dirname(os.path.abspath(__file__))
sssd_dir = os.path.join(script_dir, 'sssd')
if sssd_dir not in sys.path:
    sys.path.insert(0, sssd_dir)

# --- Import sssd components ---
try:
    from sssd.models.SSSD_ECG import SSSD_ECG
    from sssd.utils.util import calc_diffusion_hyperparams
    from sssd.inference import generate
    from sssd.visualize_ecg import visualize_ecg_npy
except ImportError as e:
    st.error(f"Error importing required modules from 'sssd': {e}")
    st.stop()



# --- Streamlit App UI ---
st.title("GenAI Toolbox: Synthetic Health Data Generator")

tab1, tab2 = st.tabs(["📊 Advanced Tabular Data Generation", "📈 Synthetic ECG Generation"])

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

    **➡️ Knowledge-Base Generation (RAG)**
    - Upload documents and build a knowledge base.
    - The model infers a dataset schema from the indexed literature and generates synthetic tabular data based on the available information.
    - Useful when structured patient data is not available.

    ### Data Synthesis
    - Select a generation mode.
    - Specify the number of rows to generate.
    - Review the inferred schema before generation.
    - Download the generated synthetic dataset for analysis or model development.
""")


    generation_mode = st.radio(
        "Generation mode",
        [
            "Literature / RAG",
            "Real Dataset (CSV)"
        ]
    )

    if generation_mode == "Literature / RAG":

        # --- Setup RAG Paths ---
        data_dir = os.path.join(script_dir, "rag", "Data")
        text_folder = os.path.join(script_dir, "rag", "DataTxt")
        index_path = os.path.join(script_dir, "rag", "DataIndex")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(text_folder, exist_ok=True)
        os.makedirs(index_path, exist_ok=True)


        st.subheader("Manage RAG Knowledge Base")
        with st.expander("Upload PDFs to build or update the knowledge base"):
            uploaded_files = st.file_uploader(
                "Upload PDF documents",
                type=['pdf'],
                accept_multiple_files=True
            )

            if st.button("Process PDFs and Rebuild Index"):
                if uploaded_files:
                    with st.spinner("Processing PDFs and rebuilding index..."):
                        try:
                            # Save each PDF
                            for file in uploaded_files:
                                file_path = os.path.join(data_dir, file.name)
                                with open(file_path, "wb") as f:
                                    f.write(file.getbuffer())

                            # Convert to text and build index
                            convert_pdfs_to_text(data_dir, text_folder)
                            create_faiss_index(text_folder, index_path)
                            st.success("Successfully processed PDFs and rebuilt the index!")
                            
                            # Clear any cached RAG system to force a reload
                            if 'rag_system' in st.session_state:
                                del st.session_state['rag_system']
                                
                        except Exception as e:
                            st.error(f"An error occurred while processing PDFs: {str(e)}")
                else:
                    st.warning("Please upload at least one PDF file.")

        # --- Initialize RAG System ---
        rag_chain = None
        index_path = os.path.join(script_dir, "rag", "DataIndex")
        if os.path.exists(os.path.join(index_path, "index.faiss")):
            try:
                rag_chain = create_rag_system(index_path)
                if rag_chain is None:
                    st.error("Failed to create RAG system. Check if the FAISS index is valid.")
                else:
                    st.success("RAG system initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing RAG system: {e}")
                st.error(f"Error type: {type(e).__name__}")
        else:
            st.warning("RAG system not ready. Please upload PDF documents in the main application to build the knowledge base first.")

        # Generate via LLM Schema
        st.subheader("Generate synthetic tabular data")
        st.markdown("""
        - Describe the dataset in natural language (e.g., 'diabetes biomarkers') or list the columns you want to generate. 
        - If left empty and a RAG index exists, the schema will be inferred from uploaded documents.
        """)
        schema_prompt = st.text_area(
            "Describe the dataset you want (optional)",
            "",
            height=80,
        )
        rows_schema = st.number_input("Rows to generate", min_value=10, max_value=50000, value=100)
        if st.button("Generate Tabular Data"):
            with st.spinner("Calling LLM for schema and generating data..."):
                try:
                    def _produce_schema_text(desc: str):
                        # Unified schema instruction; used for both RAG and direct LLM
                        schema_instr = (
                            "Produce ONLY a single, valid, minified JSON object representing a tabular data schema. "
                            "NO extra text, NO markdown fences. "
                            "Schema structure: "
                            '{"columns": {"column_name": {"type": "...", "details": {...}}}, "constraints": ["..."]}. '
                            "Valid types: 'int', 'float', 'category'. "
                            "Values can't be negative."
                            "Use european metrics such as mmol/L for blood sugar, cm for height, kg for weight, etc."
                            "For 'category': include 'values' list, optional 'probs'. "
                            "For numeric: include 'range' [min, max], optional 'dist': 'truncated_normal' with 'mean' and 'sd'. "
                            "Include sensible constraints (e.g., '\"age\" > \"medication_count\" * 5'). "
                            "Avoid any PII."
                            "Always include \"diagnosis\" as the last column with clinical ECG diagnoses, normal sinus rhythm being the most common."
                        )
                        # Build the request that will be passed either through RAG or directly
                        spec_line = (" User specification: " + desc.strip()) if (desc and desc.strip()) else ""
                        request_text = schema_instr + spec_line
                        # Prefer RAG when available to ground in domain documents; also include the user's spec
                        if rag_chain is not None:
                            return get_answer(request_text, rag_chain)
                        # Fallback to direct LLM when no RAG is available, still enforcing JSON-only output
                        if desc and desc.strip():
                            model_name = "llama3.2"
                            llm = ChatOpenAI(
                                model=model_name,
                                openai_api_key=os.environ.get("OPENAI_API_KEY"),
                                openai_api_base=os.environ.get("OPENAI_BASE_URL"),
                            )
                            sys_prefix = (
                                "You are an expert medical data scientist. "
                                "Follow the user's instructions exactly and output ONLY the JSON object with no extra text. "
                            )
                            return llm.invoke(sys_prefix + request_text).content
                        raise ValueError("No description provided and no RAG index available. Provide a description or build the RAG index.")

                    raw = _produce_schema_text(schema_prompt)

                    def _extract_first_json_block(text: str) -> str:
                        if not isinstance(text, str):
                            return ""
                        # Prefer fenced blocks first (```json ... ``` or ``` ... ```)
                        blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
                        for b in blocks:
                            candidate = b.strip()
                            if candidate:
                                return candidate
                        # Fallback: find first balanced {...} object
                        start = text.find('{')
                        if start == -1:
                            return text.strip()
                        depth = 0
                        for i in range(start, len(text)):
                            ch = text[i]
                            if ch == '{':
                                depth += 1
                            elif ch == '}':
                                depth -= 1
                                if depth == 0:
                                    return text[start:i+1].strip()
                        return text.strip()

                    def _try_load_json(possible: str):
                        # Try direct load
                        try:
                            return json.loads(possible)
                        except Exception:
                            pass
                        # Try removing trailing commas before closing braces/brackets
                        try:
                            cleaned = re.sub(r",\s*([}\]])", r"\1", possible)
                            return json.loads(cleaned)
                        except Exception:
                            return None

                    # Extract and parse JSON
                    json_text = _extract_first_json_block(raw)
                    schema = _try_load_json(json_text)
                    if schema is None:
                        # As a last resort, try the entire raw
                        schema = _try_load_json(raw.strip())
                    # Some models double-encode JSON, resulting in a JSON string containing JSON text
                    if isinstance(schema, str):
                        schema_str_attempt = _try_load_json(schema)
                        if schema_str_attempt is not None:
                            schema = schema_str_attempt
                    if schema is None:
                        preview = (raw or "").strip().replace('\n', ' ')[:300]
                        raise ValueError(f"LLM did not return valid JSON. Preview: {preview}")

                    # Validate required shape; auto-convert common variants
                    if isinstance(schema, dict):
                        cols = schema.get("columns")
                        if isinstance(cols, list):
                            converted = {}
                            for idx, item in enumerate(cols):
                                if isinstance(item, dict):
                                    name = item.get("name") or item.get("column") or f"col_{idx}"
                                    converted[name] = {k: v for k, v in item.items() if k not in {"name", "column"}}
                                elif isinstance(item, str):
                                    converted[item] = {"type": "category", "values": []}
                            schema["columns"] = converted
                    if not (isinstance(schema, dict) and isinstance(schema.get("columns"), dict)):
                        preview = json.dumps(schema)[:200] if not isinstance(schema, str) else schema[:200]
                        raise ValueError(f"Schema must be an object with a 'columns' object. Got: {preview}")

                    # Normalize possible nested 'details' under each column
                    if isinstance(schema, dict) and isinstance(schema.get("columns"), dict):
                        normalized_cols = {}
                        for col_name, spec in schema["columns"].items():
                            if isinstance(spec, dict) and isinstance(spec.get("details"), dict):
                                merged = {k: v for k, v in spec.items() if k != "details"}
                                merged.update(spec["details"])
                                normalized_cols[col_name] = merged
                            else:
                                normalized_cols[col_name] = spec
                        schema["columns"] = normalized_cols

                    # Print the schema to console for debugging
                    print("Generated JSON Schema:")
                    print(json.dumps(schema, indent=2))

                    def _sample_base(_schema, n):
                        df_loc = pd.DataFrame(index=range(n))
                        for col, spec in _schema.get("columns", {}).items():
                            # Support schemas where parameters are nested under 'details'
                            if isinstance(spec, dict) and isinstance(spec.get("details"), dict):
                                merged_spec = {k: v for k, v in spec.items() if k != "details"}
                                merged_spec.update(spec["details"])
                                spec = merged_spec
                            ctype = str(spec.get("type","category")).lower()
                            if ctype == "category":
                                vals = spec.get("values", [])
                                probs = spec.get("probs", None)
                                if not vals:
                                    continue
                                df_loc[col] = np.random.choice(vals, size=n, p=probs)
                            elif ctype in ("int","float"):
                                lo, hi = spec.get("range", [0, 1])
                                if str(spec.get("dist","uniform")).lower() == "truncated_normal":
                                    mean, sd = float(spec.get("mean", (lo+hi)/2)), float(spec.get("sd", (hi-lo)/6))
                                    vals = np.clip(np.random.normal(mean, sd, size=n), lo, hi)
                                else:
                                    vals = np.random.uniform(lo, hi, size=n)
                                df_loc[col] = vals if ctype == "float" else np.rint(vals).astype(int)
                        return df_loc

                    def _apply_copula(df_loc, _schema):
                        if GaussianMultivariate is None:
                            return df_loc
                        num_cols = []
                        for c, s in _schema.get("columns", {}).items():
                            if isinstance(s, dict) and isinstance(s.get("details"), dict):
                                s = {**{k: v for k, v in s.items() if k != "details"}, **s["details"]}
                            if str(s.get("type","")) in ("int","float"):
                                num_cols.append(c)
                        if not num_cols:
                            return df_loc
                        model = GaussianMultivariate()
                        try:
                            model.fit(df_loc[num_cols])
                            df_loc[num_cols] = model.sample(len(df_loc)).to_numpy()
                            for col, spec in _schema.get("columns", {}).items():
                                if isinstance(spec, dict) and isinstance(spec.get("details"), dict):
                                    spec = {**{k: v for k, v in spec.items() if k != "details"}, **spec["details"]}
                                if str(spec.get("type","")) in ("int","float"):
                                    lo, hi = spec.get("range", [None, None])
                                    if lo is not None and hi is not None:
                                        df_loc[col] = np.clip(df_loc[col], lo, hi)
                                    if str(spec.get("type")) == "int":
                                        df_loc[col] = df_loc[col].round().astype(int)
                        except Exception:
                            pass
                        return df_loc

                    def _enforce_constraints(df_loc, _schema):
                        for cons in _schema.get("constraints", []):
                            # Allow constraints to be dicts or strings; skip unknowns
                            if isinstance(cons, str):
                                # Optional: support simple expressions like "age >= 18"
                                try:
                                    expr = cons.strip()
                                    if expr:
                                        ok_mask = pd.eval(expr, engine='python', local_dict=df_loc.to_dict(orient='series'))
                                        if isinstance(ok_mask, pd.Series):
                                            df_loc = df_loc.loc[ok_mask]
                                except Exception:
                                    pass
                                continue
                            if not isinstance(cons, dict):
                                continue
                            ctype = str(cons.get("type",""))
                            if ctype == "inequality":
                                low, high = cons.get("low"), cons.get("high")
                                if low in df_loc.columns and high in df_loc.columns:
                                    df_loc = df_loc.loc[df_loc[high] >= df_loc[low]]
                            elif ctype == "range":
                                col = cons.get("column")
                                if col in df_loc.columns:
                                    mn, mx = cons.get("min"), cons.get("max")
                                    if mn is not None:
                                        df_loc = df_loc.loc[df_loc[col] >= mn]
                                    if mx is not None:
                                        df_loc = df_loc.loc[df_loc[col] <= mx]
                        return df_loc

                    # Retry loop to meet requested row count after constraint filtering
                    target_rows = int(rows_schema)
                    max_attempts = 6
                    batch_size = max(target_rows, 100)
                    collected = []
                    attempts = 0
                    while sum(len(df) for df in collected) < target_rows and attempts < max_attempts:
                        attempts += 1
                        df_try = _sample_base(schema, batch_size)
                        df_try = _apply_copula(df_try, schema)
                        df_try = _enforce_constraints(df_try, schema).reset_index(drop=True)
                        if not df_try.empty:
                            collected.append(df_try)
                        # adaptively increase sample size if we get too few after constraints
                        if sum(len(df) for df in collected) < target_rows:
                            batch_size = min(int(batch_size * 1.5), target_rows * 5)

                    if collected:
                        df_gen = pd.concat(collected, ignore_index=True)
                    else:
                        df_gen = pd.DataFrame()

                    if len(df_gen) == 0:
                        raise ValueError("No rows could be generated after applying constraints.")

                    # Truncate to exact requested size
                    if len(df_gen) > target_rows:
                        df_gen = df_gen.iloc[:target_rows].reset_index(drop=True)

                    st.session_state.final_synthetic_df = df_gen
                    st.success(f"Generated {len(df_gen)} rows from LLM-defined schema.")
                except Exception as e:
                    st.error(f"Failed to generate via LLM schema: {e}")

        # --- Display Final Results ---
        if 'final_synthetic_df' in st.session_state:
            st.subheader("Generated Data")
            final_df = st.session_state.final_synthetic_df
            st.dataframe(final_df)
            
            csv_final = final_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download as CSV",
                data=csv_final,
                file_name='final_synthetic_patient_data.csv',
                mime='text/csv',
            )

            st.markdown("---")
            st.subheader("Generate ECGs for tabular dataset")
            st.markdown("Map rows to ECG label vectors and synthesize matching signals.")

            # Row selection
            selectable_indices = list(final_df.index)
            default_sel = selectable_indices[: min(5, len(selectable_indices))]
            selected_rows = st.multiselect("Select rows to generate ECGs for", selectable_indices, default=default_sel)
            generate_all_rows = st.checkbox("Generate for all rows", value=False)
            num_ecg_per_row = st.number_input("ECGs per selected row", min_value=1, max_value=10, value=1)
            
            # Load label name mapping if available
            raw_mapping = None
            for p in [
                os.path.join("ptb_xl", "processed_ptb_xl_fs100", "lbl_itos.pkl"),
                os.path.join("sssd", "processed_ptb_xl_fs100", "lbl_itos.pkl"),
            ]:
                if os.path.exists(p):
                    try:
                        with open(p, 'rb') as f:
                            raw_mapping = pickle.load(f)
                    except Exception:
                        raw_mapping = None
                    break

            idx_to_name = _build_index_to_name(raw_mapping) if raw_mapping is not None else None
            name_to_idx = { (idx_to_name[i] or f"lbl_{i}"): i for i in range(len(idx_to_name)) } if idx_to_name else {}

            # Choose a categorical column to map (default to 'diagnosis' if present)
            cat_cols = [c for c in final_df.columns if not pd.api.types.is_numeric_dtype(final_df[c])]
            default_diag_col = next((c for c in cat_cols if c.lower() == 'diagnosis'), (cat_cols[0] if cat_cols else None))
            diag_col = st.selectbox("Categorical column to map to ECG labels", options=[""] + cat_cols, index=(cat_cols.index(default_diag_col)+1) if default_diag_col else 0)

            # Build an automatic mapping from unique values to label codes (no manual input required)
            suggested = {}
            if diag_col:
                unique_vals = sorted([str(v) for v in final_df[diag_col].dropna().unique()])[:30]
                for v in unique_vals:
                    key = v.strip().upper().replace(" ", "_")
                    # Heuristic matches for common codes
                    for code in ["AMI","AFIB","LBBB","RBBB","NORM","NST_","STACH","SBRAD"]:
                        if code in key:
                            suggested[v] = code
                            break
                    # If the cleaned value exactly matches a known label name, use it
                    if v not in suggested and key in (idx_to_name or []):
                        suggested[v] = key

            # Load ECG generation config
            try:
                with open('sssd/config/config_SSSD_ECG.json') as f:
                    ecg_cfg = json.load(f)
            except Exception as e:
                ecg_cfg = None
                st.warning(f"Could not load ECG config: {e}")

            effective_rows = selectable_indices if generate_all_rows else selected_rows
            if st.button("Generate Conditioned ECGs", disabled=(ecg_cfg is None or (not generate_all_rows and not selected_rows))):
                try:
                    model_config = ecg_cfg.get('wavenet_config')
                    diffusion_config = ecg_cfg.get('diffusion_config')
                    gen_config = ecg_cfg.get('gen_config')
                    if not all([model_config, diffusion_config, gen_config]):
                        st.error("ECG config missing required sections.")
                        st.stop()

                    diffusion_hyperparams_local = calc_diffusion_hyperparams(**diffusion_config)

                    num_classes = int(model_config.get("label_embed_classes"))
                    if not num_classes:
                        st.error("Model config lacks 'label_embed_classes'.")
                        st.stop()

                    # Auto mapping derived from suggestions only
                    value_to_label = {str(k): [str(v)] for k, v in suggested.items()}

                    # Build label matrix
                    def labels_for_row(row: pd.Series) -> np.ndarray:
                        vec = np.zeros((num_classes,), dtype=np.float32)
                        # categorical mapping only (automatic suggestions)
                        if diag_col:
                            val = str(row.get(diag_col, ""))
                            for code in value_to_label.get(val, []):
                                if code in name_to_idx:
                                    vec[name_to_idx[code]] = positive_score
                        return vec

                    per_row = []
                    for ridx in effective_rows:
                        row = final_df.loc[ridx]
                        vec = labels_for_row(row)
                        for _ in range(int(num_ecg_per_row)):
                            per_row.append(vec.copy())
                    label_matrix = np.stack(per_row, axis=0)

                    # Call generator with explicit label_matrix
                    generate(
                        model_config=model_config,
                        diffusion_config=diffusion_config,
                        diffusion_hyperparams=diffusion_hyperparams_local,
                        output_directory=gen_config.get("output_directory", "generated_ecg"),
                        num_samples=label_matrix.shape[0],
                        ckpt_path=gen_config.get("ckpt_path", "100000.pkl"),
                        data_path=ecg_cfg.get("trainset_config", {}).get("data_path", "data"),
                        ckpt_iter=gen_config.get("ckpt_iter", "max"),
                        label_matrix=label_matrix,
                    )
                    st.success(f"Generated {label_matrix.shape[0]} conditioned ECG(s).")
                except Exception as e:
                    st.error(f"Failed to generate conditioned ECGs: {e}")
    
    elif generation_mode == "Real Dataset (CSV)":

        uploaded_csv = st.file_uploader(
            "Upload patient dataset",
            type=["csv"]
        )

        if uploaded_csv is not None:
            real_df = pd.read_csv(uploaded_csv)

            st.subheader("Dataset Preview")
            st.dataframe(real_df.head())

            rows_to_generate = st.number_input(
                "Rows to generate",
                min_value=10,
                value=len(real_df)
            )

            if st.button("Generate Synthetic Dataset"):

                synthetic_df, schema = generate_synthetic_from_real(
                    real_df,
                    n_rows=rows_to_generate
                )

                st.session_state.final_synthetic_df = synthetic_df

                st.success(
                    f"Generated {len(synthetic_df)} synthetic rows."
                )

                with st.expander("Inferred Schema"):
                    st.json(schema)


# --- ECG Generation Tab ---
with tab2:
    st.header("Synthetic ECG Generation")
    st.markdown("Generate synthetic 12-lead ECG signals using a pre-trained SSSD-ECG model.")

    # --- Load Configurations ---
    try:
        with open('sssd/config/config_SSSD_ECG.json') as f:
            config = json.load(f)
    except FileNotFoundError:
        st.error("Fatal Error: `config/config_SSSD_ECG.json` not found.")
        st.stop()

    model_config = config.get('wavenet_config')
    diffusion_config = config.get('diffusion_config')
    gen_config = config.get('gen_config')

    if not all([model_config, diffusion_config, gen_config]):
        st.error("Fatal Error: The JSON config file is missing required sections (wavenet_config, diffusion_config, or gen_config).")
        st.stop()
        
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)

    # --- UI for Generation Parameters ---
    st.subheader("Generation Settings")
    ckpt_iter = "max"
    num_samples = st.number_input("Number of Samples to Generate", min_value=1, max_value=1000, value=1)
    
    # --- Execute Generation ---
    if st.button("Generate Synthetic ECG"):
        st.info("Starting ECG generation... this may take a moment.")
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Initializing model and loading checkpoint...")
            
            # Call the generation function from the inference script
            generate(
                model_config=model_config,
                diffusion_config=diffusion_config,
                diffusion_hyperparams=diffusion_hyperparams,
                output_directory=gen_config.get("output_directory", "generated_ecg"),
                num_samples=num_samples,
                ckpt_path=gen_config.get("ckpt_path", "100000.pkl"),
                data_path=config.get("trainset_config", {}).get("data_path", "data"),
                ckpt_iter=ckpt_iter
            )
            
            progress_bar.progress(100)
            status_text.text("Generation complete!")
            st.success("Successfully generated synthetic ECG data.")
            
        except Exception as e:
            st.error(f"An error occurred during generation: {e}")
            st.exception(e)

    # --- Visualization Section ---
    st.header("Visualize Generated Samples")
    out_dir_base = gen_config.get("output_directory", "generated_ecg")
    target_dir = out_dir_base

    if not os.path.isdir(target_dir):
        st.info(f"No generated samples found yet in `{target_dir}`. Generate first or adjust the path.")
        st.stop()

    # Find run folders (newest first). If none, fall back to legacy per-file listing
    run_folders = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
    run_folders = sorted(run_folders, key=lambda d: os.path.getmtime(os.path.join(target_dir, d)), reverse=True)

    if run_folders:
        col_sel, col_all = st.columns([3, 1])
        with col_sel:
            run_choice = st.selectbox("Select run folder", run_folders, index=0)
        with col_all:
            visualize_all = st.checkbox("Visualize all", value=False)

        run_dir = os.path.join(target_dir, run_choice)
        sample_files = sorted([f for f in os.listdir(run_dir) if f.endswith('_samples.npy')])
        if not sample_files:
            st.info("No sample files found in the selected run folder.")
            st.stop()

        # Render controls
        fs = st.number_input("Sampling rate (Hz)", min_value=50, max_value=1000, value=100)
        gain = 2.0
        # gain = st.slider("Amplitude scaling", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

        # Load label name mapping if available
        raw_mapping = None
        for p in [
            os.path.join("ptb_xl", "processed_ptb_xl_fs100", "lbl_itos.pkl"),
            os.path.join("sssd", "processed_ptb_xl_fs100", "lbl_itos.pkl"),
        ]:
            if os.path.exists(p):
                try:
                    with open(p, 'rb') as f:
                        raw_mapping = pickle.load(f)
                except Exception:
                    raw_mapping = None
                break

        if not visualize_all:
            # Per-file selection
            file_choice = st.selectbox("Select batch file", sample_files, index=0, disabled=False)
            file_index = int(file_choice.split('_')[0]) if file_choice.split('_')[0].isdigit() else 0
            try:
                data = np.load(os.path.join(run_dir, file_choice))
                batch_size = data.shape[0]
            except Exception as e:
                st.error(f"Failed to read {file_choice}: {e}")
                st.stop()
            sample_idx = st.number_input("Sample index in batch", min_value=0, max_value=max(0, batch_size-1), value=0, disabled=False)

            # Auto-render ECG plot and labels by default
            # Prepare label info
            label_info = None
            labels_path = os.path.join(run_dir, f"{file_index}_labels.npy")
            if os.path.exists(labels_path):
                try:
                    labels_np = np.load(labels_path)
                    row = labels_np[sample_idx]
                    idx_to_name = _build_index_to_name(raw_mapping, expected_len=len(row))
                    if idx_to_name:
                        label_info = {}
                        for i in range(len(row)):
                            name = idx_to_name[i] if (i < len(idx_to_name)) else None
                            disp = _humanize_diag(name) if name else f"lbl_{i}"
                            label_info[disp] = float(row[i])
                    else:
                        label_info = {f"lbl_{i}": float(row[i]) for i in range(len(row))}
                except Exception as e:
                    st.warning(f"Could not read labels: {e}")

            fig = visualize_ecg_npy(
                folder=run_dir,
                index=file_index,
                sample_idx=sample_idx,
                fs=fs,
                show=False,
                gain=gain,
                label_scores=label_info,
                top_k=10,
            )
            st.pyplot(fig)
        
        else:
            # Visualize all samples in the chosen run; disable per-sample controls
            st.info(f"Rendering all samples in `{run_choice}`")
            if st.button("Render All ECG Plots"):
                for f in sample_files:
                    file_index = int(f.split('_')[0]) if f.split('_')[0].isdigit() else 0
                    labels_path = os.path.join(run_dir, f"{file_index}_labels.npy")
                    # Load labels for this file if present
                    labels_np = None
                    if os.path.exists(labels_path):
                        try:
                            labels_np = np.load(labels_path)
                        except Exception:
                            labels_np = None
                    try:
                        data = np.load(os.path.join(run_dir, f))
                    except Exception as e:
                        st.warning(f"Skipping {f}: {e}")
                        continue
                    batch_size = data.shape[0]
                    for sidx in range(batch_size):
                        label_info = None
                        if labels_np is not None:
                            row = labels_np[sidx]
                            idx_to_name = _build_index_to_name(raw_mapping, expected_len=len(row))
                            if idx_to_name:
                                label_info = {}
                                for i in range(len(row)):
                                    name = idx_to_name[i] if (i < len(idx_to_name)) else None
                                    disp = _humanize_diag(name) if name else f"lbl_{i}"
                                    label_info[disp] = float(row[i])
                            else:
                                label_info = {f"lbl_{i}": float(row[i]) for i in range(len(row))}
                        fig = visualize_ecg_npy(
                            folder=run_dir,
                            index=file_index,
                            sample_idx=sidx,
                            fs=fs,
                            show=False,
                            gain=gain,
                            label_scores=label_info,
                            top_k=10,
                        )
                        st.pyplot(fig)
                        if label_info:
                            top_items = sorted(label_info.items(), key=lambda kv: kv[1], reverse=True)[:10]
                            st.markdown("**Top labels**")
                            for name, score in top_items:
                                st.write(f"{name}: {score:.3f}")
    else:
        # Legacy: no run folders; fall back to old per-file listing at target_dir
        st.info("No run folders detected; showing files directly in output directory.")
        all_files = sorted([f for f in os.listdir(target_dir) if f.endswith('_samples.npy')])
        if not all_files:
            st.info("No sample files found.")
            st.stop()
        file_choice = st.selectbox("Select batch file", all_files)
        file_index = int(file_choice.split('_')[0])
        try:
            data = np.load(os.path.join(target_dir, file_choice))
            batch_size = data.shape[0]
        except Exception as e:
            st.error(f"Failed to read {file_choice}: {e}")
            st.stop()
        sample_idx = st.number_input("Sample index in batch", min_value=0, max_value=max(0, batch_size-1), value=0)
        fs = st.number_input("Sampling rate (Hz)", min_value=50, max_value=1000, value=100)
        gain = 2.0
        # gain = st.slider("Amplitude scaling (higher = smaller spikes)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

        # Optional label mapping and labels
        raw_mapping = None
        for p in [
            os.path.join("ptb_xl", "processed_ptb_xl_fs100", "lbl_itos.pkl"),
            os.path.join("sssd", "processed_ptb_xl_fs100", "lbl_itos.pkl"),
        ]:
            if os.path.exists(p):
                try:
                    with open(p, 'rb') as f:
                        raw_mapping = pickle.load(f)
                except Exception:
                    raw_mapping = None
                break
        label_info = None
        labels_path = os.path.join(target_dir, f"{file_index}_labels.npy")
        if os.path.exists(labels_path):
            try:
                labels_np = np.load(labels_path)
                row = labels_np[sample_idx]
                idx_to_name = _build_index_to_name(raw_mapping, expected_len=len(row))
                if idx_to_name:
                    label_info = {}
                    for i in range(len(row)):
                        name = idx_to_name[i] if (i < len(idx_to_name)) else None
                        disp = _humanize_diag(name) if name else f"lbl_{i}"
                        label_info[disp] = float(row[i])
                else:
                    label_info = {f"lbl_{i}": float(row[i]) for i in range(len(row))}
            except Exception as e:
                st.warning(f"Could not read labels: {e}")

        # Auto-render
        fig = visualize_ecg_npy(folder=target_dir, index=file_index, sample_idx=sample_idx, fs=fs, show=False, gain=gain, label_scores=label_info)
        st.pyplot(fig)
        if label_info:
            top_items = sorted(label_info.items(), key=lambda kv: kv[1], reverse=True)[:10]
            st.markdown("**Top labels**")
            for name, score in top_items:
                st.write(f"{name}: {score:.3f}")
