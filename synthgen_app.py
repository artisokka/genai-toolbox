import streamlit as st
st.set_page_config(layout="wide")

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
    st.header("Generate High-Quality Tabular Data")
    st.markdown("""
    This tool uses a two-stage process for best results:
    1.  **Blueprint Generation (RAG):** An LLM creates a small, context-aware seed dataset based on your documents.
    2.  **Data Synthesis (SDV):** A statistical model learns from the blueprint to generate a large, high-fidelity dataset.
    """)

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
    st.caption("Describe the dataset in natural language (e.g., 'diabetes biomarkers'). If left empty and a RAG index exists, the schema will be inferred from uploaded documents.")
    schema_prompt = st.text_area(
        "Describe the dataset you want (optional)",
        "",
        height=80,
    )
    rows_schema = st.number_input("Rows to generate (schema sampler)", min_value=50, max_value=50000, value=1000)
    if st.button("Generate via LLM Schema"):
        with st.spinner("Calling LLM for schema and sampling locally..."):
            try:
                def _produce_schema_text(desc: str):
                    sys_instructions = (
                        "You are an expert medical data scientist. "
                        "Your output MUST be a single, valid, minified JSON object and nothing else. "
                        "Do not use markdown fences like ```json. "
                        "The JSON schema should be: "
                        '{"columns": {"column_name": {"type": "...", "details": {...}}}, "constraints": ["..."]}. '
                        "Valid types are: 'int', 'float', 'category'. "
                        "For 'category' type, details must include 'values' (a list of strings) and may include 'probs' (a list of probabilities). "
                        "For numeric types ('int', 'float'), details must include a 'range' [min, max] and may include 'dist': 'truncated_normal' with 'mean' and 'sd'. "
                        "Include sensible constraints, for example: '\"age\" > \"medication_count\" * 5'. "
                        "Do not include any Personally Identifiable Information (PII)."
                    )
                    if desc and desc.strip():
                        model_name="llama3.2"
                        llm = ChatOpenAI(
                            model=model_name,
                            openai_api_key=os.environ.get("OPENAI_API_KEY"),
                            openai_api_base=os.environ.get("OPENAI_BASE_URL"),
                        )
                        prompt = sys_instructions + "\nUser request: " + desc.strip()
                        return llm.invoke(prompt).content
                    if rag_chain is not None:
                        rag_req = (
                            "Based on the uploaded documents, produce ONLY a minified JSON schema for a realistic synthetic tabular dataset "
                            "capturing key variables in this domain. Follow this structure: "
                            '{"columns":{"column_name":{"type": "...", "details": {...}}}, "constraints":["..."]}. '
                            "Avoid PII. No prose, no markdown."
                        )
                        return get_answer(rag_req, rag_chain)
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

                df_gen = _sample_base(schema, int(rows_schema))
                df_gen = _apply_copula(df_gen, schema)
                df_gen = _enforce_constraints(df_gen, schema).reset_index(drop=True)

                st.session_state.final_synthetic_df = df_gen
                st.success(f"Generated {len(df_gen)} rows from LLM-defined schema.")
                st.dataframe(df_gen.head())
            except Exception as e:
                st.error(f"Failed to generate via LLM schema: {e}")

    # --- Step 2: Synthesize Data with SDV ---
    if 'blueprint_df' in st.session_state:
        st.subheader("Step 2: Generate Large Dataset with SDV")
        st.write("Blueprint Data Preview:")
        st.dataframe(st.session_state.blueprint_df.head())

        num_rows_sdv = st.number_input("Number of final synthetic records to generate", min_value=100, max_value=20000, value=1000)

        if st.button("Train Synthesizer and Generate Final Data"):
            with st.spinner("Training SDV synthesizer... This may take several minutes."):
                try:
                    blueprint_df = st.session_state.blueprint_df.copy()
                    
                    # Show original data for debugging
                    st.write("**Original Data Sample:**")
                    st.dataframe(blueprint_df.head())
                    
                    # Clean the data: remove any problematic characters and ensure proper data types
                    for col in blueprint_df.columns:
                        # Clean all columns regardless of type
                        blueprint_df[col] = blueprint_df[col].astype(str).str.strip()
                        
                        # Replace problematic values like '(pii)' with empty string - use multiple approaches
                        blueprint_df[col] = blueprint_df[col].str.replace('(pii)', '', case=False)
                        blueprint_df[col] = blueprint_df[col].str.replace('pii', '', case=False)
                        blueprint_df[col] = blueprint_df[col].str.replace(r'\(pii\)', '', case=False, regex=True)
                        blueprint_df[col] = blueprint_df[col].str.replace(r'pii', '', case=False, regex=True)
                        
                        # Also remove any values that contain 'pii' anywhere (case insensitive)
                        blueprint_df[col] = blueprint_df[col].apply(lambda x: '' if 'pii' in str(x).lower() else x)
                        
                        # Clean up any extra whitespace
                        blueprint_df[col] = blueprint_df[col].str.strip()
                        
                        # Remove any values that are just empty strings or whitespace
                        blueprint_df[col] = blueprint_df[col].replace(['', 'nan', 'None'], np.nan)
                    
                    # Remove any completely empty rows
                    blueprint_df = blueprint_df.dropna(how='all')
                    
                    # Remove rows where all values are empty strings
                    blueprint_df = blueprint_df[~(blueprint_df == '').all(axis=1)]
                    
                    # Additional validation: check for any remaining problematic values BEFORE metadata detection
                    st.write("**Pre-validation Check:**")
                    for col_name in blueprint_df.columns:
                        # Check for any remaining problematic values - more thorough check
                        # First, convert to string and check for problematic patterns
                        col_as_str = blueprint_df[col_name].astype(str)
                        
                        # Check for any remaining problematic values
                        problematic_mask = col_as_str.str.contains('pii|\(pii\)', case=False, na=False)
                        problematic_count = problematic_mask.sum()
                        if problematic_count > 0:
                            st.warning(f"Found {problematic_count} rows with problematic values in '{col_name}'. Removing them.")
                            st.write(f"Problematic values found: {blueprint_df[problematic_mask][col_name].tolist()}")
                            blueprint_df = blueprint_df[~problematic_mask]
                        
                        # Also check for any values that contain 'pii' anywhere in the string
                        pii_anywhere_mask = col_as_str.str.lower().str.contains('pii', na=False)
                        pii_anywhere_count = pii_anywhere_mask.sum()
                        if pii_anywhere_count > 0:
                            st.warning(f"Found {pii_anywhere_count} rows with 'pii' anywhere in '{col_name}'. Removing them.")
                            st.write(f"Values with 'pii': {blueprint_df[pii_anywhere_mask][col_name].tolist()}")
                            blueprint_df = blueprint_df[~pii_anywhere_mask]
                        
                        # Check for any values that are just empty strings, whitespace, or 'nan'
                        empty_mask = col_as_str.str.strip().isin(['', 'nan', 'None', 'null'])
                        empty_count = empty_mask.sum()
                        if empty_count > 0:
                            st.warning(f"Found {empty_count} rows with empty/invalid values in '{col_name}'. Removing them.")
                            st.write(f"Empty values found: {blueprint_df[empty_mask][col_name].tolist()}")
                            blueprint_df = blueprint_df[~empty_mask]
                    
                    if len(blueprint_df) == 0:
                        st.error("No valid data remaining after cleaning. Please regenerate the blueprint.")
                        st.stop()
                    
                    st.write(f"**Data Shape after cleaning:** {blueprint_df.shape}")
                    st.write("**Cleaned Data Sample:**")
                    st.dataframe(blueprint_df.head())
                    
                    # Show what values are in each column for debugging
                    st.write("**Column Value Analysis:**")
                    for col_name in blueprint_df.columns:
                        unique_values = blueprint_df[col_name].value_counts().head(5)
                        st.write(f"{col_name}: {list(unique_values.index)}")
                    
                    # Final check: ensure no problematic values remain anywhere
                    st.write("**Final PII Check:**")
                    for col_name in blueprint_df.columns:
                        # Check for any values containing 'pii' (case insensitive)
                        pii_check = blueprint_df[col_name].astype(str).str.lower().str.contains('pii', na=False)
                        if pii_check.any():
                            problematic_values = blueprint_df[pii_check][col_name].tolist()
                            st.error(f"CRITICAL: Found PII values in '{col_name}': {problematic_values}")
                            st.error("Removing these rows...")
                            blueprint_df = blueprint_df[~pii_check]
                    
                    if len(blueprint_df) == 0:
                        st.error("No valid data remaining after final PII check. Please regenerate the blueprint.")
                        st.stop()
                    
                    st.write(f"**Final data shape after PII check:** {blueprint_df.shape}")
                    
                    # Final check: ensure no problematic values remain anywhere before metadata detection
                    st.write("**Final Data Quality Check:**")
                    for col_name in blueprint_df.columns:
                        col_as_str = blueprint_df[col_name].astype(str)
                        # Check for any values containing 'pii' (case insensitive)
                        pii_check = col_as_str.str.lower().str.contains('pii', na=False)
                        if pii_check.any():
                            problematic_values = blueprint_df[pii_check][col_name].tolist()
                            st.error(f"CRITICAL: Found PII values in '{col_name}': {problematic_values}")
                            st.error("Removing these rows...")
                            blueprint_df = blueprint_df[~pii_check]
                    
                    if len(blueprint_df) == 0:
                        st.error("No valid data remaining after final quality check. Please regenerate the blueprint.")
                        st.stop()
                    
                    st.write(f"**Final data shape after quality check:** {blueprint_df.shape}")
                    
                    # One more global PII scrub and categorical normalization before SDV
                    # Drop any row where any cell still contains PII-like tokens
                    pii_row_mask = blueprint_df.astype(str).apply(
                        lambda s: s.str.contains(r"(?i)\bpii\b|\(\s*pii\s*\)", na=False)
                    ).any(axis=1)
                    if pii_row_mask.any():
                        st.warning(f"Dropping {int(pii_row_mask.sum())} row(s) containing residual PII tokens.")
                        blueprint_df = blueprint_df.loc[~pii_row_mask].copy()

                    # Normalize gender to {M, F, Other} if present; drop invalids
                    if 'gender' in blueprint_df.columns:
                        # Pre-drop any gender strings that still carry PII tokens
                        mask_gender_pii = blueprint_df['gender'].astype(str).str.contains(r"(?i)\bpii\b|\(\s*pii\s*\)", na=False)
                        if mask_gender_pii.any():
                            st.warning(f"Dropping {int(mask_gender_pii.sum())} row(s) with PII-like tokens in 'gender'.")
                            blueprint_df = blueprint_df.loc[~mask_gender_pii].copy()
                        def _norm_gender(val):
                            s = str(val).strip().lower()
                            if s in {"m", "male"}: return "M"
                            if s in {"f", "female"}: return "F"
                            if s in {"other", "o", "x", "u", "unknown", "na", "n/a"}: return "Other"
                            return np.nan
                        blueprint_df['gender'] = blueprint_df['gender'].map(_norm_gender)
                        invalid_gender = blueprint_df['gender'].isna().sum()
                        if invalid_gender:
                            st.warning(f"Dropping {int(invalid_gender)} row(s) with invalid gender values.")
                            blueprint_df = blueprint_df.dropna(subset=['gender'])
                        # Enforce whitelist strictly
                        allowed_gender = {"M", "F", "Other"}
                        not_allowed = ~blueprint_df['gender'].isin(allowed_gender)
                        if not_allowed.any():
                            st.warning(f"Dropping {int(not_allowed.sum())} row(s) with non-whitelisted gender values.")
                            blueprint_df = blueprint_df.loc[~not_allowed].copy()

                    # Normalize numeric columns: strip non-numeric chars, coerce to float, drop NaNs and out-of-range values
                    numeric_targets = {
                        'age': (0, 120),
                        'bmi': (10, 90),
                        'systolic_bp': (60, 260),
                        'diastolic_bp': (30, 200),
                        'cholesterol': (50, 1000),
                    }
                    for col, (lo, hi) in numeric_targets.items():
                        if col in blueprint_df.columns:
                            # Remove any non-numeric characters
                            blueprint_df[col] = blueprint_df[col].astype(str).str.replace(r"[^0-9\.-]", "", regex=True)
                            # Convert to numeric
                            blueprint_df[col] = pd.to_numeric(blueprint_df[col], errors='coerce')
                            # Drop NaNs
                            n_nans = blueprint_df[col].isna().sum()
                            if n_nans:
                                st.warning(f"Dropping {int(n_nans)} row(s) with invalid numeric in '{col}'.")
                                blueprint_df = blueprint_df.dropna(subset=[col])
                            # Clamp to plausible range and drop outliers
                            out_of_range = (blueprint_df[col] < lo) | (blueprint_df[col] > hi)
                            n_oob = int(out_of_range.sum())
                            if n_oob:
                                st.warning(f"Dropping {n_oob} row(s) out of range for '{col}' [{lo}, {hi}].")
                                blueprint_df = blueprint_df.loc[~out_of_range]

                    # Show final clean data
                    st.write("**Final Clean Data Sample:**")
                    st.dataframe(blueprint_df.head())
                    
                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(data=blueprint_df)
                    metadata.set_primary_key(None)

                    # First, set all columns to not PII
                    for column in metadata.columns:
                        metadata.update_column(
                            column_name=column,
                            pii=False
                        )
                    
                    # Force all columns to be either numerical or categorical (no ID columns)
                    # Prefer explicit assignment for known columns
                    for col_name in blueprint_df.columns:
                        desired = None
                        if col_name in numeric_targets:
                            desired = 'numerical'
                        elif col_name in ['gender', 'diagnosis']:
                            desired = 'categorical'
                        if desired:
                            try:
                                metadata.update_column(column_name=col_name, sdtype=desired)
                            except Exception:
                                pass
                        else:
                            current_sdtype = metadata.columns[col_name]['sdtype']
                            if current_sdtype == 'id':
                                metadata.update_column(column_name=col_name, sdtype='categorical')
                            elif not pd.api.types.is_numeric_dtype(blueprint_df[col_name]):
                                if current_sdtype not in ['categorical', 'text']:
                                    metadata.update_column(column_name=col_name, sdtype='categorical')
                    
                    # Debug: Show metadata information
                    st.write("**Metadata Configuration:**")
                    for col_name, col_info in metadata.columns.items():
                        st.write(f"- {col_name}: {col_info['sdtype']} (PII: {col_info.get('pii', False)})")
                    
                    st.write(f"**Final Data Shape:** {blueprint_df.shape}")
                    
                    synthesizer = CTGANSynthesizer(metadata)
                    synthesizer.fit(blueprint_df)
                    
                    st.session_state.final_synthetic_df = synthesizer.sample(num_rows=num_rows_sdv)
                    st.success("Final synthetic dataset generated!")
                except Exception as e:
                    st.error(f"An error occurred during SDV synthesis: {e}")

    # --- Display Final Results ---
    if 'final_synthetic_df' in st.session_state:
        st.subheader("Final Generated Data")
        final_df = st.session_state.final_synthetic_df
        st.dataframe(final_df)
        
        csv_final = final_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Final Data as CSV",
            data=csv_final,
            file_name='final_synthetic_patient_data.csv',
            mime='text/csv',
        )

        st.markdown("---")
        st.subheader("Generate ECGs Conditioned on This Tabular Data")
        st.markdown("Map rows to ECG label vectors and synthesize matching signals.")

        # Row selection
        selectable_indices = list(final_df.index)
        default_sel = selectable_indices[: min(5, len(selectable_indices))]
        selected_rows = st.multiselect("Select row indices to condition on", selectable_indices, default=default_sel)
        num_ecg_per_row = st.number_input("ECGs per selected row", min_value=1, max_value=10, value=1)
        positive_score = st.slider("Label score for positive conditions", min_value=0.1, max_value=1.0, value=1.0, step=0.1)

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

        # Build a suggested mapping from unique values to label codes
        suggested = {}
        if diag_col:
            unique_vals = sorted([str(v) for v in final_df[diag_col].dropna().unique()])[:30]
            for v in unique_vals:
                key = v.strip().upper().replace(" ", "_")
                # naive suggestions
                for code in ["AMI","AFIB","LBBB","RBBB","NORM","NST_","STACH","SBRAD"]:
                    if code in key:
                        suggested[v] = code
                        break
                if v not in suggested and key in (idx_to_name or []):
                    suggested[v] = key
        mapping_help = {
            "Myocardial_Infarction": "AMI",
            "Atrial Fibrillation": "AFIB",
            "Normal": "NORM"
        }
        default_map = {**mapping_help, **suggested}
        default_map_json = json.dumps(default_map, indent=2)
        mapping_json = st.text_area("Value-to-ECG-label mapping (JSON)", value=default_map_json, height=180)

        # Optional numeric threshold rules using pandas.eval expressions
        st.caption("Optional: add rules like {'when': 'troponin > 0.4', 'set': ['AMI']} one per line (JSON list)")
        rules_default = """
[
  {"when": "troponin > 0.4", "set": ["AMI"]}
]
""".strip()
        rules_json = st.text_area("Conditional rules (JSON list)", value=rules_default, height=120)

        # Load ECG generation config
        try:
            with open('sssd/config/config_SSSD_ECG.json') as f:
                ecg_cfg = json.load(f)
        except Exception as e:
            ecg_cfg = None
            st.warning(f"Could not load ECG config: {e}")

        if st.button("Generate Conditioned ECGs", disabled=(ecg_cfg is None or not selected_rows)):
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

                # Parse mapping and rules
                value_to_label = {}
                try:
                    parsed = json.loads(mapping_json) if mapping_json.strip() else {}
                    for k, v in parsed.items():
                        if isinstance(v, str):
                            value_to_label[str(k)] = [v]
                        elif isinstance(v, list):
                            value_to_label[str(k)] = [str(x) for x in v]
                except Exception as e:
                    st.error(f"Invalid mapping JSON: {e}")
                    st.stop()

                try:
                    rule_list = json.loads(rules_json) if rules_json.strip() else []
                    if not isinstance(rule_list, list):
                        raise ValueError("Rules must be a JSON list")
                except Exception as e:
                    st.error(f"Invalid rules JSON: {e}")
                    st.stop()

                # Build label matrix
                def labels_for_row(row: pd.Series) -> np.ndarray:
                    vec = np.zeros((num_classes,), dtype=np.float32)
                    # categorical mapping
                    if diag_col:
                        val = str(row.get(diag_col, ""))
                        for code in value_to_label.get(val, []):
                            if code in name_to_idx:
                                vec[name_to_idx[code]] = positive_score
                    # rule-based mapping
                    env = {col: row[col] for col in final_df.columns}
                    for rule in rule_list:
                        try:
                            expr = str(rule.get("when", "")).strip()
                            to_set = rule.get("set", [])
                            if expr:
                                ok = pd.eval(expr, engine='python', local_dict=env)
                                if bool(ok):
                                    for code in to_set:
                                        if code in name_to_idx:
                                            vec[name_to_idx[code]] = positive_score
                        except Exception:
                            continue
                    return vec

                per_row = []
                for ridx in selected_rows:
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
    st.header("Generation Settings")
    ckpt_iter = st.text_input("Checkpoint Iteration", value="max")
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
            # Additionally list top labels in text form
            if label_info:
                top_items = sorted(label_info.items(), key=lambda kv: kv[1], reverse=True)[:10]
                st.markdown("**Top labels**")
                for name, score in top_items:
                    st.write(f"{name}: {score:.3f}")
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
