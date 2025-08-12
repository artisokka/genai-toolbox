import streamlit as st
st.set_page_config(layout="wide")

import numpy as np
import torch
import json
import pickle
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from app_utils import ensure_ollama_running, _build_index_to_name, _humanize_diag

# --- Import RAG components ---
from rag.main import load_faiss_index, create_rag_system, get_answer, extract_table
from rag.pdf_to_text import convert_pdfs_to_text
from rag.txt_to_index import create_faiss_index

# --- Import Gemma model components ---
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

script_dir = os.path.dirname(os.path.abspath(__file__))
sssd_dir = os.path.join(script_dir, 'sssd')
if sssd_dir not in sys.path:
    sys.path.insert(0, sssd_dir)

# --- Import sssd components ---
try:
    from sssd.models.SSSD_ECG import SSSD_ECG # Adjusted import based on likely structure
    from sssd.utils.util import calc_diffusion_hyperparams, sampling_label, find_max_epoch # Adjusted import
    from sssd.visualize_ecg import visualize_ecg_npy
    from sssd.inference import generate
except ImportError as e:
    st.error(f"Error importing required modules from 'sssd': {e}")
    st.error(f"Please ensure the 'sssd' directory is structured correctly and present in the same directory as synthgen_app.py ({script_dir})")
    st.stop() # Stop execution if imports fail



# --- Helper Functions for Gemma 7B Fine-tuned Model ---

@st.cache_resource # Cache the potentially large model
def load_gemma_model(base_model_id, adapter_path):
    """Loads the fine-tuned Gemma 7B model with LoRA adapters."""
    try:
        st.write(f"Loading base model '{base_model_id}' with 4-bit quantization...")
        # Configure quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, # Or torch.float16 if bfloat16 not supported
            bnb_4bit_use_double_quant=True,
        )

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16, # Match compute dtype
            device_map="auto", # Automatically place layers on available GPU/CPU
            trust_remote_code=True,
        )
        st.write("Base model loaded.")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token # Set padding token
        tokenizer.padding_side = "right"
        st.write("Tokenizer loaded.")

        st.write(f"Loading LoRA adapter from '{adapter_path}'...")
        # Load PEFT model (LoRA adapter)
        model = PeftModel.from_pretrained(model, adapter_path)
        st.write("LoRA adapter loaded and merged.")

        model.eval() # Set model to evaluation mode
        st.success("Fine-tuned Gemma 7B model ready.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading Gemma model or adapter: {e}")
        st.error(f"Ensure base model '{base_model_id}' and adapter path '{adapter_path}' are correct and accessible.")
        return None, None

def extract_stats_from_text(model, tokenizer, text_corpus, extraction_prompt):
    """Generates text using the model to extract stats based on the prompt."""
    try:
        # --- Simple Prompt Formatting Example (adjust if your fine-tuning used a different template) ---
        # Using a basic instruction format. Some models prefer specific tags like <start_of_turn> etc.
        # Check the format used during your PEFT fine-tuning.
        prompt_template = f"<s>[INST] {extraction_prompt}\n\nText:\n{text_corpus} [/INST]</s>\nExtracted Statistics:\n"
        # --- End Prompt Formatting ---

        inputs = tokenizer(prompt_template, return_tensors="pt", padding=True).to(model.device)

        st.write("Generating extraction...")
        # Adjust generation parameters as needed
        outputs = model.generate(
            **inputs,
            max_new_tokens=200, # Limit number of generated tokens
            temperature=0.1,    # Lower temperature for more deterministic output
            do_sample=True,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1
        )

        # Decode generated tokens, skipping special tokens and the prompt part
        # Note: Decoding strategies might need adjustment based on tokenizer and model behavior
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Basic cleanup: Try to remove the prompt from the output
        # This might need refinement depending on how the model echoes the prompt
        extracted_part = generated_text.split("[/INST]</s>")[-1].strip()
        # Further refine if "Extracted Statistics:" prefix is included:
        if extracted_part.startswith("Extracted Statistics:"):
             extracted_part = extracted_part.replace("Extracted Statistics:", "", 1).strip()


        st.write("Extraction complete.")
        return extracted_part
    except Exception as e:
        st.error(f"Error during text generation: {e}")
        return "Error during extraction."

        

# --- Streamlit App UI ---
st.title("GenAI Toolbox: Synthetic Health Data Generator")

tab1, tab2 = st.tabs(["📈 Synthetic ECG Data Generation", "📊 Tabular Health Data"])

# --- ECG Generation Tab ---
with tab1:
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


# --- Tabular Data Generation Tab (Integrated with RAG) ---
with tab2:
    st.header("Query Documents with RAG")
    st.markdown("Upload PDFs, process them into the knowledge base, and ask questions about the documents.")

    # --- Paths setup ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "rag", "Data")
    text_folder = os.path.join(script_dir, "rag", "DataTxt")
    index_path = os.path.join(script_dir, "rag", "DataIndex")
    faiss_index_path = os.path.join(index_path, "index.faiss")
    synthetic_data_dir = os.path.join(script_dir, "rag", "SyntheticData")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(text_folder, exist_ok=True)
    os.makedirs(index_path, exist_ok=True)
    os.makedirs(synthetic_data_dir, exist_ok=True)  # New folder for synthetic data outputs

    status_placeholder_rag = st.empty()

    # --- PDF Upload and Processing Section ---
    st.subheader("Upload and Process PDFs")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=['pdf'],
        accept_multiple_files=True
    )

    # Display uploaded files with delete buttons
    if uploaded_files:
        st.write("Uploaded files:")
        for i, file in enumerate(uploaded_files):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{i+1}. {file.name}")
            with col2:
                if st.button("Delete", key=f"delete_{i}"):
                    uploaded_files.pop(i)
                    st.experimental_rerun()

    # Process PDFs and rebuild index
    if uploaded_files and st.button("Process PDFs and Rebuild Index"):
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

                # Clear previous RAG instance
                if 'rag_system' in st.session_state:
                    del st.session_state.rag_system

            except Exception as e:
                st.error(f"An error occurred while processing PDFs: {str(e)}")

    # --- Initialize RAG if index exists ---
    if os.path.exists(faiss_index_path):
        status_placeholder_rag.info("Found existing FAISS index. Initializing RAG system...")

        qa_chain = create_rag_system(index_path)
        if qa_chain is None:
            status_placeholder_rag.error("Failed to load RAG system: FAISS index is missing or corrupted.")
            st.info("Please upload and process PDFs to create the index.")
        else:
            status_placeholder_rag.success("RAG system components loaded successfully.")

            # --- RAG Query Interface ---
            st.subheader("Ask a Question")
            user_question = st.text_input("Enter your question about the documents:")

            if st.button("Get Answer", key="rag_query") and user_question:
                with st.spinner("Retrieving and generating answer..."):
                    answer = get_answer(user_question, qa_chain)
                    st.subheader("Answer:")
                    st.markdown(answer)

            st.markdown("---")
            # --- Synthetic Data Generation Inputs ---
            st.subheader("Generate Synthetic Patient Data")

            num_rows = st.number_input("Number of synthetic data rows:", min_value=1, max_value=10000, value=100, step=1)
            columns_input = st.text_input(
                "Optional: Specify columns (comma-separated). Leave empty to let the model auto-detect."
            )

            if st.button("Generate Synthetic Data"):
                if not qa_chain:
                    st.error("RAG system not initialized.")
                else:
                    with st.spinner("Generating synthetic data..."):
                        # Build improved prompt for clean CSV generation
                        synth_prompt = f"""
                        Generate a CSV table with exactly {num_rows} rows of synthetic patient data.
                        
                        CRITICAL REQUIREMENTS:
                        - Output ONLY the CSV data, no explanations or text
                        - Use these exact columns: {columns_input if columns_input.strip() else 'patient_id,age,gender,bmi,blood_pressure_systolic,blood_pressure_diastolic,cholesterol_total,cholesterol_hdl,cholesterol_ldl,triglycerides,diabetes_status,smoking_status,physical_activity_level,family_history_cardiac,medication_count,patient_story'}
                        - Start with the header row exactly as shown above
                        - Use simple comma separation (no quotes unless needed for text with commas)
                        - Keep patient_story brief (max 100 characters)
                        - Use simple values: patient_id (numbers), age (18-80), gender (M/F), bmi (18-40), blood pressure (90-180/60-120), cholesterol (100-300), diabetes_status (Y/N), smoking_status (Y/N), physical_activity_level (Low/Moderate/High), family_history_cardiac (Y/N), medication_count (0-5)
                        
                        Format the output as a clean CSV table only.
                        """

                        synthetic_output = get_answer(synth_prompt, qa_chain)

                        # Clean the output to extract only CSV data
                        import re
                        import pandas as pd
                        import datetime
                        from io import StringIO
                        
                        # Try to extract and clean CSV data from the response
                        lines = synthetic_output.strip().split('\n')
                        csv_lines = []
                        
                        for line in lines:
                            line = line.strip()
                            # Check if line looks like CSV (contains commas and reasonable content)
                            if ',' in line and len(line.split(',')) >= 3:
                                # Skip lines that are just separators or headers with dashes
                                if not re.match(r'^[-\s,|]+$', line) and not line.startswith('---'):
                                    csv_lines.append(line)
                                    in_csv_section = True
                            elif in_csv_section and line and not line.startswith('**') and not line.startswith('Note:'):
                                # Continue if we're in CSV section and line has content
                                csv_lines.append(line)
                        
                        if csv_lines:
                            csv_data = '\n'.join(csv_lines)
                            try:
                                # Read CSV data
                                df = pd.read_csv(StringIO(csv_data))
                                
                                # Display the table
                                st.markdown("### Generated Synthetic Patient Data")
                                st.dataframe(df, use_container_width=True)
                                
                                # Show basic statistics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Patients", len(df))
                                with col2:
                                    st.metric("Columns", len(df.columns))
                                with col3:
                                    st.metric("File Size", f"{len(csv_data)/1024:.1f} KB")
                                
                                # Add download button
                                csv_data = df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download CSV",
                                    data=csv_data,
                                    file_name=f"synthetic_patient_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                                
                                # Save to file
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                csv_path = os.path.join(synthetic_data_dir, f"synthetic_data_{timestamp}.csv")
                                df.to_csv(csv_path, index=False)
                                st.success(f"✅ Data saved to {csv_path}")
                                
                            except Exception as e:
                                st.error(f"Error parsing CSV data: {e}")
                                st.text_area("Raw Output", value=synthetic_output, height=200)
                        else:
                            st.error("Could not extract CSV data from the response")
                            st.text_area("Raw Output", value=synthetic_output, height=200)
                            
                            # Show a hint about the expected format
                            st.info("""
                            **Expected Format:** The system should output clean CSV data like:
                            ```
                            patient_id,age,gender,bmi,blood_pressure_systolic,blood_pressure_diastolic,cholesterol_total,cholesterol_hdl,cholesterol_ldl,triglycerides,diabetes_status,smoking_status,physical_activity_level,family_history_cardiac,medication_count,patient_story
                            1,45,M,28.5,120,80,150,45,95,120,N,N,Moderate,Y,2,Patient presents with mild hypertension and family history of cardiac disease...
                            2,32,F,25.5,110,75,180,50,110,85,Y,S,Low,N,1,Patient has diabetes and multiple cardiovascular risk factors...
                            ```
                            """)


    else:
        status_placeholder_rag.warning("No FAISS index found.")
        st.info("To get started, upload and process PDF files to create the index.")