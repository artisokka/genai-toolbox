import streamlit as st
import numpy as np
import torch
import json
import os
import sys
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
sssd_dir = os.path.join(script_dir, 'sssd')
if sssd_dir not in sys.path:
    sys.path.insert(0, sssd_dir)

# --- Import sssd components ---
try:
    from sssd.models.SSSD_ECG import SSSD_ECG # Adjusted import based on likely structure
    from sssd.utils.util import calc_diffusion_hyperparams, sampling_label, find_max_epoch # Adjusted import
    from sssd.visualize_ecg import plot_ecg # Assuming visualize_ecg.py is directly usable
except ImportError as e:
    st.error(f"Error importing required modules from 'sssd': {e}")
    st.error(f"Please ensure the 'sssd' directory is structured correctly and present in the same directory as synthgen_app.py ({script_dir})")
    st.stop() # Stop execution if imports fail


# --- Configuration Loading ---
# Load diffusion hyperparameters from config file (relative path)
CONFIG_PATH = os.path.join(sssd_dir, 'config', 'config_SSSD_ECG.json')
OUTPUT_DIR_DEFAULT = os.path.join(script_dir, "generated_ecg")

try:
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    diffusion_hyperparams = calc_diffusion_hyperparams(**config["diffusion_config"])
except FileNotFoundError:
    st.error(f"Configuration file not found at: {CONFIG_PATH}")
    st.stop()
except Exception as e:
    st.error(f"Error loading or processing configuration: {e}")
    st.stop()


# --- Helper Function for ECG Generation ---
# Adapted from sssd/inference.py
@st.cache_resource # Cache the model loading
def load_model(checkpoint_path, config_dict):
    """Loads the SSSD_ECG model from a checkpoint."""
    model = SSSD_ECG(**config_dict["wave_config"]).cuda() # Assuming CUDA is available
    try:
        state_dict = torch.load(checkpoint_path, map_location='cuda') # Adjust map_location if no GPU
        model.load_state_dict(state_dict['model'])
        model.eval()
        st.success(f"Model loaded successfully from {os.path.basename(checkpoint_path)}")
        return model
    except FileNotFoundError:
        st.error(f"Checkpoint file not found: {checkpoint_path}")
        return None
    except Exception as e:
        st.error(f"Error loading checkpoint: {e}")
        return None

def generate_ecg_samples(model, num_samples, batch_size, output_dir, diffusion_hyperparams_dict):
    """Generates ECG samples using the loaded model."""
    generated_files = []
    try:
        os.makedirs(output_dir, exist_ok=True)
        st.write(f"Generating {num_samples} samples...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        generated_count = 0
        while generated_count < num_samples:
            current_batch_size = min(batch_size, num_samples - generated_count)
            status_text.text(f"Generating batch ({generated_count+1}-{generated_count+current_batch_size}/{num_samples})...")

            # Use sampling_label function for generation
            # Assuming sampling_label handles device placement internally or model is already on device
            _, generated_signal = sampling_label(
                model,
                current_batch_size, # Pass batch size
                diffusion_hyperparams_dict,
                print_process=False # Don't print process details to console in Streamlit app
            )

            # Save generated samples
            for i in range(generated_signal.shape[0]):
                if generated_count < num_samples:
                    sample_index = generated_count
                    filename = os.path.join(output_dir, f"generated_ecg_{sample_index}.npy")
                    np.save(filename, generated_signal[i].cpu().numpy()) # Save as numpy array
                    generated_files.append(filename)
                    generated_count += 1

            progress = int(100 * generated_count / num_samples)
            progress_bar.progress(progress)

        status_text.text(f"Generation complete. {num_samples} samples saved to '{output_dir}'.")
        st.success("ECG generation finished!")
        return generated_files

    except Exception as e:
        st.error(f"Error during ECG generation: {e}")
        return []


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("SynthGen: Synthetic Health Data Generator")

tab1, tab2 = st.tabs(["📈 ECG Data Generation", "📊 Tabular Health Data (Placeholder)"])

# --- ECG Generation Tab ---
with tab1:
    st.header("Synthetic ECG Generation")
    st.markdown("Generate synthetic 12-lead ECG signals using a pre-trained SSSD-ECG model.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Configuration")
        uploaded_file = st.file_uploader("1. Upload Model Checkpoint (.pkl/.pt)", type=['pkl', 'pt'])
        num_samples = st.number_input("2. Number of Samples to Generate", min_value=1, max_value=1000, value=10, step=1)
        output_dir_input = st.text_input("3. Output Directory for Samples", value=OUTPUT_DIR_DEFAULT)
        # batch_size = st.number_input("Batch Size (Advanced)", min_value=1, value=config["inference_config"].get("batch_size", 8)) # Get from config or default
        batch_size = config["inference_config"].get("batch_size", 8)

        generate_button = st.button("Generate ECG Samples", key="ecg_generate", disabled=(uploaded_file is None))

    with col2:
        st.subheader("Status & Visualization")
        status_placeholder = st.empty()
        plot_placeholder = st.empty()

    if generate_button and uploaded_file is not None:
        status_placeholder.info("Starting ECG generation process...")
        plot_placeholder.empty() # Clear previous plot

        # Ensure the output directory exists
        if not os.path.exists(output_dir_input):
             try:
                 os.makedirs(output_dir_input)
                 status_placeholder.write(f"Created output directory: {output_dir_input}")
             except Exception as e:
                 status_placeholder.error(f"Could not create output directory: {e}")
                 st.stop()

        # Save the uploaded file temporarily to pass its path
        # Note: Streamlit's uploaded file is an in-memory buffer
        temp_checkpoint_path = os.path.join(script_dir, uploaded_file.name) # Save in app's root dir
        try:
            with open(temp_checkpoint_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load Model
            status_placeholder.write(f"Loading model from {uploaded_file.name}...")
            model = load_model(temp_checkpoint_path, config) # Pass config dict

            if model:
                status_placeholder.write("Model loaded. Starting generation...")
                # Generate Samples
                generated_files = generate_ecg_samples(model, num_samples, batch_size, output_dir_input, diffusion_hyperparams)

                if generated_files:
                    status_placeholder.success(f"Successfully generated {len(generated_files)} samples in '{output_dir_input}'.")

                    # --- Visualization Section ---
                    st.subheader("Visualize Generated Sample")
                    # Provide a dropdown to select a generated file for plotting
                    selected_file = st.selectbox("Select a generated sample to visualize:", generated_files, format_func=os.path.basename)

                    if selected_file:
                        try:
                            # Load the selected numpy file
                            ecg_data = np.load(selected_file)
                            # st.write(f"Shape of loaded data: {ecg_data.shape}") # Debug shape

                            # Use the plotting function from visualize_ecg.py
                            fig = plot_ecg(ecg_data) # Assuming plot_ecg returns a matplotlib figure
                            plot_placeholder.pyplot(fig)
                            st.success(f"Displayed ECG: {os.path.basename(selected_file)}")

                        except Exception as e:
                            plot_placeholder.error(f"Error loading or plotting {os.path.basename(selected_file)}: {e}")
                else:
                    status_placeholder.warning("No files were generated.")
            else:
                 status_placeholder.error("Model loading failed. Cannot generate samples.")

        except Exception as e:
             status_placeholder.error(f"An error occurred: {e}")
        finally:
             # Clean up temporary checkpoint file
             if os.path.exists(temp_checkpoint_path):
                 os.remove(temp_checkpoint_path)


# --- Tabular Data Generation Tab (Integrated with RAG) ---
with tab2:
    st.header("Query Documents with RAG")
    st.markdown("Ask questions about the documents processed into the knowledge base.")

    # --- RAG Setup ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(script_dir, "rag", "DataIndex") # Adjust path if needed

    status_placeholder_rag = st.empty()
    status_placeholder_rag.info(f"Attempting to load FAISS index from: {index_path}")

    vector_store = load_faiss_index(index_path)
    llm = get_ollama_llm() # Uses default 'llama3.1'

    if vector_store and llm:
        status_placeholder_rag.success("RAG system components loaded successfully.")
        qa_chain = create_rag_chain(vector_store, llm)

        # --- RAG Query Interface ---
        st.subheader("Ask a Question")
        user_question = st.text_input("Enter your question about the documents:")

        if st.button("Get Answer", key="rag_query") and user_question:
            if qa_chain:
                with st.spinner("Retrieving and generating answer..."):
                    answer = get_rag_answer(user_question, qa_chain)
                    st.subheader("Answer:")
                    st.markdown(answer) # Use markdown for better formatting
            else:
                st.error("RAG chain is not available. Cannot process query.")
        elif not user_question and st.session_state.get('rag_query'):
             st.warning("Please enter a question.") # Handle empty input on button click

    else:
        status_placeholder_rag.error("Failed to load RAG components. Please check index path, Ollama status, and dependencies.")