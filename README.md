## GenAI Toolbox: Synthetic Health Data Generator

A Streamlit app that lets you:
- Generate synthetic 12‑lead ECG waveforms with an SSSD‑ECG diffusion model
- Visualize generated ECGs in a standard clinical layout
- Build a FAISS index from your PDFs and query it via a local LLM (Ollama)

### What’s inside
- **ECG generator**: Structured State Space + WaveNet diffusion model, label‑conditional
- **RAG**: Convert PDFs → text → FAISS index, query using `llama3.1` via Ollama

## Setup

1) Clone the repository
```bash
git clone https://github.com/artisokka/genai-toolbox
cd genai-toolbox
```

2) Create and activate a virtualenv
```bash
python -m venv venv
# Windows
.\\venv\\Scripts\\activate
# macOS/Linux
source venv/bin/activate
```

3) Install dependencies
```bash
pip install -r requirements.txt
```

4) Download the SSSD‑ECG checkpoint (required for ECG generation)
```bash
# Create the checkpoint folder and download the 100000.pkl weights
mkdir -p sssd/checkpoint
curl -L -o sssd/checkpoint/100000.pkl "https://figshare.com/s/81834b24a4711c2a5c55?file=38890809"

# Alternatively, download via browser and place the file at:
# sssd/checkpoint/100000.pkl
```

5) Install Ollama and the local model for RAG
```bash
# Install Ollama from https://ollama.com
ollama serve
ollama pull llama3.1
```

6) Insert your .env file in rag/
```bash
OPENAI_API_KEY=<your-token>
OPENAI_BASE_URL=<your-base-url>
```

## Run the app
```bash
python -m streamlit run synthgen_app.py
```

## RAG: tabular data generation
- Index your documents to use as a knowledge base
- Specify the columns to be generated or let the app infer the dataset structure from your documents
- Optionally generate ECG:s for tabular data rows

## ECG generation
- Config file: `sssd/config/config_SSSD_ECG.json`
  - `gen_config.output_directory`: where outputs are saved (default `generated_ecg/`)
  - `gen_config.ckpt_path`: folder containing model checkpoint `.pkl` files (default `sssd/checkpoint/`)
- In the app under “Synthetic ECG Generation”:
  - Set checkpoint iteration (`max` uses the latest file in `ckpt_path`)
  - Choose number of samples and click “Generate”
- Outputs are saved to:
  - `generated_ecg/run_YYYYmmDD_HHMMSS/` containing `*_samples.npy` and matching `*_labels.npy`
- Visualization:
  - The app lists available `run_*` folders and lets you preview batches and per‑sample plots

## Project structure
```bash
.
├── synthgen_app.py                # Streamlit app (ECG + RAG)
├── sssd/
│   ├── models/                    # SSSD‑ECG model
│   ├── utils/                     # helpers
│   ├── visualize_ecg.py           # plotting utilities
│   └── config/config_SSSD_ECG.json
├── rag/
│   ├── pdf_to_text.py             # PDFs → text
│   ├── txt_to_index.py            # text → FAISS index
│   └── main.py                    # RAG chain helpers
└── generated_ecg/                 # outputs (created at runtime)
```

## References
```text
@article{ALCARAZ2023107115,
  title={Diffusion-based conditional ECG generation with structured state space models},
  journal={Computers in Biology and Medicine},
  volume={163},
  pages={107115},
  year={2023},
  doi={10.1016/j.compbiomed.2023.107115},
  author={Juan Miguel Lopez Alcaraz and Nils Strodthoff}
}
```
```text
@misc{khan2024developingretrievalaugmentedgeneration,
  title={Developing Retrieval Augmented Generation (RAG) based LLM Systems from PDFs: An Experience Report},
  author={Khan, A. A. and Hasan, M. T. and Kemell, K. K. and Rasku, J. and Abrahamsson, P.},
  year={2024},
  eprint={2410.15944},
  archivePrefix={arXiv},
  primaryClass={cs.SE}
}
```

### Original implementation
- Paper: https://www.sciencedirect.com/science/article/pii/S0010482523005802
- Code: https://github.com/AI4HealthUOL/SSSD-ECG

License: MIT