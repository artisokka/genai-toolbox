# SynthGen: Generate synthetic tabular health data and ECG's

This project implements a Signal Space Diffusion Model for generating synthetic ECG signals and XXX for implementing synthetic tabular data. It uses a combination of WaveNet architecture and diffusion models to generate high-quality ECG waveforms.

## Features
- ECG signal generation using diffusion models
- WaveNet-based architecture
- Label-conditional generation
- Streamlit web interface for easy interaction

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure 

.
├── sssd/
│ ├── config/
│ │ └── config_SSSD_ECG.json
│ └── ...
├── rag/
│ └── main.py
├── requirements.txt
└── README.md

## Configuration

The model configuration is defined in `sssd/config/config_SSSD_ECG.json` and includes:
- Diffusion parameters (T, beta values)
- WaveNet architecture settings
- Training configuration
- Dataset parameters
- Generation settings

## Usage

Run the Streamlit app:
```bash
python -m streamlit run synthgen_app.py
```

## Model Architecture

The model uses:
- WaveNet-based architecture with residual layers
- Diffusion process with T=200 steps
- S4 layers for improved temporal modeling
- Label conditioning for controlled generation

## Requirements
- Python 3.8+
- PyTorch
- Streamlit
- Additional dependencies in requirements.txt

## Credits

This project builds upon the research presented in:

Lopez Alcaraz, J. M., & Strodthoff, N. (2023). Diffusion-based conditional ECG generation with structured state space models. Computers in Biology and Medicine, 163, 107115.

```text
@article{ALCARAZ2023107115,
    title = {Diffusion-based conditional ECG generation with structured state space models},
    journal = {Computers in Biology and Medicine},
    volume = {163},
    pages = {107115},
    year = {2023},
    issn = {0010-4825},
    doi = {https://doi.org/10.1016/j.compbiomed.2023.107115},
    url = {https://www.sciencedirect.com/science/article/pii/S0010482523005802},
    author = {Juan Miguel Lopez Alcaraz and Nils Strodthoff},
    keywords = {Cardiology, Electrocardiography, Signal processing, Synthetic data, Diffusion models, Time series}
}
```

### Original Implementation
- [Link to original paper](https://www.sciencedirect.com/science/article/pii/S0010482523005802)
- [Link to original code repository](https://github.com/AI4HealthUOL/SSSD-ECG)

This implementation extends the original work by integrating SSSD-ECG inference functionality as a part of a larger web app.

## License
MIT License

## Contact
[Your contact information]

