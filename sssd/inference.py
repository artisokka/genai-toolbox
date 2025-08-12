import os
import argparse
import json
import numpy as np
import torch
import time
from datetime import datetime

from sssd.utils.util import find_max_epoch, print_size, sampling_label, calc_diffusion_hyperparams
from sssd.models.SSSD_ECG import SSSD_ECG



def generate_four_leads(tensor):
    leadI = tensor[:,0,:].unsqueeze(1)
    leadschest = tensor[:,1:7,:]
    leadavf = tensor[:,7,:].unsqueeze(1)

    leadII = (0.5*leadI) + leadavf

    leadIII = -(0.5*leadI) + leadavf
    leadavr = -(0.75*leadI) -(0.5*leadavf)
    leadavl = (0.75*leadI) - (0.5*leadavf)

    leads12 = torch.cat([leadI, leadII, leadschest, leadIII, leadavr, leadavl, leadavf], dim=1)

    return leads12



def generate(model_config,
                diffusion_config,
                diffusion_hyperparams,
                output_directory,
             num_samples,
             ckpt_path,
             data_path,
             ckpt_iter
             ):
    
    
    """
    Generates synthetic ECG data
    """
    
    # --- 1. Setup Environment ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Save directly under the provided output directory (no model/diffusion subdir)
    os.makedirs(output_directory, exist_ok=True)
    
    # Create a unique run subfolder to avoid overwriting
    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_directory = os.path.join(output_directory, f'run_{run_stamp}')
    os.makedirs(run_directory, exist_ok=True)
    print(f"Output directory: {run_directory}", flush=True)

    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].to(device)

    # --- 2. Initialize and Load Model ---
    net = SSSD_ECG(**model_config).to(device)
    print_size(net)

    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    
    model_path = os.path.join(ckpt_path, f'{ckpt_iter}.pkl')
    try:
        if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
            raise FileNotFoundError(f'Checkpoint missing or empty at: {model_path}')
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        try:
            net.load_state_dict(state_dict, strict=True)
            print(f'Successfully loaded model at iteration {ckpt_iter} onto device "{device}" (strict=True)')
        except RuntimeError as e:
            print("Warning: strict load failed due to key mismatch. Retrying with strict=False. Details:\n", e)
            try:
                net.load_state_dict(state_dict, strict=False)
                print('Non-strict load succeeded. Some parameters were not loaded exactly. Proceeding...')
            except Exception as e2:
                raise Exception(f'Error loading model non-strictly: {e2}')
    except FileNotFoundError:
        raise Exception(f'Model checkpoint not found at: {model_path}')
    except Exception as e:
        print("\n" + "="*80)
        print("CRITICAL ERROR: MODEL LOADING FAILED - ARCHITECTURE MISMATCH")
        print("="*80)
        print(f"Error details: {e}")
        print("\nThis error means the model definition in your Python code (SSSD_ECG.py, S4Model.py)")
        print("does NOT match the architecture of the saved checkpoint file. The layers or their")
        print("parameters are different.")
        print("\nTroubleshooting:")
        print("1. Ensure you are using the exact 'SSSD_ECG.py' and 'S4Model.py' files that were")
        print("   used to train and save the model checkpoint.")
        print("2. The error message above might list 'unexpected keys' or 'missing keys'.")
        print("   This tells you exactly which layers are different between your code and the file.")
        print("="*80 + "\n")
        raise Exception(f'Error loading model: {e}')

    # --- 3. Generate Data ---
    try:
        labels = np.load(os.path.join(data_path, 'labels/ptbxl_test_labels.npy'))
        # Select only the requested number of samples
        total = labels.shape[0]
        if num_samples < total:
            idx = np.random.choice(total, size=num_samples, replace=False)
            labels = labels[idx]
        else:
            labels = labels[:num_samples]
    except FileNotFoundError:
        print("Warning: ptbxl_test_labels.npy not found. Generating random labels instead.")
        num_classes = model_config.get("label_embed_classes")
        if not num_classes:
            raise ValueError("Could not determine number of classes from model_config for random label generation.")
        labels = np.random.rand(num_samples, num_classes)

    # Single batch honoring num_samples
    label_batches = [labels]
    
    for i, label_batch in enumerate(label_batches):
        cond = torch.from_numpy(label_batch).to(device).float()
        
        start_time = time.time()

        generated_audio = sampling_label(
            net,
            (len(label_batch), model_config["in_channels"], 1000),
            diffusion_hyperparams,
            cond=cond
        )
        generated_audio12 = generate_four_leads(generated_audio)
        
        end_time = time.time()
        print(f'Generated {len(label_batch)} samples in {end_time - start_time:.2f} seconds')

        # --- 4. Save Output ---
        samples_outfile = os.path.join(run_directory, f'{i}_samples.npy')
        np.save(samples_outfile, generated_audio12.detach().cpu().numpy())
        print(f'Saved samples to {samples_outfile}')
        
        labels_outfile = os.path.join(run_directory, f'{i}_labels.npy')
        np.save(labels_outfile, cond.detach().cpu().numpy())
        print(f'Saved labels to {labels_outfile}')