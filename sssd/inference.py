# Edited to support generating ecg samples using an untrained model for establishing a baseline
import os
import argparse
import json
import numpy as np
import torch
import time
from utils.util import find_max_epoch, print_size, sampling_label, calc_diffusion_hyperparams
from models.SSSD_ECG import SSSD_ECG
from utils.monitoring import ProgressMonitor


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


def save_checkpoint(samples, labels, batch_idx, subbatch_idx, output_directory):
    """Save intermediate results as checkpoint"""
    checkpoint_dir = os.path.join(output_directory, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f'batch_{batch_idx}_subbatch_{subbatch_idx}.npz')
    np.savez(checkpoint_path, samples=samples, labels=labels)
    return checkpoint_path


def load_checkpoint(batch_idx, subbatch_idx, output_directory):
    """Load checkpoint if it exists"""
    checkpoint_path = os.path.join(output_directory, 'checkpoints', f'batch_{batch_idx}_subbatch_{subbatch_idx}.npz')
    if os.path.exists(checkpoint_path):
        checkpoint = np.load(checkpoint_path)
        return checkpoint['samples'], checkpoint['labels']
    return None, None


def generate(output_directory,
             num_samples,
             ckpt_path,
             ckpt_iter,
             batch_size=100):  # Reduced batch size for better memory management
    
    """
    Generate data based on ground truth with improved memory management and monitoring
    """
    # generate experiment (local) path
    local_path = "ch{}_T{}_betaT{}".format(model_config["res_channels"], 
                                           diffusion_config["T"], 
                                           diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # predefine model
    net = SSSD_ECG(**model_config).cuda()
    print_size(net)
    
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    
    # Handle untrained model (iteration 0)
    if ckpt_iter == 0:
        print('Using untrained model for baseline generation')
    else:
        # load checkpoint
        model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
        print(f"Looking for checkpoint at: {model_path}")
        print(f"Checkpoint exists: {os.path.exists(model_path)}")
        
        if not os.path.exists(model_path):
            raise Exception('No valid model found at iteration {}'.format(ckpt_iter))
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))

    # Load labels
    labels = np.load('data/ptbxl_test_labels.npy')
    label_batches = [
        labels[0:400],
        labels[400:800],
        labels[800:1200],
        labels[1200:1600],
        labels[1600:2000],
        labels[2000:]
    ]
    
    # Initialize progress monitor
    monitor = ProgressMonitor(
        total_steps=diffusion_config["T"],
        batch_size=batch_size,
        num_batches=len(label_batches)
    )
    
    # Process each batch
    for batch_idx, label_batch in enumerate(label_batches):
        print(f"\nProcessing batch {batch_idx + 1}/{len(label_batches)}")
        
        # Split batch into smaller subbatches
        num_subbatches = (num_samples + batch_size - 1) // batch_size
        all_samples = []
        all_labels = []
        
        for subbatch_idx in range(num_subbatches):
            start_idx = subbatch_idx * batch_size
            end_idx = min((subbatch_idx + 1) * batch_size, num_samples)
            current_batch_size = end_idx - start_idx
            
            # Check for existing checkpoint
            samples, labels = load_checkpoint(batch_idx, subbatch_idx, output_directory)
            if samples is not None:
                print(f"\nLoading checkpoint for batch {batch_idx + 1}, subbatch {subbatch_idx + 1}")
                all_samples.append(samples)
                all_labels.append(labels)
                continue
            
            # Generate new samples
            cond = torch.from_numpy(label_batch[start_idx:end_idx]).cuda().float()
            
            # Generate samples with monitoring
            generated_audio = sampling_label(net, (current_batch_size, 8, 1000), 
                                          diffusion_hyperparams,
                                          cond=cond,
                                          monitor=monitor,
                                          current_batch=batch_idx)
            
            generated_audio12 = generate_four_leads(generated_audio)
            
            # Save checkpoint
            save_checkpoint(
                generated_audio12.detach().cpu().numpy(),
                cond.detach().cpu().numpy(),
                batch_idx,
                subbatch_idx,
                output_directory
            )
            
            all_samples.append(generated_audio12.detach().cpu().numpy())
            all_labels.append(cond.detach().cpu().numpy())
            
            # Clear GPU memory
            torch.cuda.empty_cache()
        
        # Combine all subbatches
        samples = np.concatenate(all_samples, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        # Save final results
        outfile = f'{batch_idx}_samples.npy'
        new_out = os.path.join(output_directory, outfile)
        np.save(new_out, samples)
        print(f'\nSaved generated samples for batch {batch_idx + 1}')
        
        outfile = f'{batch_idx}_labels.npy'
        new_out = os.path.join(output_directory, outfile)
        np.save(new_out, labels)
        print(f'Saved labels for batch {batch_idx + 1}')
        
        # Clear memory
        del all_samples, all_labels
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config_SSSD_ECG.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', type=str, default='100000',
                        help='Which checkpoint to use; assign a number or "max"')
    parser.add_argument('-n', '--num_samples', type=int, default=400,
                        help='Number of utterances to be generated')
    parser.add_argument('-b', '--batch_size', type=int, default=100,
                        help='Batch size for generation (default: 100)')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    print(config)

    gen_config = config['gen_config']
    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    model_config = config['wavenet_config']

    # Convert ckpt_iter to int if it's not 'max'
    if args.ckpt_iter != 'max':
        args.ckpt_iter = int(args.ckpt_iter)

    # Get absolute path to checkpoint directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(current_dir, 'exp', 'ch64_T50_betaT0.02')
    print(f"Using checkpoint directory: {checkpoint_dir}")
    print(f"Directory exists: {os.path.exists(checkpoint_dir)}")
    
    # Override the checkpoint path in gen_config
    gen_config['ckpt_path'] = checkpoint_dir

    # Modify model config to match checkpoint architecture
    model_config['res_channels'] = 64
    model_config['skip_channels'] = 64
    model_config['diffusion_step_embed_dim_in'] = 128
    model_config['diffusion_step_embed_dim_mid'] = 512
    model_config['diffusion_step_embed_dim_out'] = 512
    model_config['label_embed_dim'] = 128

    print("Modified model configuration to match checkpoint:")
    print(json.dumps(model_config, indent=2))

    generate(**gen_config,
             ckpt_iter=args.ckpt_iter,
             num_samples=args.num_samples,
             batch_size=args.batch_size)

