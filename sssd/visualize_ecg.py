# AI Generated class to visualize ECG samples

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_ecg(signal, title="ECG Signal", save_path=None):
    """
    Plot a single ECG signal with all leads
    
    Parameters:
    signal: numpy array of shape (12, 1000) containing all 12 leads
    title: title for the plot
    save_path: if provided, save the plot to this path
    """
    # Lead names in standard ECG order
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Create subplots for each lead
    fig, axes = plt.subplots(6, 2, figsize=(15, 20))
    fig.suptitle(title, fontsize=16)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Plot each lead
    for i in range(12):
        ax = axes[i]
        ax.plot(signal[i], linewidth=1)
        ax.set_title(f'Lead {lead_names[i]}')
        ax.grid(True)
        ax.set_xlim(0, 1000)
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_generated_samples(samples_path, labels_path, output_dir, num_samples=5):
    """
    Visualize multiple generated ECG samples
    
    Parameters:
    samples_path: path to the samples.npy file
    labels_path: path to the labels.npy file
    output_dir: directory to save the plots
    num_samples: number of samples to visualize
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving visualizations to: {os.path.abspath(output_dir)}")
    
    # Load the data
    samples = np.load(samples_path)
    labels = np.load(labels_path)
    
    # Ensure we don't try to plot more samples than available
    num_samples = min(num_samples, samples.shape[0])
    
    # Plot each sample
    for i in range(num_samples):
        # Create a title with the label information
        label_info = f"Labels: {labels[i]}"
        title = f"Generated ECG Sample {i+1}\n{label_info}"
        
        # Plot and save
        save_path = os.path.join(output_dir, f"ecg_sample_{i+1}.png")
        plot_ecg(samples[i], title=title, save_path=save_path)
        print(f"Saved plot for sample {i+1} to {os.path.abspath(save_path)}")

if __name__ == "__main__":
    # Path to generated samples
    base_dir = "sssd_label_cond/ch64_T200_betaT0.02"
    output_dir = "visualization_output/ch64_test_samples"
    
    # Print the full path we're looking in
    print(f"Looking for samples in: {os.path.abspath(base_dir)}")
    
    # Visualize samples from each batch
    for i in range(6):
        samples_path = os.path.join(base_dir, f"{i}_samples.npy")
        labels_path = os.path.join(base_dir, f"{i}_labels.npy")
        
        if os.path.exists(samples_path) and os.path.exists(labels_path):
            print(f"\nVisualizing samples from batch {i}")
            batch_output_dir = os.path.join(output_dir, f"batch_{i}")
            visualize_generated_samples(samples_path, labels_path, batch_output_dir)
        else:
            print(f"Skipping batch {i} - files not found at:")
            print(f"  {os.path.abspath(samples_path)}")
            print(f"  {os.path.abspath(labels_path)}") 