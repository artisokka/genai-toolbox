# AI Generated class to monitor the progress of the model
import torch
import time
from tqdm import tqdm
import psutil
import os
import datetime

class ProgressMonitor:
    def __init__(self, total_steps, batch_size, num_batches):
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.current_step = total_steps - 1  # Start at T-1 since we count down
        self.current_batch = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.steps_per_second = 0
        self.processing_speeds = []  # Store recent processing speeds
        self.speed_window = 10  # Number of steps to average for speed calculation
        
    def get_gpu_memory_usage(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        return 0
    
    def get_system_memory_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2  # Convert to MB
    
    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def calculate_eta(self, current_step, current_batch, elapsed_time):
        # Calculate steps completed (remember we're counting down)
        steps_completed = (current_batch * self.total_steps) + (self.total_steps - current_step - 1)
        if steps_completed > 0:
            time_per_step = elapsed_time / steps_completed
        else:
            time_per_step = 0
            
        # Calculate remaining steps
        remaining_steps = (self.total_steps * (self.num_batches - current_batch - 1)) + current_step + 1
        
        # Calculate ETA
        if time_per_step > 0:
            eta = remaining_steps * time_per_step
            return eta
            
        return 0
    
    def update(self, step, current_batch):
        self.current_step = step
        self.current_batch = current_batch
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        self.last_update_time = current_time

        # Calculate progress across all batches (remember we're counting down)
        steps_completed = (current_batch * self.total_steps) + (self.total_steps - step - 1)
        total_steps = self.total_steps * self.num_batches
        total_progress = (steps_completed / total_steps) * 100
        
        # Calculate speed and ETA
        if elapsed_time > 0:
            self.steps_per_second = steps_completed / elapsed_time
            eta = self.calculate_eta(step, current_batch, elapsed_time)
        else:
            self.steps_per_second = 0
            eta = 0

        # Format time strings
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        eta_str = str(datetime.timedelta(seconds=int(eta))) if eta > 0 else "Calculating..."

        # Get GPU memory info
        gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        system_memory = psutil.Process().memory_info().rss / 1024**2

        # Print progress
        print(f"\rBatch {current_batch + 1}/{self.num_batches} | Step {step}/{self.total_steps} | "
              f"Progress: {total_progress:.1f}% | Time: {elapsed_str} | ETA: {eta_str} | "
              f"GPU Memory: {gpu_memory:.1f}MB | System Memory: {system_memory:.1f}MB", end="")

    def new_batch(self):
        print("\n")  # New line after each batch
        self.current_step = self.total_steps - 1  # Reset to T-1 for new batch
        self.last_update_time = time.time()
        self.processing_speeds = []  # Reset processing speeds for new batch
        self.speed_window = 10  # Reset speed window for new batch 