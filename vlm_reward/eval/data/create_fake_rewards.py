"""
Create fake rewards with the expected length and format for the given gifs
"""

import os
import torch
from PIL import Image

# Define the directory containing the GIFs
gif_directory = 'anne_tests/gifs/'
rewards_directory = 'anne_tests/rewards/'

# Iterate through the files in the directory
for filename in os.listdir(gif_directory):
    if filename.endswith('.gif'):
        gif_path = os.path.join(gif_directory, filename)
        
        # Open the gif using PIL's Image module
        gif = Image.open(gif_path)
        
        # Get the number of frames in the gif
        num_frames = gif.n_frames
        
        # Create a tensor of random positive numbers for each frame
        tensor = torch.rand(num_frames)
        
        # Save the tensor to a .pt file with the same name as the gif
        tensor_filename = os.path.splitext(filename)[0] + '.pt'
        tensor_save_path = os.path.join(rewards_directory, tensor_filename)
        torch.save(tensor, tensor_save_path)
        
        print(f"Saved tensor for {filename} to {tensor_save_path}")