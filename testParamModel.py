import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import os
import sys
from safetensors.torch import load_model
from mlp_trainer import MLP
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference with a trained model")
    parser.add_argument("image_file", type=str, help="Path to the input image file (PNG or JPG)")
    parser.add_argument("model_path", type=str, help="Path to the trained model file (.pth)")

    args = parser.parse_args()
    return args

args = parse_arguments() 
# Load noisy images
image = Image.open(args.image_file)

# Apply transformations to convert the image to a tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to a consistent size
    transforms.ToTensor(),           # Convert images to PyTorch tensors
])
image_tensor = transform(image)
# Preprocess images
print(image_tensor.size())
input_size = image_tensor.size()
hidden_size = 100
output_size = 10 
# Create an instance of your model
model = MLP(input_size, hidden_size, output_size, "paramOpt_config.yml")

# Load the saved model state
model.load_state_dict(torch.load(args.model_path))
# Pass images through model
image_tensor = torch.stack([image_tensor])
predicted_params = model(image_tensor)

# Adjust parameters
model.adjust_params(predicted_params)

# Generate noisy images
noisy_img = model.add_noise(image_tensor)
imageGauss = transforms.ToPILImage()(noisy_img.squeeze(0))
# Visualize the denoised image
imageGauss.save("param_output.png")
