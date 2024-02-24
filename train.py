from env_setup import DistributedEnvironmentSetup
from mlp_trainer import train_model, MLP
from PIL import Image
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import torch
import sys

def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load the image
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            
            # Apply transformations to convert the image to a tensor
            transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Resize images to a consistent size
                transforms.ToTensor(),           # Convert images to PyTorch tensors
            ])
            image_tensor = transform(image)
            
            images.append(image_tensor)

    return torch.stack(images)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference with a trained model")
    parser.add_argument("image_directory", type=str, help="Path to the input image directory of (PNG or JPG)")
    parser.add_argument("test", type=str, help="Path to the trained model file (.pth)")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use for distributed training")

    args = parser.parse_args()
    return args

def setup_model(train_images_tensor, device):
    input_size = train_images_tensor.size()[1:]
    hidden_size = 100
    output_size = 10  # Number of parameters to predict: threshold, mean, std, noise_type
    model = MLP(input_size, hidden_size, output_size, "paramOpt_config.yml", device=device)
    model.to(device)
    return model

def main():
    verbose = True
    args = parse_arguments()
    if not torch.cuda.is_available():
        sys.exit("CUDA device not available.")
    if verbose:
        print("Args loaded and CUDA device found.")
    device = torch.device("cuda")

    # # Initialize distributed environment
    # environment_setup = DistributedEnvironmentSetup(num_gpus=args.num_gpus)
    # if verbose:
    #     print("Env setup.")
    # environment_setup.initialize_process_group()
    # if verbose:
    #     print("Env setup Initialized.")

    # Load dataset
    train_images_tensor = load_images_from_directory(args.image_directory)
    model = setup_model(train_images_tensor, device)
    if args.test.lower() == "true":
        model.update_csv(filename="test.csv")
        sys.exit("Test Complete. Check csv File")
    train_images_tensor = train_images_tensor.to(device)
    train_dataset = TensorDataset(train_images_tensor, train_images_tensor)
    # Set up distributed data loader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Train the model
    train_model(model, train_loader, num_epochs=4, learning_rate=0.0001, device=device)

if __name__ == "__main__":
    main()
