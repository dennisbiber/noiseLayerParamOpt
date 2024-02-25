from env_setup import DistributedEnvironmentSetup
from mlp_trainer import train_model, MLP
from PIL import Image
from torchvision import transforms
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import os
import torch
import sys

_parallelProcess = True

def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference with a trained model")
    parser.add_argument("image_directory", type=str, help="Path to the input image directory of (PNG or JPG)")
    parser.add_argument("test", type=str, help="Path to the trained model file (.pth)")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use for distributed training")

    args = parser.parse_args()
    return args


def load_images_from_directory(directory, device):
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
            image_tensor = transform(image).to(device)
            
            images.append(image_tensor)

    return torch.stack(images)


def setup_model(train_images_tensor, device, rank):
    input_size = train_images_tensor.size()[1:]
    hidden_size = 100
    output_size = 10  # Number of parameters to predict: threshold, mean, std, noise_type
    model = MLP(input_size, hidden_size, output_size, "paramOpt_config.yml", device=device)
    model.to(device)
    if _parallelProcess:
        model = DDP(model, [rank])
    return model


def main(rank, world_size, batch_size):
    print("Starting Main")
    if _parallelProcess:
        # Initialize distributed environment
        environment_setup = DistributedEnvironmentSetup(world_size, rank)
        environment_setup.initialize_process_group()
    args = parse_arguments()

    if not torch.cuda.is_available():
        sys.exit("CUDA device not available.")
    device = torch.device("cuda")

    # Load dataset
    train_images_tensor = load_images_from_directory(args.image_directory, device)
    model = setup_model(train_images_tensor, device, rank)
    if args.test.lower() == "true":
        model.update_csv(filename="test.csv")
        sys.exit("Test Complete. Check csv File")
    
    # Set up data loader
    train_dataset = TensorDataset(train_images_tensor, train_images_tensor)
    if _parallelProcess:
        train_loader = environment_setup.setup_distributed_dataloader(train_dataset, batch_size)
    else:
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Train the model
    train_model(model, train_loader, _parallelProcess, rank, num_epochs=4, learning_rate=0.0001, device=device)
    if _parallelProcess:
        environment_setup.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    batch_size = 1
    mp.spawn(main, args=(world_size, batch_size,), nprocs=world_size)
