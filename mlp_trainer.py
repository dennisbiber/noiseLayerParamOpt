import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import os
from noise_layer import AddNoise
from safetensors.torch import save_file
import sys
import yaml
import csv
from datetime import datetime
import argparse

# Define the multilayer perceptron architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, configPath):
        super(MLP, self).__init__()
        self.configPath = configPath
        flattened_size = input_size[0] * input_size[1] * input_size[2]
        self.fc1 = nn.Linear(flattened_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.add_noise = AddNoise(
            threshold=0.06, exponent=0.0, slope=1.0, 
            intercept=7.0, noise_type='gaussian', mean=0.1, std=0.1,
            grid_size=(36, 27), circle_size_factor=0.24, 
            heightSkew=0.25, widthSkew=0.4, inversionBool=False
        )

    def forward(self, x):
        # Flatten the input tensor if needed
        x = x.view(x.size(0), -1)
        # Pass through the first fully connected layer
        x = torch.relu(self.fc1(x))
        
        # Pass through the second fully connected layer to generate parameters
        parameters = self.fc2(x)
        
        return parameters
    
    def load_params(self):
        yaml_data = read_yaml_file(self.configPath)
        params = yaml_data.get("params")
        return params
    
    def get_current_datetime_string(self):
        now = datetime.now()
        datetime_string = now.strftime("%Y-%m-%d_%H-%M")
        return datetime_string

    def update_csv(self, filename="csv_test.csv"):
        vars = self.load_params()
        append_to_csv(vars, self.get_current_datetime_string, filename)

    def adjust_params(self, predictions):
        vars = self.load_params()
        self.add_noise.threshold = predictions[:, 0].clamp(vars["thresh"]["low"], vars["thresh"]["high"]) # Adjust threshold between +0.2 and 1
        self.add_noise.exponent = predictions[:, 1].clamp(vars["exp"]["low"], vars["exp"]["high"]) # Adjust exponent between 0.1 and 3
        self.add_noise.slope = predictions[:, 2].clamp(vars["slope"]["low"], vars["slope"]["high"]) # Adjust slope between -2 and 2
        self.add_noise.mean = predictions[:, 3].clamp(vars["mean"]["low"], vars["mean"]["high"]) # Adjust mean between +0.2 and 1
        self.add_noise.std = predictions[:, 4].clamp(vars["std"]["low"], vars["std"]["high"]) # Adjust std between +0.2 and 1
        # Adjust grid_size between (2, 2) and (8, 8)
        self.add_noise.grid_size = (int(predictions[:, 5].clamp(vars["grid"]["hY"]["low"], vars["grid"]["hY"]["high"]) ** 2), 
                                    int(predictions[:, 6].clamp(vars["grid"]["wX"]["low"], vars["grid"]["wX"]["high"]) ** 2))
        self.add_noise.circle_size_factor = predictions[:, 7].clamp(vars["thresh"]["low"], vars["thresh"]["high"])  # Adjust circle_size_factor between 0 and 0.8
        self.add_noise.hSkew = predictions[:, 8].clamp(vars["hskew"]["low"], vars["hskew"]["high"])        # Adjust heightSkew between 0.2 and 0.8
        self.add_noise.wSkew = predictions[:, 9].clamp(vars["wskew"]["low"], vars["wskew"]["high"])         # Adjust widthSkew between 0.2 and 0.8
        self.add_noise.invert = vars["invert"]
    

def train_model(model, train_loader, num_epochs=4, learning_rate=0.00025):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for img, params in train_loader:
            optimizer.zero_grad()
            
            # Forward pass to get predictions for parameter adjustment
            predictions = model(img)
            img = img.squeeze(1)
            # Adjust parameters based on predictions
            model.adjust_params(predictions)
            # img = img.unsqueeze(0)
            # Add noise to the clean images
            noisy_images = model.add_noise(img)

            # Calculate loss between clean images and noisy images
            loss = criterion(noisy_images, img)

            # # Apply noise layer to each noisy image in the batch
            # noisy_images = [model.add_noise(image) for image in images]
            # noisy_images = torch.stack(noisy_images)  # Convert list of noisy images to tensor

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * noisy_images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # torch.save(model.state_dict(), "denoiser_model.pth")
    model.update_csv()
    filename = model.get_current_datetime_string()
    torch.save(model.state_dict(), f"para_models/{filename}.pth")
    # save_model_as_safetensor(model, "paramModel(0.4-0.6Thresh).safetensors")


def save_model_as_safetensor(model, filename):
    model_state_dict = model.state_dict()
    save_file(model_state_dict, filename)

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def json2csv(conf, retunrGrid=False):
    keys = conf.keys()
    l1 = [(conf[k]["low"], conf[k]["high"]) for k in keys if k != "grid" and k != "invert"]
    grid = [(conf[k]["hY"]["low"], conf[k]["hY"]["high"], conf[k]["wX"]["low"], conf[k]["wX"]["high"]) for k in keys if k == "grid"]
    l = l1 + grid
    ks = [k for k in keys if k != "grid"]
    return l, ks

def writeRow(writer, row):
    writer.writerow(row)

def append_to_csv(conf, model_name, csv_file):
    l, ks = json2csv(conf)
    if not os.path.isfile(csv_file):
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writeRow(writer, (ks + ["model_name"]))
            writeRow(writer, (l + [model_name]))
    else:
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writeRow(writer, (l + [model_name]))

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
    parser.add_argument("image_file", type=str, help="Path to the input image directory of (PNG or JPG)")
    parser.add_argument("test", type=str, help="Path to the trained model file (.pth)")

    args = parser.parse_args()
    return args


# Example of how to use the denoising neural network with your custom noise layer
def main():
    args = parse_arguments()
    # Load your dataset and create clean images
    # directory = "/home/dev/code/data/training_data"
    directory = args.image_file
    train_images_tensor = load_images_from_directory(directory)
    # model.noise_layer.circle_size_factor = 1 - (epoch*0.08)
    input_size = train_images_tensor.size()[1:]
    hidden_size = 100
    output_size = 10  # Number of parameters to predict: threshold, mean, std, noise_type
    print("Input size:", input_size)
    # Initialize the denoising neural network print(type(images))with your custom noise layer
    model = MLP(input_size, hidden_size, output_size, "paramOpt_config.yml")
    if args.test.lower() == "true":
        model.update_csv(filename="test.csv")
        sys.exit("Test Complete. Check csv File")
    train_dataset = TensorDataset(train_images_tensor, train_images_tensor)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    train_model(model, train_loader)
    model.eval()

if __name__ == "__main__":
    main()