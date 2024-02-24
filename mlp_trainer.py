import torch
import torch.nn as nn
import torch.optim as optim
from utils import read_yaml_file, append_to_csv
from noise_layer import AddNoise
from datetime import datetime

# Define the multilayer perceptron architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, configPath, device):
        super(MLP, self).__init__()
        self.configPath = configPath
        flattened_size = input_size[0] * input_size[1] * input_size[2]
        self.fc1 = nn.Linear(flattened_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.add_noise = AddNoise(
            threshold=0.06, exponent=0.0, slope=1.0, 
            intercept=7.0, noise_type='gaussian', mean=0.1, std=0.1,
            grid_size=(36, 27), circle_size_factor=0.24, 
            heightSkew=0.25, widthSkew=0.4, inversionBool=False,
            device=device
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

    def adjust_params(self, predictions, device=torch.device('cpu')):
        vars = self.load_params()
        # Move threshold and noise parameters to the specified device
        self.add_noise.threshold = predictions[:, 0].clamp(vars["thresh"]["low"], vars["thresh"]["high"]).to(device)
        self.add_noise.exponent = predictions[:, 1].clamp(vars["exp"]["low"], vars["exp"]["high"]).to(device)
        self.add_noise.slope = predictions[:, 2].clamp(vars["slope"]["low"], vars["slope"]["high"]).to(device)
        self.add_noise.mean = predictions[:, 3].clamp(vars["mean"]["low"], vars["mean"]["high"]).to(device)
        self.add_noise.std = predictions[:, 4].clamp(vars["std"]["low"], vars["std"]["high"]).to(device)
        # Adjust grid_size between (2, 2) and (8, 8)
        self.add_noise.grid_size = (
            int(predictions[:, 5].clamp(vars["grid"]["hY"]["low"], vars["grid"]["hY"]["high"]) ** 2),
            int(predictions[:, 6].clamp(vars["grid"]["wX"]["low"], vars["grid"]["wX"]["high"]) ** 2)
        )
        self.add_noise.circle_size_factor = predictions[:, 7].clamp(vars["thresh"]["low"], vars["thresh"]["high"]).to(device)
        self.add_noise.hSkew = predictions[:, 8].clamp(vars["hskew"]["low"], vars["hskew"]["high"]).to(device)
        self.add_noise.wSkew = predictions[:, 9].clamp(vars["wskew"]["low"], vars["wskew"]["high"]).to(device)
        self.add_noise.invert = vars["invert"]
    

def train_model(model, train_loader, num_epochs=4, learning_rate=0.00025, device=torch.device('cpu')):
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
            model.adjust_params(predictions, device)
            noisy_images = model.add_noise(img)

            # Calculate loss between clean images and noisy images
            loss = criterion(noisy_images, img)
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
