import torch
from torchvision import transforms
from PIL import Image
from mlp_trainer import MLP
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference with a trained model")
    parser.add_argument("image_file", type=str, help="Path to the input image file (PNG or JPG)")
    parser.add_argument("model_path", type=str, help="Path to the trained model file (.pth)")

    args = parser.parse_args()
    return args


def setup_image(imgFilepath, device):
    # Load noisy images
    image = Image.open(imgFilepath)

    # Apply transformations to convert the image to a tensor
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to a consistent size
        transforms.ToTensor(),           # Convert images to PyTorch tensors
    ])
    image_tensor = transform(image).to(device)
    return image_tensor


def setup_model(image_tensor, device, modelFilepath):
    # Preprocess images
    input_size = image_tensor.size()
    hidden_size = 100
    output_size = 10 
    # Create an instance of your model
    model = MLP(input_size, hidden_size, output_size, "paramOpt_config.yml", device).to(device)
    state_dict = torch.load(modelFilepath)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    return model


def main():
    args = parse_arguments()
    imgFilepath = args.image_file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # setup image 
    image_tensor = setup_image(imgFilepath, device)
    imgName = imgFilepath.split("/")[-1].split(".")[0]
    
    # setup model
    modelFilepath = args.model_path
    model = setup_model(image_tensor, device, modelFilepath)
    modelName = modelFilepath.split("/")[-1].split(".")[0]

    # Pass images through model
    image_tensor = torch.stack([image_tensor])
    predicted_params = model(image_tensor)

    # Adjust parameters
    model.adjust_params(predicted_params, device)

    # Generate noisy images
    noisy_img = model.add_noise(image_tensor)
    imageGauss = transforms.ToPILImage()(noisy_img.squeeze(0))
    # Visualize the denoised image
    saveName = imgName + "_" + modelName + ".png"
    imageGauss.save(saveName)

if __name__ == "__main__":
    main()