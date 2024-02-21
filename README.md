csv_test.csv: 
    - csv file of the training parameters associated with the name the model was saved as
text.csv: 
    - csv file for testing the model saving mechanisms to avoid training and failing on saving
noise_layer.py:
    - import: from noise_layer import AddNoise
    - torch.nn.Module for adding noise, and removing, to a data source based on 11 parameters
mlp_trainer.py:
    - training pipeline for getting the AddNoise parameters optimzer
paramOpt_config.yml:
    - Configuration for mlp_trainer for AddNoise parameters ranges and static variables
testParamModel.py:
    - Takes an image and a model from mlp_trainer to demonstrate a single noise addition of the provided model

/para_models/ - model save directory


    The mlp_trainer script data-directory is a directory of images as .jpg or .png. It will only take 
file extensions of .jpg or .png, so if the directory has other files, like annotations, it will ignore them.

mlp_trainer args:
    mlp_trainer.py /path/to/data-directory {true|false}

To Test the mlp_trainer save mechanisms
    python mlp_trainer.py /path/to/data-directory true

To train the mlp_trainer model:
    python mlp_trainer.py /path/to/data-directory false


    The testParamModel script take the image_file as a .jpg or .png that you want to apply the model to. The second
argument is the path to the model. The mlp_trainer saves the models as a .pth extension.

testParamModel args:
    python testParamModel.py /path/to/image_file /path/to/model_file.pth