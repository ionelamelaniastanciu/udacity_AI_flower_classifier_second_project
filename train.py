###
# Main
###

import torch
from files_transformer import data_load
from train_neural_network import train_neural_network
from test import test_neural_network
from checkpoint_saver import save_model
from torchvision import models
from torchvision.models import VGG16_Weights, AlexNet_Weights
from checkpoint_saver import load_model
from get_input_args import get_input_args
from test_image import *

def main():
    """
        Main programm
    """
    input_args = get_input_args()
    print(input_args)

    gpu = input_args.__getattribute__("gpu")
    device = torch.device("cuda" if torch.cuda.is_available() and gpu == "cuda" else "cpu")
    torch.set_default_device(device)

    arch = input_args.__getattribute__("arch")

    if arch == "alexnet":
        model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        model.epochs = 10
        model.architecture = arch
        model.input_size = 9216
    else:
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        model.architecture = "vgg16"
        model.input_size = 25088
        model.epochs = 7

    lr = input_args.__getattribute__("learning_rate")
    model.learning_rate = lr

    # Images processing
    train_loader = data_load("train", device)
    valid_loader = data_load("valid", device)

    model.class_to_idx = train_loader.dataset.class_to_idx

    model.device = device
    model.device_name = gpu
    model.to(device)

    # Training neural network
    train_neural_network(train_loader, valid_loader, model)

    # Save the checkpoint
    checkpoint_file = "checkpoint_" + model.architecture + "_command_line_" + ".pth"
    save_model(checkpoint_file, model)

    # Testing neural network
    test_loss, accuracy = test_neural_network(model)
    print(f"test_loss {test_loss}")
    print(f"accuracy {accuracy}")

    # Load the checkpoint
    model_load = load_model(checkpoint_file)

    # Test an image
    image_path = input_args.__getattribute__("flower_file")
    correct_name, identified_as, prob, top_labels = test_image(image_path, model_load)

    print("[Flower=]" + correct_name + "is identified as " + identified_as)
    print(f"probs: {prob}")
    print(f"top_labels:{top_labels}")

    # Test all images
    directory_path = input_args.__getattribute__("dir")
    test_all_images(directory_path, model_load)

# Call to main function to run the program
if __name__ == "__main__":
    main()
