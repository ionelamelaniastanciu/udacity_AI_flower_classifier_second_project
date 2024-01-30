###
#  Checkpoint saver
###
import torch
from torchvision import models
from torchvision.models import VGG16_Weights, AlexNet_Weights

def save_model(checkpoint_path, model):
    """
        Save model to the checkpoint path file
    :param:
        checkpoint_path = checkpoint path
        model = neural network model
    :return:
        None
    """
    checkpoint_dict = {'input_size': model.input_size,
                       'output_size': 102,
                       'arch': model.architecture,
                       'features': model.features,
                       'avgpool': model.avgpool,
                       'classifier': model.classifier,
                       'state_dict': model.state_dict(),
                       'class_to_idx': model.class_to_idx}

    torch.save(checkpoint_dict, checkpoint_path)

##########################################################################################

def load_model(checkpoint_path):
    """
          Loads the checkpoint given as a string parameter
    :param:
        checkpoint_path = checkpoint path
    :return:
        model_ = the loaded model from the checkpoint path
    """
    checkpoint = torch.load(checkpoint_path)

    if checkpoint['arch'] == "alexnet":
        model_ = models.alexnet(weights=AlexNet_Weights.DEFAULT)
    else:
        model_ = models.vgg16(weights=VGG16_Weights.DEFAULT)

    model_.class_to_idx = checkpoint['class_to_idx']
    model_.load_state_dict(checkpoint['state_dict'], strict=False)
    model_.classifier = checkpoint['classifier']
    model_.architecture = checkpoint['arch']
    # optimizer.load_state_dict(checkpoint['optimizer'])

    return model_  # , optimizer
