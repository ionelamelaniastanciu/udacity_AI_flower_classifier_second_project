###
# Process images
###

from torchvision import transforms, datasets
import torch
from PIL import Image
import torch.utils.data

##########################################################################################

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

##########################################################################################

data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(size=224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                      ])

valid_transforms = transforms.Compose([transforms.RandomResizedCrop(256),
                                       transforms.CenterCrop(size=224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.RandomResizedCrop(256),
                                      transforms.CenterCrop(size=224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


##########################################################################################

def data_load(type, device):
    """
        Creates an iterable dataset to load data from
    :param:
         dataset = the dataset of the data to be processed
         device = the device for the pytorch generator
    :return:
        dataset = the iterable dataset of the data to be processed
    """
    if type == "train":
        train_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
        train_loader = torch.utils.data.DataLoader(train_datasets,
                                                   batch_size=64,
                                                   shuffle=True,
                                                   generator=torch.Generator(device))
        return train_loader
    elif type == "valid":
        valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
        valid_loader = torch.utils.data.DataLoader(valid_datasets,
                                                   batch_size=64,
                                                   shuffle=True,
                                                   generator=torch.Generator(device))
        return valid_loader
    else:
        test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
        test_loader = torch.utils.data.DataLoader(test_datasets,
                                                  batch_size=64,
                                                  shuffle=True,
                                                  generator=torch.Generator(device))
        return test_loader


##########################################################################################

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model
        returns an Numpy array
    '''

    img = Image.open(image)
    img = img.resize((256, 256))

    transform = transforms.Compose([transforms.CenterCrop(size=224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    img = transform(img)

    return img
