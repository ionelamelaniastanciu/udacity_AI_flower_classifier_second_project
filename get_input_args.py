###
# Get input arguments
###

import argparse

def get_input_args():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's
    argparse module to created and defined these 3 command line arguments. If
    the user fails to provide some or all of the 3 arguments, then the default
    values are used for the missing arguments.
    Command Line Arguments:
      1. Image Folder as --dir with default value 'flowers'
      2. CNN Model Architecture as --arch with default value 'vgg'
      3. Text File with flower names as --flower_file with default value 'cat_to_name.json'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", type=str, default="flowers/test", help="path to folder of images")
    parser.add_argument("--arch", default="vgg16")
    parser.add_argument("--flower_file", default="flowers/test/9/image_06413.jpg")
    parser.add_argument("--learning_rate", default=0.001)
    parser.add_argument("--hidden_units1", default=256)
    parser.add_argument("--hidden_units2", default=128)

    parser.add_argument("--gpu", default="cuda")

    return parser.parse_args()
