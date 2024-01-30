###
# Test iamge
###

import re
from predict import predict
import json
from os import listdir


##########################################################################################

def test_image(image_path, model):
    '''
        Test an image
    :return:
    '''
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    flower_label = re.findall('\d+', image_path)[0]

    top_probs, top_labels = predict(image_path, model)
    prob = top_probs[0].detach().numpy()

    correct_name = cat_to_name[flower_label]
    identified_as = cat_to_name[top_labels[0]]

    return correct_name, identified_as, prob, top_labels


##########################################################################################

def test_all_images(directory_path, model):
    '''
        Test all the images and save the result in a file
    :param model:
    :return: None
    '''

    file_result = "test_all_images_" + model.architecture + ".txt"

    with open(file_result, "w") as fr:

        fr.write("Model architecture " + model.architecture)
        fr.write("####################################")

        all_images_labels = listdir(directory_path)

        for image_label in all_images_labels:

            dir_imgs_label = directory_path + "/" + image_label
            all_imgs = listdir(dir_imgs_label)

            for img in all_imgs:
                image_path = dir_imgs_label + "/" + img

                correct_name, identified_as, prob, top_labels = test_image(image_path, model)

                fr.write(image_path)
                fr.write("[Flower=]" + correct_name + "is identified as " + identified_as + "\n")
                fr.write(f"probs: {prob} \n")
                fr.write(f"top_labels:{top_labels} \n \n")
