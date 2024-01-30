###
# Predict function for the first 5 probabilities
###

import torch
from files_transformer import process_image

##########################################################################################

def predict(image_path, model_, topk=5):
    '''
        Predict the class (or classes) of an image using a trained deep learning neural network
        returns:
            topk_probs = the first highest topk probabilities
            topk_labels = the first highest topk labels
    '''
    model_.to("cpu")
    model_.eval()

    processed_image = process_image(image_path)
    processed_image.unsqueeze_(0)

    with torch.no_grad():
        output = model_.forward(processed_image)
        probs = torch.exp(output)
        topk_probs, top_labels = probs.topk(topk)

        # Convert indices to classes
        idx_to_class = {val: key for key, val in model_.class_to_idx.items()}
        topk_labels = [idx_to_class[lab] for lab in top_labels.tolist()[0]]

        return topk_probs, topk_labels
