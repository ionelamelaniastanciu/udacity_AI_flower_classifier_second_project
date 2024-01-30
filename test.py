import torch
from files_transformer import data_load

##########################################################################################

def test_neural_network(model):
     """
        Test the neural network
     :param:
            model = the neural network model object
     :return:
            test_loss = loss of the test data
            accuracy = accuracy of the test data
     """
     test_loader = data_load("test", model.device)
     with torch.no_grad(): # we don't want to use gradient to compute

        model.eval()
        testing_loss = 0.0
        accuracy = 0

        for images_, labels_ in test_loader:

            images = images_.to(model.device)
            labels = labels_.to(model.device)

            output = model(images)

            test_loss = model.criterion(output, labels)
            exp_output = torch.exp(output)
            top_p, top_class = exp_output.topk(1, dim=1)
            validator = top_class == labels.view(*top_class.shape)

            accuracy += torch.mean(validator.type(torch.FloatTensor))
            testing_loss += test_loss.item()

        test_loss = testing_loss / len(test_loader)
        accuracy = accuracy.item() / len(test_loader)

        return test_loss, accuracy
