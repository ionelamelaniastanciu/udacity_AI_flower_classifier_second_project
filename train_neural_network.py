###
# Train function for the epochs
###

from torch import nn
import time
from torch import optim
import torch
from collections import OrderedDict


##########################################################################################

def train_neural_network(train_loader,
                         valid_loader,
                         model,
                         features1=256,
                         features2=128,
                         mode_file="a"):
    """
        Train the neural network
    :param:
        train_loader = training images
        valid_loader = validating images
        epochs = number of epochs
        device = by default is cpu
    :return:
        model = the trained neural network after epochs
        total_time = time in milliseconds for training the neural network
    """
    epochs = model.epochs
    file_output = "train_results_" + model.architecture + ".txt"

    with open(file_output, mode_file) as train_results_file:
        train_results_file.write("####################################\n")
        train_results_file.write("Training neural network on " + model.device_name)
        train_results_file.write("\nModel architecture " + model.architecture + "\n")
        train_results_file.write(f"Features: {features1} {features2} \n")
        train_results_file.write(f"Total epochs : {epochs} \n")
        train_results_file.write(f"Learning rate {model.learning_rate}")
        train_results_file.write("\n####################################\n")

        # Definition of the Classifier
    model.classifier = nn.Sequential(OrderedDict([
        ("linear1", nn.Linear(model.input_size, features1, bias=True)),  # VGG16 7x7x512 = 25088
        ("dropout1", nn.Dropout(0.3)),
        ("relu1", nn.ReLU()),
        ("linear2", nn.Linear(features1, features2, bias=True)),
        ("dropout2", nn.Dropout(0.2)),
        ("relu2", nn.ReLU()),
        ("output", nn.Linear(features2, 102, bias=True)),  # we have 102 types of flowers
        ("logsoftmax", nn.LogSoftmax(dim=1))
    ]))

    train_losses = []
    validation_losses = []
    steps = 0
    print_every = 20

    device = model.device

    criterion = nn.NLLLoss()
    model.criterion = criterion

    optimizer = optim.Adam(model.classifier.parameters(), lr=model.learning_rate)
    model.optimizer = optimizer

    start = time.time()

    for epoch in range(epochs):

        # Setting to training mode
        running_loss = 0.0
        model.train()

        for inputs_t, labels_t in train_loader:

            steps += 1

            # move inputs and labels to the device
            inputs = inputs_t.to(device)
            labels = labels_t.to(device)

            # call the neural network
            output = model.forward(inputs)

            # Apply criterion
            loss = criterion(output, labels)

            # Apply Backpropagation
            optimizer.zero_grad(set_to_none=True)

            # loss.requires_grad = True
            loss.backward()

            # call Adam to propagate the computations
            optimizer.step()

            # compute running_loss
            running_loss += loss.item()

            # after every batch we check the validation accuracy
            if steps % print_every == 0:

                # Setting to validation mode
                model.eval()
                valid_loss = 0.0
                accuracy = 0.0

                with torch.no_grad():  # we don't want to use gradient to compute

                    for images_, labels_ in valid_loader:
                        images_valid = images_.to(device)
                        labels_valid = labels_.to(device)

                        valid_output = model.forward(images_valid)

                        validating_loss = criterion(valid_output, labels_valid)
                        valid_loss += validating_loss.item()

                        exp_output = torch.exp(valid_output)

                        top_p, top_class = exp_output.topk(1, dim=1)

                        validator = (top_class == labels_valid.view(*top_class.shape))

                        accuracy += torch.mean(validator.type(torch.FloatTensor)).item()

                training_loss = running_loss / print_every
                train_losses.append(training_loss)

                validation_loss = valid_loss / len(valid_loader)
                validation_losses.append(validation_loss)

                validation_acc = accuracy / len(valid_loader)

                with open(file_output, "a") as train_results_file:
                    train_results_file.write(f"Epoch: {epoch + 1}/{epochs}")
                    train_results_file.write(f"Training Loss: {training_loss:.3f} | ")
                    train_results_file.write(f"Validation Loss: {validation_loss:.3f} | ")
                    train_results_file.write(f"Accuracy: {validation_acc:.3f}\n")

                # Setting to training mode
                running_loss = 0.0
                model.train()

    total_time = time.time() - start

    return model, total_time, validation_losses, train_losses
