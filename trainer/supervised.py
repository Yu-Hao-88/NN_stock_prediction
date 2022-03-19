from utils.utils import saveModel
import numpy as np
import torch

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')


def trainer(device, net, criterion, optimizer, trainloader, devloader, tensorboard, epoch_n=100, path="./checkpoint/save.pt"):
    best_valid_loss = float("inf")
    net = net.to(device)
    train_loss_list = []
    valid_loss_list = []

    for epoch in range(epoch_n):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        train_loss = 0.0
        valid_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, data_index = data
            inputs = inputs.unsqueeze(1).to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            # print statistics
            # running_loss += loss.item()
            # if i % 10 == 9:    # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0

        ######################
        # validate the model #
        ######################
        net.eval()
        for i, data in enumerate(devloader, 0):
            # move tensors to GPU if CUDA is available
            inputs, labels, data_index = data
            inputs = inputs.unsqueeze(1).to(device)
            labels = labels.to(device)

            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = net(inputs)

            # calculate the batch loss
            loss = criterion(outputs, labels)
            # update average validation loss
            valid_loss += loss.item()

        # calculate average losses
        train_loss = train_loss/len(trainloader)
        valid_loss = valid_loss/len(devloader)

        # print training/validation statistics
        print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            train_loss, valid_loss))
        tensorboard.log_on_tensorboard('train', epoch, train_loss)
        tensorboard.log_on_tensorboard('valid', epoch, valid_loss)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # Save model
            saveModel(net, path)

    print('Finished Training')

    return train_loss_list, valid_loss_list


def tester(device, net, criterion, testloader):
    loss = 0
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        for data in testloader:
            inputs, labels, data_index = data
            inputs = inputs.unsqueeze(1).to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss += criterion(outputs, labels)

    return (loss/len(testloader)).item()
