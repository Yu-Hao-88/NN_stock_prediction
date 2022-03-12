from utils.utils import pickleStore, readData
from preprocessing.preprocessing import preprocess, transform_dataset, train_test_split
from dataset.dataset import Dataset
from model.model import LSTMPredictor
from trainer.supervised import trainer, tester
from mytensorboard.tensorboard import TensorBoard

import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


"""
# Project#1 Stock prediction

2022/3/2 Neural Network

For your references:

*   [Pytorch official website](https://pytorch.org/)

*   [Google Colab official tutorial](https://colab.research.google.com/notebooks/welcome.ipynb?hl=zh-tw#scrollTo=gJr_9dXGpJ05)

*   [Using outer files in Google colab](https://colab.research.google.com/notebooks/io.ipynb#scrollTo=BaCkyg5CV5jF)

"""

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

if __name__:
    # Device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Data
    data = readData("./data/1795_history.csv")
    print('Num of samples:', len(data))

    # Preprocess
    prices = preprocess(data)
    # Divide trainset and test set
    train, test = train_test_split(prices, 0.8)
    # Set the N(look_back)=5
    look_back = 5
    trainX, trainY = transform_dataset(train, look_back)
    testX, testY = transform_dataset(test, look_back)
    # Get dataset
    trainset = Dataset(trainX, trainY)
    testset = Dataset(testX, testY)
    # Get dataloader
    batch_size = 100
    # num_workers should set 1 if put data on CUDA
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
        # num_workers=1
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_cuda,
        # num_workers=1
    )

    # Model
    net = LSTMPredictor(look_back)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.002)

    # Training
    retrain = False
    model_name = 'init'
    checkpoint = f"./checkpoint/{model_name}.pt"
    if not os.path.isfile(checkpoint) or retrain:
        tensorboard = TensorBoard()
        tensorboard.init_tensorboard_writers(model_name)

        trainer(device, net, criterion, optimizer, trainloader,
                testloader, tensorboard, epoch_n=100, path=checkpoint)

        tensorboard.close_tensorboard_writers()
    else:
        net.load_state_dict(torch.load(checkpoint))

    # Test the model
    test = tester(device, net, criterion, testloader)
    # Show the difference between predict and groundtruth (loss)
    print('Test Loss Result: ', test)

    # Predict
    predict = net.predict(torch.tensor(
        [[126, 124, 124, 122.5, 121]], dtype=torch.float32, device=device))
    print('Predict Result', predict)
