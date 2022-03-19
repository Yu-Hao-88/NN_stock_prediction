from utils.utils import pickleStore, readData
from preprocessing.preprocessing import preprocess, transform_dataset, train_test_split
from dataset.dataset import Dataset
from model.model import LSTMPredictor, GRUModel
from trainer.supervised import trainer, tester
from mytensorboard.tensorboard import TensorBoard
import matplotlib.pyplot as plt

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
    target_days = 5
    trainX, trainY = transform_dataset(train, look_back, target_days)
    testX, testY = transform_dataset(test, look_back, target_days)
    # Get dataset
    trainset = Dataset(trainX, trainY)
    testset = Dataset(testX, testY)
    # Get dataloader
    batch_size = 128
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
    # net = LSTMPredictor(look_back, target_days)
    net = GRUModel(
        input_dim=look_back*5,
        hidden_dim=64,
        layer_dim=2,
        output_dim=target_days,
        dropout_prob=0.2,
        device=device
    )

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.002)

    # Training
    retrain = True
    model_name = 'GRU_baseline'
    checkpoint = f"./checkpoint/{model_name}.pt"
    if not os.path.isfile(checkpoint) or retrain:
        epochs = 500
        tensorboard = TensorBoard()
        tensorboard.init_tensorboard_writers(model_name)

        train_loss, valid_loss = trainer(
            device, net, criterion, optimizer, trainloader,
            testloader, tensorboard, epoch_n=epochs, path=checkpoint)

        tensorboard.close_tensorboard_writers()

        plt.plot(list(range(epochs)), train_loss, '-b', label='train loss')
        plt.plot(list(range(epochs)), valid_loss, '-r', label='valid loss')

        plt.xlabel("n iteration")
        plt.legend(loc='upper right')
        plt.title(model_name)

        # save image
        # should before show method
        plt.savefig(f"./data/img/{model_name}.png")
        # plt.show()

    net.load_state_dict(torch.load(checkpoint))

    # Test the model
    test = tester(device, net, criterion, testloader)
    # Show the difference between predict and groundtruth (loss)
    print('Test Loss Result: ', test)

    # https://www.cnyes.com/twstock/ps_historyprice.aspx?code=1795
    # Predict
    predict_input = torch.tensor(
        [[[
            111.00,	112.50,	104.50,	106.00,	10631,  # 3/07
            105.00,	109.00,	101.50,	101.50, 9793,
            104.00,	107.50,	102.00, 103.00, 6808,
            106.00,	109.50, 104.50, 107.00, 8314,
            104.00,	105.50,	101.00,	102.50, 6565,  # 3/11
        ]]], dtype=torch.float32, device=device)
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        predict = net(predict_input)
    print('Predict Result', predict)
    answer = torch.tensor([[101.50, 99.70, 99.00, 108.50,
                          114.00]], dtype=torch.float32, device=device)
    loss = criterion(answer, predict)
    print('loss: ', loss)
