import numpy as np
import torch
import statistics


def preprocess(data):
    """
    # (OPTIONAL)
    # Save all the columns to variables
        date   = data[:, 0] # the first column of data
        open   = data[:, 1]
        high   = data[:, 2]
        low    = data[:, 3]
        close  = data[:, 4]
        volume = data[:, 5]
    """
    date = data[:, 0]
    open = np.array(data[:, 1]).astype(np.float64)
    high = np.array(data[:, 2]).astype(np.float64)
    low = np.array(data[:, 3]).astype(np.float64)
    close = np.array(data[:, 4]).astype(np.float64)
    volume = np.array(data[:, 5]).astype(np.float64)

    stats = dict({
        "open": {"std": statistics.stdev(open), "mean": statistics.mean(open)},
        "high": {"std": statistics.stdev(high), "mean": statistics.mean(high)},
        "low": {"std": statistics.stdev(low), "mean": statistics.mean(low)},
        "close": {"std": statistics.stdev(close), "mean": statistics.mean(close)},
        "volume": {"std": statistics.stdev(volume), "mean": statistics.mean(volume)},
    })

    prices = np.array([[open, high, low, close, volume] for date, open, high, low,
                      close, volume in data]).astype(np.float64)
    # print(prices)
    return prices, stats


def train_test_split(data, percentage=0.8):
    train_size = int(len(data) * percentage)
    train, test = data[:train_size], data[train_size:]
    return train, test


def transform_dataset(dataset, stats, look_back=5, target_days=1):
    # N days as training sample
    dataX = [np.reshape(normalize(dataset[i:(i + look_back)], stats), (1, -1))[0]
             for i in range(len(dataset)-look_back-target_days)]
    # 1 day as groundtruth
    dataY = [dataset[i + look_back:i+look_back+target_days, 3]
             for i in range(len(dataset)-look_back-target_days)]

    return torch.tensor(np.array(dataX), dtype=torch.float32), torch.tensor(np.array(dataY), dtype=torch.float32)


def normalize(rows, stats):
    tmp = np.array([__normalize_specify(row, stats) for row in rows])

    return tmp


def __normalize_specify(row, stats):
    tmp = [
        (row[0]-stats['open']['mean'])/stats['open']['std'],
        (row[1]-stats['high']['mean'])/stats['high']['std'],
        (row[2]-stats['low']['mean'])/stats['low']['std'],
        (row[3]-stats['close']['mean'])/stats['close']['std'],
        (row[4]-stats['volume']['mean'])/stats['volume']['std'],
    ]

    return tmp
