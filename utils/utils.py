import pickle
import numpy as np
import torch


def pickleStore(savethings, filename):
    dbfile = open(filename, 'wb')
    pickle.dump(savethings, dbfile)
    dbfile.close()
    return


def pikleOpen(filename):
    file_to_read = open(filename, "rb")
    p = pickle.load(file_to_read)
    return p


def readData(f):
    tmp = np.genfromtxt(f, delimiter='\t', dtype=str)[1:]
    tmp = np.flip(tmp, axis=0)
    return tmp


def saveModel(net, path):
    torch.save(net.state_dict(), path)
