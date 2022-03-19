import numpy as np
import torch
import visdom
import argparse
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy.random as random
from collections import Counter

#============= ARGUMENT PARSING ==============

def setup_args():

    options = argparse.ArgumentParser()

    #setup
    options.add_argument('-sd', action="store", dest="save_file", default="./results/")
    options.add_argument('-dat1', action="store", dest="source_data_file", default="./data/ZFDOME_WT.npy")
    options.add_argument('-dat2', action="store", dest="target_data_file", default="./data/ZF50_WT.npy")
    options.add_argument('-env', action="store", dest="env", default="ubOT_ZF")

    # training arguments
    options.add_argument('-bs', action="store", dest="batch_size", default = 128, type = int)
    options.add_argument('-iter', action="store", dest="max_iter", default = 1000, type = int)
    options.add_argument('-citer', action="store", dest="critic_iter", default = 2, type = int)
    options.add_argument('-lambG', action="store", dest="lambG", default=10, type = float)
    options.add_argument('-lambG2', action="store", dest="lambG2", default=0.01, type = float)
    options.add_argument('-lamb0', action="store", dest="lamb0", default=0.001, type = float)
    options.add_argument('-lamb1', action="store", dest="lamb1", default=1.0, type = float)
    options.add_argument('-lamb2', action="store", dest="lamb2", default=1.0, type = float)
    options.add_argument('-psi', action="store", dest="psi2", default="JS")
    options.add_argument('-lrG', action="store", dest="lrG", default=1e-4, type = float)
    options.add_argument('-lrD', action="store", dest="lrD", default=1e-4, type = float)

    #options.add_argument('-psi', action="store", dest="psi2", default="EQ")

    # model arguments
    options.add_argument('-nz', action="store", dest="nz", default=50, type = int)
    options.add_argument('-nh', action="store", dest="n_hidden", default=50, type = int)

    return options.parse_args()

#============= DATA LOADING ==============

def setup_data_loaders(batch_size, data_file_1, data_file_2):
    data_1 = np.load(data_file_1)
    data_2 = np.load(data_file_2)

    max_length = max(len(data_1), len(data_2))

    idx_1 = np.random.permutation(len(data_1))
    idx_2 = np.random.permutation(len(data_2))

    # upsample smaller dataset
    if len(data_1)<max_length:
        idx_1 = np.random.randint(0, len(data_1), max_length)
    elif len(data_2)<max_length:
        idx_2 = np.random.randint(0, len(data_2), max_length)

    data_1 = data_1[idx_1]
    data_2 = data_2[idx_2]

    dataset = TensorDataset(torch.from_numpy(data_1).float(), torch.from_numpy(data_2).float())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return dataloader


#============= DATA LOGGING ==============


class Tracker(object):
    def __init__(self):
        self.epoch = 0
        self.sum = 0
        self.count = 0

        self.list_avg = []

    def add(self, val, count=1):
        self.sum += val*count
        self.count += count

    def tick(self):
        avg = self.sum / self.count
        self.list_avg.append(avg)

        self.epoch += 1
        self.sum = 0
        self.count = 0

    def get_status(self):
        return self.list_avg


#============= DATA VISUALIZATION ==============

def init_visdom(env):
    vis = visdom.Visdom()
    vis.close(env=env)
    return vis


def plot(tracker, tracker_plot, scale_plot, s_scale, env, vis):
    # update plots

    if tracker_plot is None:
        W1Plot=vis.line(
            Y=np.array(tracker.get_status()),
            X=np.array(range(tracker.epoch)),
            env=env,
        )
    else:
        W1Plot=vis.line(
            Y=np.array(tracker.get_status()),
            X=np.array(range(tracker.epoch)),
            env=env,
            win=tracker_plot,
            update='replace',
        )

    if scale_plot is None:
        scale_plot=vis.boxplot(
            X=s_scale,
            env=env,
        )

    else:
        scale_plot=vis.boxplot(
            X=s_scale,
            env=env,
            win=scale_plot,
        )

    return W1Plot, scale_plot
