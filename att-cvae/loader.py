from __future__ import print_function, division
import os
import torch
import pandas as pd
import yaml
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils import get_interpolated_ego_trajectories_json

class LbpsmDataset(Dataset):
    """ A dataset from BEV probabilitic semantic mapping in a local frame
    """

    def __init__(self, data_path, cfg, semantic_tf=None, use_gpu=False):
        """
        Args:
            TODO:

        """
        self.data_seqs = get_interpolated_ego_trajectories_json(data_path, cfg, False, semantic_tf, use_gpu)
        return
        
    def __len__(self):
        return len(self.data_seqs)

    
    def __getitem__(self, idx):
        return self.data_seqs[idx]

if __name__=="__main__":
    data_path = "/home/dfpazr/Documents/CogRob/avl/dataset-v1/datasetv1-api/lbpsm_extraction/lbpsm-data/"
    conf_path = "/home/dfpazr/Documents/CogRob/avl/TritonNet/lbpsm_net/lbpsm/config/model_config.yaml"
    with open(conf_path, "r") as conf_file:
        cfg = yaml.safe_load(conf_file)

    train = LBPSMDataset(data_path, cfg)
