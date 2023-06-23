from __future__ import print_function, division
import os
import time
import torch
import torch.nn as nn
import pandas as pd
import yaml
import cv2
import math
from loader import LbpsmDataset
from models import TridentTf 
from skimage import io, transform
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def calc_compliance(bev, x_p, y_p): 
    road = np.array([128, 64, 128])
    lane = np.array([255, 255, 255])
    cw = np.array([140, 140, 200])

    for r in range(max(0, y_p), min(399, y_p+1)):
        for c in range(max(0, x_p), min(399, x_p+1)):
            if bev[r, c][0] == road[0] and bev[r, c][1] == road[1] and bev[r, c][2] == road[2]:
                return True
            elif bev[r, c][0] == lane[0] and bev[r, c][1] == lane[1] and bev[r, c][2] == lane[2]: 
                return True
            elif bev[r, c][0] == cw[0] and bev[r, c][1] == cw[1] and bev[r, c][2] == cw[2]: 
                return True
            elif bev[r, c][0] == 0 and bev[r, c][1] == 0 and bev[r, c][2] == 255: 
                return True
    return False

def param_count(nn_model):
    total_params = sum(p.numel() for p in nn_model.parameters())
    trainable_params = sum(p.numel() for p in nn_model.parameters() if p.requires_grad)
    print("Total Params: {}".format(total_params))
    print("Total Trainable Params: {}".format(trainable_params))

if __name__=="__main__":

    data_path = "/home/dfpazr/Documents/CogRob/avl/TridentNetv1-v2/data/IntersectionScenes/Graphs/testing"
    conf_path = "/home/dfpazr/Documents/CogRob/avl/coarse_av_cgm/graph-cvae/config/model_config_stpf.yaml"
    weights_path = "/home/dfpazr/Documents/CogRob/avl/coarse_av_cgm/weights/intersection-scenes/att-stpf.pth"
    output_dir = "/home/dfpazr/Documents/CogRob/avl/TridentNetv1-v2/test-results/stpf/"

    with open(conf_path, "r") as conf_file:
        cfg = yaml.safe_load(conf_file)

    semantic_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x-0.5),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        ])


    test = LbpsmDataset(data_path, cfg, semantic_tf, use_gpu=True)
    dataloader = DataLoader(test, batch_size=1, shuffle=False, num_workers=0)
    horizon_length = cfg["nn_params"]["horizon"]
    h_map_dim = cfg["training_params"]["h_map_dim"]
    h_traj_dim = cfg["training_params"]["h_traj_dim"]
    z_dim = cfg["training_params"]["z_dim"]
    dec_dim = cfg["training_params"]["decoder_dim"]
    weight_separations = [i for i in range(5,40,5)] 

    all_avg_ade = []
    all_avg_ade_half = []
    all_avg_fde = []
    all_min_ade = []
    all_min_ade_half = []
    all_min_fde = []

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = TridentTf(horizon_length, h_map_dim, h_traj_dim, z_dim, dec_dim, device)
    model.to(device) 
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    curr_eval = 0
    ade = []
    ade_half = []
    fde = []
    mde_argmax = []
    mde_max = []

    percent_compliance_half = []
    percent_compliance = []

    mid = math.floor((horizon_length-1)/2)
    counter = 0
    inf_time = 0.0
    
    param_count(model)

    for batch in dataloader:
        idx, data_paths, lbpsm_map, waypoints_px, waypoints_xy, t_traj, p_traj = batch

        waypoints_xy = waypoints_xy.to(device)
        t_traj = t_traj.to(device)
        p_traj = p_traj.to(device)
        
        osm_traj = torch.cat((t_traj, p_traj), 2)

        for i in range(waypoints_xy.shape[0]):
            lbpsm_map_i = lbpsm_map[i].unsqueeze(0)
            waypoints_i = waypoints_xy[i].unsqueeze(0)
            osm_traj_i = osm_traj[i].unsqueeze(0)
            osm_traj_i = osm_traj_i.permute((0, 2, 1))
            waypoints = waypoints_i.permute((0, 2, 1))

            data_paths_i = data_paths[i]
            idx_i = idx[i]

            waypoints = waypoints_i.float().permute((0, 2, 1))
            time_start = time.time() 
            p_y_mz_mu, argmax_idx, loss, _ = model(lbpsm_map_i, osm_traj_i, waypoints, train=False)
            time_end = time.time()
            inf_time += (time_end - time_start)
            counter += 1

            y = p_y_mz_mu[argmax_idx]
            lbpsm_np = cv2.imread(str(data_paths_i) + "/semantic/" + str(idx_i.item()) + ".png")

            # Computer ADE
            ade_xy = np.linalg.norm(waypoints.cpu().numpy() - y.detach().cpu().numpy(), axis=2)
            ade_xy = np.mean(ade_xy, -1)

            # Compute ADE_half
            ade_xy_mid = np.linalg.norm(waypoints[0, 0:mid, :].cpu().numpy() - y[0, 0:mid, :].detach().cpu().numpy(), axis=-1)
            ade_xy_mid = np.mean(ade_xy_mid, -1)

            # Compute FDE
            fde_xy = np.linalg.norm(waypoints[0, -1, :].cpu().numpy() - y[0, -1, :].detach().cpu().numpy(), axis=-1)
        
            # Compute MDE
            mde_xy = np.linalg.norm(waypoints.cpu().numpy() - y.detach().cpu().numpy(), axis=2)
    
            mde_xy_argmax = np.argmax(mde_xy)
            mde_xy_max = np.max(mde_xy)

            ade.append(ade_xy)
            ade_half.append(ade_xy_mid)
            fde.append(fde_xy)
            mde_argmax.append(mde_xy_argmax)
            mde_max.append(mde_xy_max)


            compliance = 0
            compliance_h = 0
                           
            for j in range(waypoints.shape[1]):
                x_px = int((waypoints[0, j, 1]/cfg["maps"]["resolution"]) + 200)
                y_px = int((waypoints[0, j, 0]/cfg["maps"]["resolution"]) + 200)

                x_px_pred = int((y[0, j, 1]/cfg["maps"]["resolution"]) + 200)
                y_px_pred = int((y[0, j, 0]/cfg["maps"]["resolution"]) + 200)

                if calc_compliance(lbpsm_np, x_px_pred, y_px_pred):
                    compliance += 1
                    if j < mid + 1:
                        compliance_h += 1

                lbpsm_np = cv2.circle(lbpsm_np, (x_px, y_px), 1, (255, 0, 0), -1)

                lbpsm_np = cv2.circle(lbpsm_np,
                                    (x_px_pred,
                                     y_px_pred),
                                    1, (0, 255, 0), -1)

            cv2.imwrite(output_dir + "/" + str(curr_eval) + ".png", lbpsm_np)
            curr_eval += 1

            percent_compliance_half.append(compliance_h/5)
            percent_compliance.append(compliance/10)

    print("Avg ADE: " + str(np.mean(ade)))
    print("Avg ADE_half: " + str(np.mean(ade_half)))
    print("Avg FDE: " + str(np.mean(fde)))
    print("MDE Mode: " + str(stats.mode(mde_argmax)[0]))
    print("MDE Max: " + str(np.mean(mde_max)))
    print("compliance: " + str(np.mean(percent_compliance)))
    print("compliance half: " + str(np.mean(percent_compliance_half)))
    print("Avg. Inference: " + str(inf_time/counter) + "s")

    np.save("mde_stats.npy", mde_argmax)

    all_avg_ade.append(np.mean(ade))
    all_avg_ade_half.append(np.mean(ade_half))
    all_avg_fde.append(np.mean(ade))

    all_min_ade.append(min(ade))
    all_min_ade_half.append(min(ade_half))
    all_min_fde.append(min(fde))


