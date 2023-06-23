from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import pandas as pd
import yaml
import cv2
from loader import LbpsmDataset
from models import TridentTf 
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

if __name__=="__main__":


    conf_path = "/path/to/graph-cvae/config/model_config.yaml"

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

    data_path = cfg["train_path"] 
    output_dir = cfg["vis_train"] 

    training = LbpsmDataset(data_path, cfg, semantic_tf, use_gpu=True)
    batch_size = cfg["training_params"]["batch_size"]
    dataloader = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=0)

    horizon_length = cfg["nn_params"]["horizon"]
    h_map_dim = cfg["training_params"]["h_map_dim"]
    h_traj_dim = cfg["training_params"]["h_traj_dim"]
    z_dim = cfg["training_params"]["z_dim"]
    dec_dim = cfg["training_params"]["decoder_dim"]
    total_epochs = cfg["training_params"]["number_epochs"]

    nn_model = TridentTf(horizon_length, h_map_dim, h_traj_dim, z_dim, dec_dim)
    for param in nn_model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    if torch.cuda.is_available():
        nn_model.cuda()
    nn_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, nn_model.parameters()), lr=9e-4)
    nn_model = nn_model.train()
    curr_epoch = 1
    

    L = 0
    writer = SummaryWriter()
    # TRAIN
    loss_terms = [0,0,0,0]
    while(1):
        for batch in dataloader:
            nn_optimizer.zero_grad()
            idx, img_labels, lbpsm_map, waypoints_px, waypoints_xy, t_traj, p_traj = batch
            osm_traj = torch.cat((t_traj, p_traj), 2)
            L = 0
            for i in range(waypoints_xy.shape[0]):
                lbpsm_map_i = lbpsm_map[i].unsqueeze(0)
                waypoints_i = waypoints_xy[i].unsqueeze(0)
                osm_traj_i = osm_traj[i].unsqueeze(1)
                osm_traj_i = osm_traj_i.permute((2, 1, 0))
                
                #osm_traj_i = osm_traj[i].unsqueeze(0) 
                #osm_traj_i = osm_traj_i.permute((0, 2, 1))

                waypoints = waypoints_i.permute((0, 2, 1))
            
                p_y_mz_mu, argmax_idx, loss, loss_terms = nn_model(lbpsm_map_i, osm_traj_i, waypoints, train=True)
                
                L += loss
            L = L/batch_size
            print("L: " + str(L) + " log_p: " + str(loss_terms[0]) + " kl: " + str(loss_terms[1]) + " mse: " + str(loss_terms[2]))
            L.backward()
            nn_optimizer.step()
            torch.cuda.empty_cache()

        writer.add_scalar('Loss/train', L, curr_epoch)
        writer.add_scalar('Loss/train/log_p', loss_terms[0], curr_epoch)
        writer.add_scalar('Loss/train/kl', loss_terms[1], curr_epoch)
        writer.add_scalar('Loss/train/mse', loss_terms[2], curr_epoch)
        
        # Save and validate every 10 epochs
        if curr_epoch % 5 == 0:
            torch.save(nn_model.state_dict(), "./weights/cvae-" + str(curr_epoch) + ".pth")

            nn_eval = nn_model.eval()
            curr_eval = 0
            for batch in dataloader:
                idx, data_paths, lbpsm_map, waypoints_px, waypoints_xy, t_traj, p_traj = batch
                osm_traj = torch.cat((t_traj, p_traj), 2)

                for i in range(waypoints_xy.shape[0]):
                    lbpsm_map_i = lbpsm_map[i].unsqueeze(0)
                    waypoints_i = waypoints_xy[i].unsqueeze(0)
                    osm_traj_i = osm_traj[i].unsqueeze(1)
                    osm_traj_i = osm_traj_i.permute((2, 1, 0))
                    #osm_traj_i = osm_traj[i].unsqueeze(0) 
                    #osm_traj_i = osm_traj_i.permute((0, 2, 1))
                    data_paths_i = data_paths[i]
                    idx_i = idx[i]

                    #waypoints = waypoints_i.float().cuda().permute((0, 2, 1))
                    waypoints = waypoints_i.permute((0, 2, 1))
                
                    p_y_mz_mu, argmax_idx, loss, _ = nn_eval(lbpsm_map_i, osm_traj_i, waypoints, train=False)

                    y = p_y_mz_mu[argmax_idx]
                    lbpsm_np = cv2.imread(str(data_paths_i) + "/semantic/" + str(idx_i.item()) + ".png")

                    for j in range(waypoints.shape[1]):
                        
                        lbpsm_np = cv2.circle(lbpsm_np, 
                                            (int((waypoints[0, j, 1]/cfg["maps"]["resolution"]) + 200),
                                            int((waypoints[0, j, 0]/cfg["maps"]["resolution"]) + 200)),
                                            1, (255, 0, 0), -1)
                        
                        lbpsm_np = cv2.circle(lbpsm_np, 
                                            (int((y[0, j, 1]/cfg["maps"]["resolution"]) + 200),
                                            int((y[0, j, 0]/cfg["maps"]["resolution"]) + 200)),
                                            1, (0, 255, 0), -1)
                    cv2.imwrite(output_dir + str(curr_eval) + ".png", lbpsm_np)
                    curr_eval += 1

            nn_model = nn_model.train()

        if curr_epoch == total_epochs:
            break
        curr_epoch += 1

