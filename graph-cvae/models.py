import cv2
import numpy as np
import math
import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision import transforms
from torch.autograd import Variable

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=90):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:x.size(0), :], requires_grad=False)
        return self.dropout(x)

class SemanticEncoder(nn.Module):

    def __init__(self, h_map_dim):
        super(SemanticEncoder, self).__init__()

        self.h_map_dim = h_map_dim
        self.semantic_conv_layers = []
        self.semantic_fc_layers = []
        self.semantic_conv_layers.append(nn.Conv2d(3, 32, 7, stride=(2, 2)))
        self.semantic_conv_layers.append(nn.LeakyReLU())
        self.semantic_conv_layers.append(nn.Conv2d(32, 48, 5, stride=(2, 2)))
        self.semantic_conv_layers.append(nn.LeakyReLU())
        self.semantic_conv_layers.append(nn.Conv2d(48, 64, 3, stride=(2, 2)))
        self.semantic_conv_layers.append(nn.LeakyReLU())
        self.semantic_conv_layers.append(nn.Conv2d(64, 64, 3, stride=(2, 2)))

        self.semantic_fc_layers.append(nn.Linear(7744, 1024))
        self.semantic_fc_layers.append(nn.LeakyReLU())

        self.semantic_conv = nn.Sequential(*self.semantic_conv_layers)
        self.semantic_fc = nn.Sequential(*self.semantic_fc_layers)

    def forward(self, semantic_map):
        
        semantic_h = self.semantic_conv(semantic_map)
        semantic_h = semantic_h.view(semantic_h.size(0), -1)
        semantic_h = self.semantic_fc(semantic_h)

        return semantic_h

class OsmEncoder(nn.Module):
    def __init__(self, traj_dim, d_model, n_head, num_layers, drop_out):
        super(OsmEncoder, self).__init__()


        self.pe = PositionalEncoding(d_model, dropout=drop_out, max_len=1000)
        self.feat_encoder = nn.Linear(traj_dim, d_model)

        enc_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_head,
                                            dim_feedforward=2048, dropout=drop_out)

        self.tf_encoder = TransformerEncoder(enc_layer, num_layers=num_layers)
        self.reduce_fc1 = nn.Linear(10240, 5120)
        self.fc_lr1 = nn.LeakyReLU()
        self.reduce_fc2 = nn.Linear(5120, 1024)
        self.fc_lr2 = nn.LeakyReLU()
        self.h_mask = self.src_mask(40)
        self.h_mask = self.h_mask.cuda() 
 
    def src_mask(self, dim):
        m = (torch.triu(torch.ones(dim, dim)) == 1).transpose(0, 1)
        return m.float().masked_fill(m == 0, float('-inf')).masked_fill(m == 1, float(0.0))

    def forward(self, osm_traj):
        use_mask = True
        h = self.feat_encoder(osm_traj)
        h = self.pe(h)

        if use_mask and torch.cuda.is_available():
            h = self.tf_encoder(h, self.h_mask)
        else:
            h = self.tf_encoder(h)
        
        h2 = h.view(h.size(1), -1) 
        h3 = self.fc_lr1(self.reduce_fc1(h2))
        h4 = self.fc_lr2(self.reduce_fc2(h3))
        
        return h4 

class OsmAttention(nn.Module):
    def __init__(self):
        super(OsmAttention, self).__init__()
    
        self.q_proj = nn.Linear(3, 3)
        self.k_proj = nn.Linear(3, 3)
        self.v_proj = nn.Linear(3, 3)

    def forward(self, osm_traj):
        """
            osm_traj: OSM representation with shape (N, L, D) where N is batch size,
                      L is number of features, and D is feature dimension.
        """
        
        N, L, D, = osm_traj.shape 
        q = self.q_proj(osm_traj)
        k = self.k_proj(osm_traj)
        v = self.v_proj(osm_traj)

        sm_arg = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(3)
        softmax_prod = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(3), -1)
        attention = torch.matmul(softmax_prod, v)
        attention = attention.reshape(N, L, D)
        
        return osm_traj + attention
        
        
    
    
class OsmPool(nn.Module):
    def __init__(self, traj_dim, d_model, n_head, num_layers, drop_out):
        super(OsmPool, self).__init__()

        self.fc1 = nn.Linear(3, 64)
        self.fc_lr1 = nn.LeakyReLU()

        self.fc2 = nn.Linear(64, 128)
        self.fc_lr2 = nn.LeakyReLU()

        self.fc3 = nn.Linear(128, 1024)
        self.fc_lr3 = nn.LeakyReLU()
        
        

    def forward(self, osm_traj):
        
        h1 = self.fc_lr1(self.fc1(osm_traj))
        h2 = self.fc_lr2(self.fc2(h1))
        h3 = self.fc_lr3(self.fc3(h2)) # (1, 40, 1024)
        
        h4 = h3.permute((0, 2, 1)) 
        h4 = F.max_pool1d(h4, kernel_size=h4.shape[2]).squeeze(2) 

        return h4



class OsmMLP(nn.Module):
    def __init__(self):
        super(OsmMLP, self).__init__()

        self.reduce_fc1 = nn.Linear(120, 2400)
        self.fc_lr1 = nn.LeakyReLU()

        self.reduce_fc2 = nn.Linear(2400, 1024)
        self.fc_lr2 = nn.LeakyReLU()
 

    def forward(self, osm_traj):
        
        h2 = osm_traj.reshape(osm_traj.size(0), -1) 
        h3 = self.fc_lr1(self.reduce_fc1(h2))
        h4 = self.fc_lr2(self.reduce_fc2(h3))
        
        return h4 

class MapEncoder(nn.Module):
    def __init__(self, h_map_dim):
        super(MapEncoder, self).__init__()

        self.h_map_dim = h_map_dim
        
        # Semantic Map layers
        self.semantic_encoder = SemanticEncoder(h_map_dim)

        # OSM layers 
        #self.osm_encoder = OsmEncoder(3, 512, 8, 6, drop_out=0.1)
        #self.osm_encoder = OsmMLP(3, 256, 8, 6, drop_out=0.1)
        #self.osm_encoder = OsmPool(3, 256, 8, 6, drop_out=0.1)
        self.osm_att = OsmAttention()
        self.osm_encoder = OsmMLP()
        self.osm_att_lr1 = nn.LeakyReLU()

        # Conc. layers
        self.fc_merged1 = nn.Linear(2048, 256)
        self.fc_lr1 = nn.LeakyReLU()
        self.fc_merged2 = nn.Linear(256, self.h_map_dim)



    def forward(self, semantic_map, osm_traj):
        
            
        # semantic encoder
        h_semantic = self.semantic_encoder(semantic_map)    

        # osm encoder
        osm_att = self.osm_att_lr1(self.osm_att(osm_traj))
        h_osm_traj = self.osm_encoder(osm_att)

        h_map = torch.cat((h_osm_traj, h_semantic), 1)
        h_map = self.fc_merged1(h_map)
        h_map = self.fc_lr1(h_map)
        h_map = self.fc_merged2(h_map) 

        return h_map
        
class FutureTrajectoryEncoder(nn.Module):
    def __init__(self, h_y_dim, horizon):
        super(FutureTrajectoryEncoder, self).__init__()
            
        self.h_y_dim = h_y_dim 
        self.horizon = horizon
        self.h = nn.Linear(2, self.h_y_dim) 
        self.c = nn.Linear(2, self.h_y_dim) 
        self.traj_encoder = nn.LSTM(input_size=2, 
                                    hidden_size=self.h_y_dim,
                                    bidirectional=True,
                                    batch_first=True) 
    def forward(self, waypoints):
        """
            waypoints: [batch_size, horizon, 2]
            returns: [batch_size, num_layers*num_directions(i.e 4)*h_y_dim] 
        """

        # Requires [1, horizon, 2]
        _, h_traj = self.traj_encoder(waypoints)

        # [batch_size, 4, 32]
        h_traj = torch.cat(h_traj, dim=0).permute(1,0,2)
        return torch.reshape(h_traj, (-1, h_traj.shape[1]*h_traj.shape[2]))

class FutureTrajectoryDecoder(nn.Module):
    def __init__(self, z_dim, h_m_dim, dec_dim, horizon, dev):
        super(FutureTrajectoryDecoder, self).__init__()
            
        self.dev = dev
        self.z_dim = z_dim
        self.h_m_dim = h_m_dim 
        self.dec_dim = dec_dim
        self.horizon = horizon
        self.h_init = nn.Linear(z_dim + h_m_dim, dec_dim)
        self.gru = nn.GRUCell(z_dim + h_m_dim + 2, dec_dim)

        self.p_y_mz_mu_t = nn.Linear(dec_dim, 2)
        self.p_y_mz_cov_t = nn.Linear(dec_dim, 3)

    def forward(self, zh_m):
        # Decode zh_m complete for GRU: (1, z_dim, z_dim+h_m_dim)
        #                    mu->(1, z_dim, horizon, 2)        
        #                    cov->(1, z_dim, horizon, 2, 2)        
        # Decode zh_m max for GRU: (1, 1, z_dim+h_m_dim)
        #                    mu->(1, 1, horizon, 2)        
        #                    cov->(1, 1, horizon, 2, 2)     
        mu_t = None
        cov_t = None
        all_mu = []
        all_cov = []

        
        z_dim = zh_m.shape[0]
        h_current = self.h_init(zh_m)
        mu_t = torch.zeros(2).to(self.dev).repeat(z_dim, 1)

        for i in range(self.horizon):

            gru_input = torch.cat([zh_m, mu_t], dim=1)
            h_next = self.gru(gru_input, h_current)
            

            # cov decoded as [1, horizon, (var_x, cov_xy, var_y)]
            #             ->[[1, horizon, ((var_x, cov_xy), (cov_xy, var_y))]
            mu_t = self.p_y_mz_mu_t(h_next)
            cov_t = self.p_y_mz_cov_t(h_next)

            cov_x = torch.clamp(cov_t[:, 0], 0.2, 1)
            cov_xy = torch.clamp(cov_t[:, 1], -0.1, 0.1)
            cov_y = torch.clamp(cov_t[:, 2], 0.2, 1)
            full_cov_t = torch.cat([cov_x.unsqueeze(1),
                              cov_xy.unsqueeze(1),
                              cov_xy.unsqueeze(1),
                              cov_y.unsqueeze(1)], dim=1).reshape(-1, 2, 2)

            # unsqueeze a dimension
            all_mu.append(mu_t.reshape((z_dim, 2)).unsqueeze(1))
            all_cov.append(full_cov_t.reshape((z_dim, 2, 2)).unsqueeze(1))

            h_current = h_next

        all_mu = torch.cat(all_mu, dim=1) 
        all_cov = torch.cat(all_cov, dim=1) 
        return all_mu, all_cov
    
class ProbDists(nn.Module):
    def __init__(self, h_m_dim, h_y_dim, z_dim, dec_dim, dev):
        super(ProbDists, self).__init__()
        self.h_m_dim = h_m_dim
        self.h_y_dim = h_y_dim
        self.z_dim = z_dim
        self.dec_dim = dec_dim
        self.dev = dev
        
        # FC for converting h_m, h_my embeddings to a z_dim dimension
        self.z_m = nn.Linear(h_m_dim, z_dim)
        self.z_my = nn.Linear(h_m_dim + h_y_dim, z_dim)
        
        
    def one_hot_categorical(self, h, train=False):
        h_norm = h - torch.mean(h, dim=1, keepdim=True)
        # logit clip
        logits = torch.clamp(h_norm, -2.5, 2.5)
        return torch.distributions.OneHotCategorical(logits=logits)

    def p_z_m(self, h_m, train=False):
        h_m = self.z_m(h_m)
        p_z_m = self.one_hot_categorical(h_m, train)

        return p_z_m

    def q_z_my(self, h_my, train=False):
        h_my = self.z_my(h_my) 
        q_z_my = self.one_hot_categorical(h_my, train)
        
        return q_z_my 

    def sample_z(self, p_z_m, z_dim, train=False, full=False):
        #if(train or full):
        # If training or full distribution needed return I
        #I = torch.eye(z_dim).unsqueeze(2)
        z_complete = torch.eye(z_dim).unsqueeze(0)
        # [1, z_dim, z_dim]
        z_argmax_idx = torch.argmax(p_z_m.probs, dim=1)
        z_argmax = torch.eye(z_dim)[z_argmax_idx]

        return z_complete.to(self.dev).squeeze(), z_argmax.to(self.dev), z_argmax_idx.to(self.dev)
    
    def estimate_kl(self, q_z_my, p_z_m):
        kl = torch.distributions.kl_divergence(q_z_my, p_z_m)
        kl_sum = torch.sum(torch.mean(kl, dim=0, keepdim=True)) 
        return kl_sum     
        
        

    def p_y_mz(self, mu, cov, y, horizon, train=False):
        """
            mu: p_y_mz mean [1, z_dim, horizon, 2]
            cov: p_y_mz covariance [1, z_dim, horizon, 2, 2]
            y: values to be used for evaluating MVN pdf: [1, horizon, 2]
        """
        batch_size = mu.shape[0] 
        y_mean = y.unsqueeze(1) - mu

        # MVN: (batch_size, z_dim, horizon, 2)
        p_y_mz = torch.distributions.multivariate_normal.MultivariateNormal(mu, cov)

        return p_y_mz


class TridentTf(nn.Module):
    def __init__(self, horizon, h_m_dim, h_y_dim, z_dim, dec_dim, dev):
        super(TridentTf, self).__init__()
        self.horizon = horizon
        self.h_m_dim = h_m_dim
        self.h_y_dim = h_y_dim
        self.z_dim = z_dim
        self.dec_dim = dec_dim

        self.dev = dev

        # Encode map: h_m
        self.map_encoder = MapEncoder(self.h_m_dim)

        # Encode ground truth future trajectories (train/eval only): h_y
        self.future_traj_encoder = FutureTrajectoryEncoder(h_y_dim, horizon)
        self.prob_dist = ProbDists(h_m_dim, h_y_dim*4, z_dim, dec_dim, self.dev)

        # Decoder
        self.traj_decoder = FutureTrajectoryDecoder(z_dim, h_m_dim, dec_dim, horizon, self.dev)     
       

    
    def forward(self, semantic_map, osm_traj, waypoints, train=False):
        """
            waypoints: [1, horizon, 2]
        """
        
        # Encode Future Trajectories and Current Map
        #   map encoding: [1, h_map_dim=60]
        h_m = self.map_encoder(semantic_map, osm_traj)
        # gt trajectory encoding: [1, h_traj_dim=h_y_dim*4] 
        h_y = self.future_traj_encoder(waypoints)

        # gt trajectory and map encodings: [1, h_m_dim + h_y_dim]
        h_my = torch.cat([h_m, h_y], dim=1)
        p_z_m = self.prob_dist.p_z_m(h_m, train)
        q_z_my = self.prob_dist.q_z_my(h_my,  train)

        kl = self.prob_dist.estimate_kl(q_z_my, p_z_m)

        # obtain z: (z_dim, z_dim), (1, z_dim), (1)
        if(train):
            z_complete, z_argmax, argmax_idx = self.prob_dist.sample_z(q_z_my, self.z_dim, train) 
        else:
            z_complete, z_argmax, argmax_idx = self.prob_dist.sample_z(p_z_m, self.z_dim, train) 

        # Decode
        # zh_m: (batch_size, z_dim, z_dim + h_m_dim)
        zh_m_comp = torch.cat([z_complete, h_m.repeat(self.z_dim, 1)], dim=1) 
        
        p_y_mz_mu, p_y_mz_cov = self.traj_decoder(zh_m_comp)

        # Include y_ground truth and evaluate p_y_mz
        # mu: [batch_size, z_dim, horizon, 2]
        # cov: [batch_size, z_dim, horizon, 2, 2]
        p_y_mz = self.prob_dist.p_y_mz(p_y_mz_mu, p_y_mz_cov, waypoints, self.horizon)

        # [1, z_dim, horizon]
        log_p_y_mz = p_y_mz.log_prob(waypoints.unsqueeze(1))
            
        if(train):
            # [1, z_dim, horizon]
            log_q_z_my = q_z_my.logits - torch.logsumexp(q_z_my.logits, dim=-1, keepdim=True) 
            log_q_z_my = log_q_z_my.unsqueeze(2).repeat(1,1,self.horizon)
            
            p_y_m = torch.logsumexp(log_p_y_mz + log_q_z_my, dim=1)
            

        else:
            # [1, z_dim, horizon]
            log_p_z_m = p_z_m.logits - torch.logsumexp(p_z_m.logits, dim=-1, keepdim=True) 
            log_p_z_m = log_p_z_m.unsqueeze(2).repeat(1,1,self.horizon)

            p_y_m = torch.logsumexp(log_p_y_mz + log_p_z_m, dim=1)
            
        
        p_y_m = torch.clamp(p_y_m, max=6.0)
        # [batch_size, horizon]

        ELBO = torch.mean(p_y_m, dim=1) - kl 
        
        loss = -ELBO + F.mse_loss(p_y_mz_mu[argmax_idx], waypoints)
        loss_terms = [torch.mean(p_y_m, dim=1), -kl, F.mse_loss(p_y_mz_mu[argmax_idx], waypoints)]

        return p_y_mz_mu, argmax_idx, loss, loss_terms


    def generate(self, semantic_map, osm_map):
        """
            waypoints: [1, horizon, 2]
        """
        
        # Encode Future Trajectories and Current Map
        #   map encoding: [1, h_map_dim=60]
        h_m = self.map_encoder(semantic_map, osm_map)
        # gt trajectory encoding: [1, h_traj_dim=h_y_dim*4] 

        # gt trajectory and map encodings: [1, h_m_dim + h_y_dim]
        p_z_m = self.prob_dist.p_z_m(h_m, False)

        # obtain z: (z_dim, z_dim), (1, z_dim), (1)

        z_complete, z_argmax, argmax_idx = self.prob_dist.sample_z(p_z_m, self.z_dim, False) 

        # Decode
        # zh_m: (batch_size, z_dim, z_dim + h_m_dim)
        zh_m_comp = torch.cat([z_complete, h_m.repeat(self.z_dim, 1)], dim=1) 

        p_y_mz_mu, p_y_mz_cov = self.traj_decoder(zh_m_comp)
       

        # Return trajectory   
        return p_y_mz_mu[argmax_idx]
