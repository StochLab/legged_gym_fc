import torch
import numpy as np 
from torch.distributions import Normal
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

def build_mlp(n_in, hidden, n_out, act_fn):
    hidden.append(n_out)
    li = []
    li.append(nn.Linear(n_in,hidden[0]))
    for i in range(1,len(hidden)):
        li.append(act_fn)
        li.append(nn.Linear(hidden[i-1],hidden[i]))
    return nn.Sequential(*nn.ModuleList(li))

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


class VelocityModel(nn.Module):
    def __init__(self, partial_state_dim = 34,
                 history_size = 30,
                 latent_model_hidden = [512, 512, 256],
                 activation='elu'):
        
        super(VelocityModel, self).__init__()
        
        activation = get_activation(activation)

        self.latent_model = build_mlp(history_size*partial_state_dim,
                                      latent_model_hidden,3,
                                      activation)
                
        self.optimizer = Adam(self.parameters(), lr = 3e-4)
    
    def get_velocity_prediction(self, partial_state_history):
        return self.latent_model(partial_state_history)
    
    
    def update(self, partial_state_history, velocity):
        loss = F.mse_loss(velocity,self.get_velocity_prediction(partial_state_history))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()






