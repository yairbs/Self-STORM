import torch
from torch import nn
import numpy as np

class Prox(nn.Module):
    def __init__(self, init_alpha=None, init_beta=None):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.init_parameters(init_alpha, init_beta)

    def init_parameters(self, init_alpha, init_beta):
      if init_alpha is None:
        nn.init.constant_(self.alpha, 0.95)
      else:
        nn.init.constant_(self.alpha, torch.tensor(np.squeeze(init_alpha)))
      if init_beta is None:
        nn.init.constant_(self.beta, 8)
      else:
        nn.init.constant_(self.alpha, torch.tensor(np.squeeze(init_beta)))

    def forward(self, x):
      B,C,H,W = x.size()
      x2 = x.reshape(B,C,H*W)
      i1 = torch.nanquantile(x2, 0.01,dim = 2,keepdim=True)
      i99 = torch.nanquantile(x2, 0.99,dim = 2,keepdim=True)
      th = i1+(i99-i1)*self.alpha
      th = torch.tile(th,[1,1,H*W])
      th = th.reshape(B,C,H,W)
      mask = (th>1e-14).float()
      th_new = th*mask + (1-mask)
      result = nn.functional.relu(x) / (1 + torch.exp(-(self.beta/th_new * (torch.abs(x) - (th*mask)))))
      return result