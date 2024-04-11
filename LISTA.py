import torch
from torch import nn
from utils import create_gaussian_filter
from Prox import Prox

class LISTA_block(nn.Module):

  def __init__(self, kernel_size=30, pad=1, init_alpha=None, init_beta=None):
    super().__init__()
    self.prox = Prox(init_alpha, init_beta)
    self.prox_f = Prox(init_alpha, init_beta)
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=pad, bias=False)
    self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=pad, bias=False)
    self.init_parameters(kernel_size)

  def init_parameters(self, kernel_size):
    with torch.no_grad():
      g = create_gaussian_filter(size=kernel_size)
      self.conv1.weight.data = torch.tensor([[g]], dtype=torch.float)
      self.conv2.weight.data = torch.tensor([[g]], dtype=torch.float)

  def forward(self, x1, x2):
    """ 
    Args:

    Returns:

    """
    B,C,H,W = x2.size()

    xc = self.conv1(x2.reshape(B*C,1,H,W))
    xc = self.prox(xc)

    xc = self.conv2(xc)

    xc = xc.reshape(B,C,H,W)
    y = x1 + x2 - xc

    output = self.prox_f(y)

    return output

class LISTA(nn.Module):

  def __init__(self, kernel_size=30, init_alpha=None, init_beta=None):
    """
    Args:

    """
    super().__init__()
    self.prox = Prox(init_alpha, init_beta)

    kernel_size = int((kernel_size-1)/2 + 1)
    self.pad = int((kernel_size-1)/2)

    self.block = LISTA_block(kernel_size=kernel_size, pad=self.pad, init_alpha=init_alpha, init_beta=init_beta)

    self.conv_i = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=self.pad, bias=False)
    self.conv_i2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=self.pad, bias=False)
    self.init_parameters(kernel_size)

  def init_parameters(self, kernel_size):
    with torch.no_grad():
      g = create_gaussian_filter(size=kernel_size)
      self.conv_i.weight.data = torch.tensor([[g]], dtype=torch.float)
      self.conv_i2.weight.data = torch.tensor([[g]], dtype=torch.float)

  def forward(self, x, iters):
    """
    Args:

    Returns:

    """
    x_first = x

    B,C,H,W = x_first.size()
    xc = self.conv_i(x_first.reshape(B*C,1,H,W))
    xc = self.prox(xc)

    xc = self.conv_i2(xc)
    xc = xc.reshape(B,C,H,W)

    x_prev = x_first
    b = self.block
    for i in range(iters):
      x_prev = b(x_first, x_prev)

    output = x_prev
    return output