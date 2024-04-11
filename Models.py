import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import create_gaussian_filter
from LISTA import LISTA


class DecoderEncoder(nn.Module):
  """Model-based autoencoder"""
  def __init__(self, e_kernel_size=30, d_kernel_size=30, scale_factor=4):
    super().__init__()
    self.scale_factor = scale_factor
    e_ks = int((e_kernel_size-1)/2 + 1)
    
    self.decoder = LISTA(kernel_size=d_kernel_size)
    self.decoder = self.decoder.float()

    self.prox1 =  nn.ReLU(inplace=False)
    self.prox2 =  nn.ReLU(inplace=False)

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=e_ks, stride=1, padding=int((e_ks-1)/2), bias=False)
    self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=e_ks, stride=1, padding=int((e_ks-1)/2), bias=False)
    self.init_parameters(e_ks)

  def init_parameters(self, kernel_size):
    with torch.no_grad():
      g = create_gaussian_filter(size=kernel_size)
      self.conv1.weight.data = torch.tensor([[g]], dtype=torch.float)
      self.conv2.weight.data = torch.tensor([[g]], dtype=torch.float)

  def forward(self, x, iters, interp_mode='nearest'):
    scale_factor = self.scale_factor
    upsample_output = F.interpolate(x, scale_factor=scale_factor, mode=interp_mode)
    upsample_output = upsample_output.float()
    decoder_output = self.decoder(upsample_output, iters)
    decoder_output = decoder_output.float()

    B,C,H,W = decoder_output.size()
    ec1 = self.conv1(decoder_output.reshape(B*C,1,H,W))
    ec1 = self.prox1(ec1)
    ec2 = self.conv2(ec1)
    ec2 = self.prox2(ec2)
    encoder_output = ec2.reshape(B,C,H,W)
    encoder_output = encoder_output.float()

    return encoder_output, decoder_output


class ZSSR(nn.Module):
     """ZSSR-like model"""
     def __init__(self, scale_factor=4, e_kernel_size=30):
          super(ZSSR, self).__init__()

          self.scale_factor = scale_factor
          d = 1
          no_channels = 64
          e_ks = int((e_kernel_size-1)/2 + 1)

          self.input = nn.Conv2d(in_channels=d, out_channels=no_channels, kernel_size=3, stride=1, padding=1, bias=False)
          self.conv1 = nn.Conv2d(in_channels=no_channels, out_channels=no_channels, kernel_size=3, stride=1, padding=1, bias=False)
          self.conv2 = nn.Conv2d(in_channels=no_channels, out_channels=no_channels, kernel_size=3, stride=1, padding=1, bias=False)
          self.conv3 = nn.Conv2d(in_channels=no_channels, out_channels=no_channels, kernel_size=3, stride=1, padding=1, bias=False)
          self.conv4 = nn.Conv2d(in_channels=no_channels, out_channels=no_channels, kernel_size=3, stride=1, padding=1, bias=False)
          self.conv5 = nn.Conv2d(in_channels=no_channels, out_channels=no_channels, kernel_size=3, stride=1, padding=1, bias=False)
          self.conv6 = nn.Conv2d(in_channels=no_channels, out_channels=no_channels, kernel_size=3, stride=1, padding=1, bias=False)
          self.output = nn.Conv2d(in_channels=no_channels, out_channels=d, kernel_size=3, stride=1, padding=1, bias=False)
          self.relu = nn.ReLU(inplace=False)

		      # weights initialization
          for m in self.modules():
            if isinstance(m, nn.Conv2d):
              n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
              m.weight.data.normal_(0, sqrt(2. / n))

          self.conv1_e = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=e_ks, stride=1, padding=int((e_ks-1)/2), bias=False)
          self.conv2_e = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=e_ks, stride=1, padding=int((e_ks-1)/2), bias=False)
          self.init_parameters(e_ks)

     def init_parameters(self, kernel_size):
        with torch.no_grad():
          g = create_gaussian_filter(size=kernel_size)
          self.conv1_e.weight.data = torch.tensor([[g]], dtype=torch.float)
          self.conv2_e.weight.data = torch.tensor([[g]], dtype=torch.float)

     def forward(self, x, interp_mode='nearest'):
        
        scale_factor = self.scale_factor
        upsample_output = F.interpolate(x, scale_factor=scale_factor, mode=interp_mode)
        upsample_output = upsample_output.float()

        residual = upsample_output

        inputs = self.input(self.relu(upsample_output))
        out = inputs
        
        out = self.conv1(self.relu(out))
        out = self.conv2(self.relu(out))
        out = self.conv3(self.relu(out))
        out = self.conv4(self.relu(out))
        out = self.conv5(self.relu(out))
        out = self.conv6(self.relu(out))

        out = self.output(self.relu(out))
        decoder_output = torch.add(out, residual)

        ec1 = self.conv1_e(decoder_output)
        ec1 = self.relu(ec1)
        ec2 = self.conv2_e(ec1)
        ec2 = self.relu(ec2)
        encoder_output = ec2.float()

        return encoder_output, decoder_output