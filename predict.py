import numpy as np
import matplotlib.pyplot as plt
import torch
from model_UNet import *
from utils import *
from utils_eval import *

WEIGHTS_PATH = '/content/gdrive/My Drive/Unet_27-4-2019'

model = UNet(n_channels=3,n_classes=1) #only one class - high prob if sampling point, low prob if not
model = model.to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH))
model.eval()

def predict(rgb_imgs,depth_imgs):
  #takes rgb image and dense depth gt image
  #produces depth image interpolated from sparse depth sampling image
  
  rgb_imgs = rgb_imgs.to(device)
  depth_imgs = depth_imgs.numpy().squeeze(0).squeeze(0)
  sampling_pred_prob = model(rgb_imgs)
  sampling_pred_prob = sampling_pred_prob.detach().cpu().numpy().squeeze(0).squeeze(0)
  sampling_pred = img2centers(sampling_pred_prob, n_points=num_samples)
  depth_net = interpDepth(depth_imgs, sampling_pred, kind='linear')
  
  return (sampling_pred,depth_net)
