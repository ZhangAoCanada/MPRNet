import sys
sys.path.append('/content/drive/MyDrive/DERAIN/MPRNet/Deraining')
from config import Config 
opt = Config('training.yml')
from difflib import restore
import torch.nn as nn
import torch
import os
from torch.utils.data import DataLoader
import utils
from data_RGB import get_validation_data
from MPRNet_custom import MPRNet
from skimage import img_as_ubyte
from skimage.measure import compare_psnr, compare_ssim
from tqdm import tqdm
import time, cv2
import numpy as np
from torchinfo import summary
from PIL import Image
import torchvision.transforms.functional as TF
from collections import OrderedDict

from ptflops import get_model_complexity_info

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location=torch.device('cpu'))
    # checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


model_restoration = MPRNet()
# summary(model_restoration)

model_path = './checkpoints/Deraining_2070images/models/MPRNet/model_best_256.pth'
load_checkpoint(model_restoration, model_path)
# utils.load_checkpoint(model_restoration, model_path)
print("===>Testing using weights: ", model_path)
model_restoration.to(device)
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

model_restoration = model_restoration.module
# model_restoration.to(device)


# video_path = "/home/ao/tmp/clip_videos/h97cam_water_video.mp4"
video_path = "/content/drive/MyDrive/DERAIN/DATA_captured/something_else/dusty_water_video1.mp4"

video = cv2.VideoCapture(video_path)


sample_image = None
while True:
    ret, frame = video.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 360))
    sample_image = frame
    break


def preProcess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image.astype(np.float32))
    # image = TF.to_tensor(TF.center_crop(Image.fromarray(image), (720, 1000)))
    return image


input_img = preProcess(sample_image)
input_img = input_img.unsqueeze(0)
print("[INFO] input shape ", input_img.shape)
input_img.to(device)
# input_img.cuda()


macs, params = get_model_complexity_info(model_restoration, (360, 640, 3), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
