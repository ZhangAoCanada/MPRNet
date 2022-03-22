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
from MPRNet import MPRNet
from skimage import img_as_ubyte
from skimage.measure import compare_psnr, compare_ssim
from tqdm import tqdm
import time, cv2
import numpy as np
from torchinfo import summary
from PIL import Image
import torchvision.transforms.functional as TF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_restoration = MPRNet()
# summary(model_restoration)

utils.load_checkpoint(model_restoration,'./checkpoints/Deraining/models/MPRNet/model_best.pth')
print("===>Testing using weights: ",'./checkpoints/Deraining/models/MPRNet/model_best.pth')
model_restoration.to(device)
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

video_saving_dir = "./checkpoints/Deraining/videos/MPRNet/"
if not os.path.exists(video_saving_dir):
    os.mkdir(video_saving_dir)

# video_path = "/content/drive/MyDrive/DERAIN/DATA_captured/something_else/sample_video.mp4"
# output_video_path = os.path.join(video_saving_dir, "result_video.avi")
video_path = "/content/drive/MyDrive/DERAIN/DATA_captured/something_else/sample_video1.mp4"
output_video_path = os.path.join(video_saving_dir, "result_video1.avi")
video = cv2.VideoCapture(video_path)
video_saving = cv2.VideoWriter(output_video_path,cv2.VideoWriter_fourcc('M','J','P','G'),30,(2000,720))


with torch.no_grad():
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = frame[:, 180:1200, :]
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_img = TF.center_crop(pil_img, (720, 1000))
        input_img = TF.to_tensor(pil_img)
        restored = model_restoration(input_img.unsqueeze(0))
        restored = torch.clamp(restored[0], 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored_img = img_as_ubyte(restored[0])
        image = np.concatenate((frame, restored_img[..., ::-1]), axis=1)
        video_saving.write(image)




# with torch.no_grad():
#     psnr_val_rgb = []
#     for ii, data_val in enumerate(tqdm(val_loader), 0):
#         target = data_val[0].to(device)
#         input_ = data_val[1].to(device)
#         filenames = data_val[2]

#         restored = model_restoration(input_)

#         restored = torch.clamp(restored[0], 0, 1)
#         restored_img = restored.cpu().numpy().squeeze().transpose((1,2,0))
#         gt = target[0].cpu().numpy().squeeze().transpose((1,2,0))

#         # restored = restored.permute(0, 2, 3, 1)
#         restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
#         for batch in range(len(restored)):
#             restored_img = img_as_ubyte(restored[batch])
#             utils.save_img((os.path.join('./checkpoints/Deraining/results/MPRNet/', filenames[batch] + '.png')), restored_img)
