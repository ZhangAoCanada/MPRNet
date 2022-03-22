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

# ----------------- from TransWeather ------------------
def calc_psnr(im1, im2):
    im1, im2 = img_as_ubyte(im1), img_as_ubyte(im2)
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    ans = [compare_psnr(im1_y, im2_y)]
    return ans

def calc_ssim(im1, im2):
    im1, im2 = img_as_ubyte(im1), img_as_ubyte(im2)
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    ans = [compare_ssim(im1_y, im2_y)]
    return ans
# ------------------------------------------------------


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_restoration = MPRNet()
# summary(model_restoration)

utils.load_checkpoint(model_restoration,'./checkpoints/Deraining/models/MPRNet/model_best.pth')
print("===>Testing using weights: ",'./checkpoints/Deraining/models/MPRNet/model_best.pth')
model_restoration.to(device)
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


val_dataset = get_validation_data('/content/drive/MyDrive/DERAIN/test', {'patch_size':opt.TRAINING.VAL_PS})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, drop_last=False)

all_inference_time = []
psnr_list = []
ssim_list = []
with torch.no_grad():
    psnr_val_rgb = []
    for ii, data_val in enumerate(tqdm(val_loader), 0):
        target = data_val[0].to(device)
        input_ = data_val[1].to(device)
        filenames = data_val[2]

        start_time = time.time()
        with torch.no_grad():
            restored = model_restoration(input_)
        all_inference_time.append(time.time() - start_time)

        restored = torch.clamp(restored[0], 0, 1)
        restored_img = restored.cpu().numpy().squeeze().transpose((1,2,0))
        gt = target[0].cpu().numpy().squeeze().transpose((1,2,0))

        # --- Calculate the average PSNR --- #
        psnr_list.extend(calc_psnr(restored_img, gt))
        # --- Calculate the average SSIM --- #
        ssim_list.extend(calc_ssim(restored_img, gt))

        # for res, tar in zip(restored[0], target):
        #     psnr_val_rgb.append(utils.torchPSNR(res, tar))

        # restored = restored.permute(0, 2, 3, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        for batch in range(len(restored)):
            restored_img = img_as_ubyte(restored[batch])
            utils.save_img((os.path.join('./checkpoints/Deraining/results/MPRNet/', filenames[batch] + '.png')), restored_img)

    # psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
    # print("[Best_PSNR: %.4f Total_time: %.4f  Avg_time: %.4f]" % (psnr_val_rgb, totaltime, totaltime / num))

avr_psnr = sum(psnr_list) / (len(psnr_list) + 1e-10)
avr_ssim = sum(ssim_list) / (len(ssim_list) + 1e-10)
print("[RESULTS] PSNR: {:.4f}, SSIM: {:.4f}, Average time: {:.4f} ms".format(avr_psnr, avr_ssim, np.mean(all_inference_time)*1000))
