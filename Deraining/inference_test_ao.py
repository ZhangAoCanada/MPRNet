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
from data_RGB import get_validation_data, get_test_rain_L_data, get_test_rain_H_data
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

model_path = './checkpoints/Deraining_2070images/models/MPRNet/model_best.pth'

testset_root_dir = "/content/drive/MyDrive/DERAIN/DATA_20220325/test_specific"
sub_dir_names = ["dawn_cloudy", "night_outdoors", "sunny_outdoors", "underground"]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_restoration = MPRNet()
# summary(model_restoration)

utils.load_checkpoint(model_restoration, model_path)
print("===>Testing using weights: ", model_path)
model_restoration.to(device)
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

def inferenceOneDir(testset_path, sub_name):
    test_rainL_dataset = get_test_rain_L_data(testset_path, {'patch_size':opt.TRAINING.VAL_PS})
    test_rainL_loader = DataLoader(dataset=test_rainL_dataset, batch_size=1, shuffle=False, drop_last=False)
    test_rainH_dataset = get_test_rain_H_data(testset_path, {'patch_size':opt.TRAINING.VAL_PS})
    test_rainH_loader = DataLoader(dataset=test_rainH_dataset, batch_size=1, shuffle=False, drop_last=False)
    print("===>Test RainL dataset size: ", len(test_rainL_dataset))
    print("===>Test RainH dataset size: ", len(test_rainH_dataset))

    all_inference_time = []
    psnr_list = []
    ssim_list = []

    # all_inference_time_L = []
    # psnr_list_L = []
    # ssim_list_L = []
    # with torch.no_grad():
    #     psnr_val_rgb = []
    #     for ii, data_val in enumerate(tqdm(test_rainL_loader), 0):
    #         target = data_val[0].to(device)
    #         input_ = data_val[1].to(device)
    #         filenames = data_val[2]

    #         start_time = time.time()
    #         restored = model_restoration(input_)
    #         all_inference_time.append(time.time() - start_time)
    #         all_inference_time_L.append(time.time() - start_time)

    #         restored = torch.clamp(restored[0], 0, 1)
    #         restored_img = restored.cpu().numpy().squeeze().transpose((1,2,0))
    #         gt = target[0].cpu().numpy().squeeze().transpose((1,2,0))

    #         # --- Calculate the average PSNR --- #
    #         psnr_list.extend(calc_psnr(restored_img, gt))
    #         psnr_list_L.extend(calc_psnr(restored_img, gt))
    #         # --- Calculate the average SSIM --- #
    #         ssim_list.extend(calc_ssim(restored_img, gt))
    #         ssim_list_L.extend(calc_ssim(restored_img, gt))

    #         # for res, tar in zip(restored[0], target):
    #         #     psnr_val_rgb.append(utils.torchPSNR(res, tar))

    #         # restored = restored.permute(0, 2, 3, 1)
    #         restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    #         for batch in range(len(restored)):
    #             restored_img = img_as_ubyte(restored[batch])
    #             utils.save_img((os.path.join('./checkpoints/Deraining_2070images/results/MPRNet/', filenames[batch] + '_pred.png')), restored_img)

    #     # psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
    #     # print("[Best_PSNR: %.4f Total_time: %.4f  Avg_time: %.4f]" % (psnr_val_rgb, totaltime, totaltime / num))


    all_inference_time_H = []
    psnr_list_H = []
    ssim_list_H = []
    with torch.no_grad():
        psnr_val_rgb = []
        for ii, data_val in enumerate(tqdm(test_rainH_dataset), 0):
            target = data_val[0].to(device)
            input_ = data_val[1].to(device)
            filenames = data_val[2]
            print("------ debug -----", target.shape, input_.shape)

            start_time = time.time()
            restored = model_restoration(input_)
            all_inference_time.append(time.time() - start_time)
            all_inference_time_H.append(time.time() - start_time)

            restored = torch.clamp(restored[0], 0, 1)
            restored_img = restored.cpu().numpy().squeeze().transpose((1,2,0))
            gt = target[0].cpu().numpy().squeeze().transpose((1,2,0))

            # --- Calculate the average PSNR --- #
            psnr_list.extend(calc_psnr(restored_img, gt))
            psnr_list_H.extend(calc_psnr(restored_img, gt))
            # --- Calculate the average SSIM --- #
            ssim_list.extend(calc_ssim(restored_img, gt))
            ssim_list_H.extend(calc_ssim(restored_img, gt))

            # for res, tar in zip(restored[0], target):
            #     psnr_val_rgb.append(utils.torchPSNR(res, tar))

            # restored = restored.permute(0, 2, 3, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                utils.save_img((os.path.join('./checkpoints/Deraining_2070images/results/MPRNet/', filenames[batch] + '_pred.png')), restored_img)

        # psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        # print("[Best_PSNR: %.4f Total_time: %.4f  Avg_time: %.4f]" % (psnr_val_rgb, totaltime, totaltime / num))

    print("----------------------- ", sub_name, " -----------------------")

    avr_psnr_L = sum(psnr_list_L) / (len(psnr_list_L) + 1e-10)
    avr_ssim_L = sum(ssim_list_L) / (len(ssim_list_L) + 1e-10)
    print("[RainL RESULTS] PSNR: {:.4f}, SSIM: {:.4f}, Average time: {:.4f} ms".format(avr_psnr_L, avr_ssim_L, np.mean(all_inference_time_L)*1000))

    avr_psnr_H = sum(psnr_list_H) / (len(psnr_list_H) + 1e-10)
    avr_ssim_H = sum(ssim_list_H) / (len(ssim_list_H) + 1e-10)
    print("[RainH RESULTS] PSNR: {:.4f}, SSIM: {:.4f}, Average time: {:.4f} ms".format(avr_psnr_H, avr_ssim_H, np.mean(all_inference_time_H)*1000))

    avr_psnr = sum(psnr_list) / (len(psnr_list) + 1e-10)
    avr_ssim = sum(ssim_list) / (len(ssim_list) + 1e-10)
    print("[OVERALL RESULTS] PSNR: {:.4f}, SSIM: {:.4f}, Average time: {:.4f} ms".format(avr_psnr, avr_ssim, np.mean(all_inference_time)*1000))

for sub_n in sub_dir_names:
    testset_path = os.path.join(testset_root_dir, sub_n)
    inferenceOneDir(testset_path, sub_n)