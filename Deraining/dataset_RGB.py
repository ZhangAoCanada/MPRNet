import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        # inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'data')))
        # tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'gt')))

        # self.inp_filenames = [os.path.join(rgb_dir, 'data', x)  for x in inp_files if is_image_file(x)]
        # self.tar_filenames = [os.path.join(rgb_dir, 'gt', x) for x in tar_files if is_image_file(x)]

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'rain_L')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'gt')))

        self.inp_filenames = [os.path.join(rgb_dir, 'rain_L', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'gt', x) for x in tar_files if is_image_file(x)]

        # self.inp_filenames = [x  for x in inp_files if is_image_file(x)]
        # self.tar_filenames = [x for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)

        w,h = tar_img.size
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw!=0 or padh!=0:
            inp_img = TF.pad(inp_img, (0,0,padw,padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0,0,padw,padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr     = random.randint(0, hh-ps)
        cc     = random.randint(0, ww-ps)
        aug    = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr+ps, cc:cc+ps]
        tar_img = tar_img[:, rr:rr+ps, cc:cc+ps]

        # Data Augmentations
        if aug==1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug==2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug==3:
            inp_img = torch.rot90(inp_img,dims=(1,2))
            tar_img = torch.rot90(tar_img,dims=(1,2))
        elif aug==4:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=2)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=2)
        elif aug==5:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=3)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=3)
        elif aug==6:
            inp_img = torch.rot90(inp_img.flip(1),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(1),dims=(1,2))
        elif aug==7:
            inp_img = torch.rot90(inp_img.flip(2),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(2),dims=(1,2))
        
        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename

class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal, self).__init__()

        # inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'data')))
        # tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'gt')))

        # self.inp_filenames = [os.path.join(rgb_dir, 'data', x)  for x in inp_files if is_image_file(x)]
        # self.tar_filenames = [os.path.join(rgb_dir, 'gt', x) for x in tar_files if is_image_file(x)]

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'rain_L')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'gt')))

        self.inp_filenames = [os.path.join(rgb_dir, 'rain_L', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'gt', x) for x in tar_files if is_image_file(x)]

        # self.inp_filenames = [x  for x in inp_files if is_image_file(x)]
        # self.tar_filenames = [x for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)

        # # Resizing image in the multiple of 16"
        # wd_new,ht_new = inp_img.size
        # if ht_new>wd_new and ht_new>256:
        #     wd_new = int(np.ceil(wd_new*256/ht_new))
        #     ht_new = 256
        # elif ht_new<=wd_new and wd_new>256:
        #     ht_new = int(np.ceil(ht_new*1024/wd_new))
        #     wd_new = 256
        # wd_new = int(16*np.ceil(wd_new/16.0))
        # ht_new = int(16*np.ceil(ht_new/16.0))
        # inp_img = inp_img.resize((wd_new,ht_new), Image.ANTIALIAS)
        # tar_img = tar_img.resize((wd_new, ht_new), Image.ANTIALIAS)

        # Validate on center crop
        inp_img = TF.center_crop(inp_img, (480,720))
        tar_img = TF.center_crop(tar_img, (480,720))

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)

        ### NOTE: how to solve this ###
        # inp = TF.center_crop(inp, (480,720))

        inp = TF.to_tensor(inp)
        return inp, filename
