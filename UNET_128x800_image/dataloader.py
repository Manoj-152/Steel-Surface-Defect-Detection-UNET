import os
import numpy as np
from PIL import Image
import pandas as pd
from glob import glob
import random
from torch.utils.data import Dataset
from torchvision import transforms
import torch

train_data_dir = 'severstal-steel-defect-detection/train_images'
test_data_dir = 'severstal-steel-defect-detection/test_images'

def mask_maker(encoded,defect_type,mask,h):

    encoded = encoded.split(" ")
    pts = np.zeros([int(len(encoded)/2),2],dtype=int)
    for i in range(int(len(encoded)/2)):
        pts[i] = [int(encoded[2*i]),int(encoded[2*i+1])]
    
    for i in range(int(len(encoded)/2)):
        initial, increment = pts[i]
        for k in range(increment):
            pt = initial + k - 1
            c = pt//h
            r = pt%h
            mask[r][c] = defect_type
    return mask


class Defect_Dataset(Dataset):
    def __init__(self, root_dir, split, img_h, img_w):
        self.root_dir = root_dir
        self.img_h, self.img_w = img_h, img_w
        self.split = split
        if split in ['train','val']:
            self.path_list = sorted(glob(os.path.join(self.root_dir,"train_images",'*.jpg')))
            self.df = pd.read_csv(os.path.join(root_dir,'train.csv'))
            if split == 'train':
                self.image_files = random.choices(self.path_list, k=int(0.8*len(self.path_list)))
            else:
                self.image_files = random.choices(self.path_list, k=int(0.2*len(self.path_list)))
        elif split == 'test':
            self.df = None
            self.path_list = sorted(glob(os.path.join(self.root_dir,"test_images","*.jpg")))
        else:
            print('Invalid Split')
        self._init_transform()

    def _init_transform(self):
        brightness_changer = transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0)
        rand_changer = transforms.Lambda(lambda x: brightness_changer(x) if random.random() < 0.6 else x)
        self.transform_img = transforms.Compose([rand_changer,
                                            transforms.Resize((self.img_h,self.img_w),interpolation=Image.NEAREST),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.3402,0.3402,0.3402], std=[0.1506,0.1506,0.1506])
                                            ])
        self.transform_mask = transforms.Compose([transforms.Resize((self.img_h,self.img_w),interpolation=Image.NEAREST)])

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = Image.open(image_path).convert('RGB')
        wt, ht = image.size

        mask = np.zeros([ht,wt],dtype=int)
        if self.df is not None:
            for i in range(len(self.df)):
                if self.df['ImageId'].iloc[i] == self.image_files[index].split('/')[2]:
                    defect = self.df['ClassId'].iloc[i]
                    encoded_pixels = self.df['EncodedPixels'].iloc[i]
                    mask = mask_maker(encoded_pixels,defect,mask,ht)
            
            image = self.transform_img(image)
            mask = self.transform_mask(torch.from_numpy(mask).unsqueeze(0))
            return image, torch.squeeze(mask)

        else:
            return image
    
    def __len__(self):
        return len(self.image_files)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torch
    from tqdm import tqdm
    dataset = Defect_Dataset("severstal-steel-defect-detection", "train", 128, 800)
    dataloader = DataLoader(dataset, batch_size=32)

    #checking mean and standard deviation values
    n_images = 0
    mean = 0
    var = 0
    pbar = tqdm(total=len(dataloader))
    for (img, _) in dataloader:
        img = img.view(img.shape[0], img.shape[1], -1)
        n_images += img.size(0)
        mean += img.mean(2).sum(0) 
        var += img.var(2).sum(0)
        pbar.update(1)
    pbar.close()
    mean /= n_images
    var /= n_images
    std = torch.sqrt(var)
    print(mean)
    print(std)

    #checking image and mask size
    img, mask = next(iter(dataloader))
    print(img.shape)
    print(mask.shape)
    print(mask.max())
    print(mask.min())
    print(mask.dtype)
