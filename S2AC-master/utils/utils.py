# -*- coding: utf-8 -*-
"""
@ project: S2AC
@ author: wzy
@ file: utils.py
@ time: 2023/10/10 10:28
"""
# -*- coding: utf-8 -*-

import torch
from torch.utils import data
from torchvision import datasets, transforms
import pickle

# office path
Amazon_ImagePath = 'E:/pyworkspace/deep-coral-master/data/office31/amazon/images'
RealWorld_ImagePath = 'E:/pyworkspace/deep-coral-master/data/OfficeHomeDataset_10072016/Real World'
ImageNettrainpath='E:/pyworkspace/JigsawPuzzlePytorch-master/Dataset/imagenet/ILSVRC2012_img_train'
ImageNet_train_process='E:/pyworkspace/JigsawPuzzlePytorch-master/Dataset/imagenet/ILSVRC_train'
ImageNetvalpath='E:/Kaggle/input/imagenet-mini/val'
ImageNet_val_process='E:/pyworkspace/JigsawPuzzlePytorch-master/Dataset/imagenet/ILSVRC_val'
zoom_blur='E:/Kaggle/input/ImageNet-C/blur~/zoom_blur'
saturate='E:/Kaggle/input/ImageNet-C/extra~/saturate'
spatter='E:/Kaggle/input/ImageNet-C/extra~/spatter'
speckle_noise='E:/Kaggle/input/ImageNet-C/extra~/speckle_noise'
brightness='E:/Kaggle/input/ImageNet-C/weather~/brightness'
defocus_blur='E:/Kaggle/input/ImageNet-C/blur~/defocus_blur/5'
elastic_transform='E:/Kaggle/input/ImageNet-C/digital~/elastic_transform'
snow='E:/Kaggle/input/ImageNet-C/weather~/snow'
shot_noise='E:/Kaggle/input/ImageNet-C/noise~/shot_noise'
pixelate='E:/Kaggle/input/ImageNet-C/digital~/pixelate'
motion_blur='E:/Kaggle/input/ImageNet-C/blur~/motion_blur'
jpeg_compression='E:/Kaggle/input/ImageNet-C/digital~/jpeg_compression'
impulse_noise='E:/Kaggle/input/ImageNet-C/noise~/impulse_noise'
fog='E:/Kaggle/input/ImageNet-C/weather~/fog'
frost='E:/Kaggle/input/ImageNet-C/weather~/frost'
glass_blur='E:/Kaggle/input/ImageNet-C/blur~/glass_blur'
gaussian_noise='E:/Kaggle/input/ImageNet-C/extra~/gaussian_blur'
contrast='E:/Kaggle/input/ImageNet-C/digital~/contrast'
Imagenetmini='E:/Kaggle/input/imagenet-mini/train'
def get_data_mean_and_std(path):
    dataset = datasets.ImageFolder(path,
                                   transform=transforms.ToTensor())
    dataloader = data.DataLoader(dataset,batch_size=1)
    mean = [0,0,0]
    std = [0,0,0]
    print(len(dataset))
    for i in range(3):
        mean_every = 0
        std_every = 0
        for _,(xs,_) in enumerate(dataloader):
            img = xs[0][i].numpy()
            mean_every += img.mean()
            std_every += img.std()
        mean[i] = mean_every/len(dataset)
        std[i] = std_every/len(dataset)
    return mean,std
# x,y = get_data_mean_and_std(ImageNetvalpath)
# print(x,y)
def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        print('[INFO] Object saved to {}'.format(path))