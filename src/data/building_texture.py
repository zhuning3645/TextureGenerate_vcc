import json
import random
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import os, sys
import math
import json
import importlib
from pathlib import Path

import cv2
import random
from PIL import Image
import webdataset as wds
import pytorch_lightning as pl

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from src.utils.train_util import instantiate_from_config



class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size=8, 
        num_workers=4, 
        train=None, 
        validation=None, 
        test=None, 
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs['train'] = train
        if validation is not None:
            self.dataset_configs['validation'] = validation
        if test is not None:
            self.dataset_configs['test'] = test
    
    def setup(self, stage):

        if stage in ['fit']:
            self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
            print("Loaded datasets:", self.datasets.keys())
        else:
            raise NotImplementedError

    def train_dataloader(self):

        sampler = DistributedSampler(self.datasets['train'])
        return wds.WebLoader(self.datasets['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):

        sampler = DistributedSampler(self.datasets['validation'])
        return wds.WebLoader(self.datasets['validation'], batch_size=1, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def test_dataloader(self):

        return wds.WebLoader(self.datasets['test'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='building_texture/',
        meta_fname='texture_refence.json',
        input_image_dir='normals',          # 法线贴图目录
        ref_dir='ref_crops',           # 裁剪参考图目录
        rgb_dir='rgb_images',          # 目标RGB目录
        img_size=768,
        num_ref=3,                     # 参考图数量
        validation=False,
    ):
        self.root_dir = Path(root_dir)
        self.input_image_dir = input_image_dir
        self.ref_dir = ref_dir
        self.rgb_dir = rgb_dir
        self.num_ref = num_ref
        self.img_size = img_size

        # 加载元数据
        with open(os.path.join(root_dir, meta_fname)) as f:
            filtered_dict = json.load(f)
        self.paths = filtered_dict['good_objs']
        
        # 图像预处理
        self.transform = T.Compose([
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
            #T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
        ])

        print(f'Dataset initialized with {len(self.paths)} samples')

    def __len__(self):
        return len(self.paths)

    def load_image(self, path):
        """加载图像并应用预处理"""
        img = Image.open(path).convert('RGB')
        return self.transform(img)

    def __getitem__(self, index):
        while True:
            try:
                uid = self.paths[index]
                
                # 加载主法线贴图
                main_normal_path = os.path.join(
                    self.root_dir, self.input_image_dir, uid, 'main_normal.png'
                )
                main_normal = self.load_image(main_normal_path)

                # 加载参考法线贴图（随机选择1个裁剪）
                ref_dir = self.root_dir / self.ref_dir / uid
                ref_files = [f for f in ref_dir.iterdir() if f.is_file() and f.suffix == '.png']  # 过滤PNG文件
                selected_ref = random.choice(ref_files)

                ref_normals = self.load_image(selected_ref)
                
                # 加载目标RGB图像
                rgb_path = os.path.join(
                    self.root_dir, self.rgb_dir, uid, 'target_rgb.png'
                )
                target_rgb = self.load_image(rgb_path)

                
                return {
                    'input_images': main_normal,      # [3, H, W]
                    'ref_normals': ref_normals,      # [3, H, W] 
                    'render_gt': target_rgb         # [3, H, W]
                }
                
            except Exception as e:
                print(f"Error loading {self.paths[index]}: {e}")
                index = random.randint(0, len(self.paths)-1)
                continue

class ValidationData(Dataset):
    def __init__(self,
        root_dir='building_texture/',
        input_image_size=768,

    ):
        self.root_dir = Path(root_dir)
        self.input_image_size = input_image_size

        # 获取所有图片文件路径（假设目录下直接存储图片）
        self.paths = sorted([
            os.path.join(self.root_dir, f) 
            for f in os.listdir(self.root_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        print('============= length of dataset %d =============' % len(self.paths))

    def __len__(self):
        return len(self.paths)

    def load_im(self, path, color):
        '''加载单张图片并处理透明度'''
        pil_img = Image.open(path)
        pil_img = pil_img.resize(
            (self.input_image_size, self.input_image_size), 
            resample=Image.BICUBIC
        )

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        if image.shape[-1] == 4:
            alpha = image[:, :, 3:]
            image = image[:, :, :3] * alpha + color * (1 - alpha)
        else:
            alpha = np.ones_like(image[:, :, :1])

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha
    
    def __getitem__(self, index):
        # 直接加载单张图片路径
        input_image_path = self.paths[index]
        
        # 背景颜色（默认白色）
        bkg_color = [1.0, 1.0, 1.0]

        # 加载单张图片
        image, alpha = self.load_im(input_image_path, bkg_color)
        
        # 添加批次维度以保持接口兼容性
        data = {
            'input_images': image.unsqueeze(0),  # [1, C, H, W]
            'input_alphas': alpha.unsqueeze(0)   # [1, 1, H, W]
        }
        return data
