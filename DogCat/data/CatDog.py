# -*- coding: UTF-8 -*-
'''
@Project ： 实验二猫狗分类
@File    ：CatDog.py
@Author  ：郑家祥
@Date    ：2021/6/3 9:11 
@Description：声明CatDog类，读取原始数据集的标签制作成训练、验证、测试数据集
'''
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CatDog(Dataset):

    def __init__(self, root, transform=None, train=True, test=False):
        '''
        描述：获取训练集与测试集，并将原始训练集按照7:3划分为新的训练集与测试集
        '''
        self.imgs = [os.path.join(root, img) for img in os.listdir(root)]
        self.train = train
        self.test = test
        if transform is None:
            if self.test or not train:
                self.transform = transforms.Compose([
                    transforms.Resize(size=(224,224)),
                    transforms.CenterCrop(size=(224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(size=(256, 256)),
                    transforms.RandomResizedCrop(size=(224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    def __getitem__(self, index):
        """
        描述：一次返回一张图片的标签和数据，能够通过索引取值
        """
        img_path = self.imgs[index]
        label =  1 if 'dog' in img_path.split('.')[0] else 0
        data = Image.open(img_path)
        data = self.transform(data)
        return data, label

    def __len__(self):
        """
        描述：返回图片的数量
        """
        return len(self.imgs)