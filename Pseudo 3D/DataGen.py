import torch
from torchvision import transforms
import tifffile as tiff
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F
from scripts.helpers import *


transform = transforms.Compose([
    transforms.ToTensor(),
])

class Data_Gen(Dataset):
    def __init__(self, data_path, label_path, test_ids, transform=transform, mode = 'train'):
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.mode = mode
        if mode=='train':
            self.images = [img for img in os.listdir(data_path) if img[:4] not in test_ids]
            self.labels = [img for img in os.listdir(label_path) if img[:4] not in test_ids]# Print the lengths for debugging
        else:
            self.images = [img for img in os.listdir(data_path) if img[:4] in test_ids]
            self.labels = [img for img in os.listdir(label_path) if img[:4] in test_ids]# Print the lengths for debugging
        print(f"Number of images: {len(self.images)}")
        print(f"Number of labels: {len(self.labels)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #print(idx)
        img_name = os.path.join(self.data_path, self.images[idx])
        #print(img_name)
        label_name = os.path.join(self.label_path, self.labels[idx])
        image = tiff.imread(img_name)
        image = np.float32(image)
        if len(image.shape)==3:        
           image = image.squeeze(0)
        else:
            image = image
        image = image/255
        #print(image.max())
        label = tiff.imread(label_name)
        label = np.float32(label)
        #print(label.shape)


        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
            if label.shape[0]==128:
                label = label.squeeze(1)
                label = label.permute(1,0)
            else:
                label = label.squeeze(0)
            
            
                #label = label.permute(1,0)
            label[label==255] = 1 
            label[label ==128] = 2
            label[label<0] = 0
            #label = label.permute(1,0)
            #print(label.max(), image.max())
            #print(image.shape, label.shape)
        return image.cuda(), label.cuda(), self.images[idx], self.labels[idx]



class CustomTestDataset(Dataset):
    def __init__(self, data_path, label_path, test_ids, transform=None):
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.images = [
            img for img in os.listdir(data_path)
            if img[:3] in test_ids and not any(sub in img for sub in ['horizontal', 'vertical', 'shift', 'rotation'])
        ]
        self.labels = [
            img for img in os.listdir(label_path)
            if img[:3] in test_ids and not any(sub in img for sub in ['horizontal', 'vertical', 'shift', 'rotation'])
        ]
        #print(f"Number of images: {len(self.images)}")
        #print(f"Number of labels: {len(self.labels)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #print(idx)
        img_name = os.path.join(self.data_path, self.images[idx])
        label_name = os.path.join(self.label_path, self.labels[idx])
    
        image = tiff.imread(img_name)
        image = np.float32(image)
        if len(image.shape)==3:        
           image = image.squeeze(0)
        else:
            image = image
        image = image/255
        #print(image.max())
                
        #print(label_name)
        
        label = tiff.imread(label_name)
        label = np.float32(label)
        #print(label.shape)


        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
            if label.shape[0]!=128:
                label = label.squeeze(0)
                label = label.permute(1,0)
            else:
                label = label.squeeze(1)
            
            
                #label = label.permute(1,0)
            label[label==255] = 1  # Assuming the class values are 255, 128, 0 initially
            label[label==128] = 2
            label[label<0] = 0
            label = label.permute(1,0)
            #print(label.max(), image.max())
            #print(image.shape, label.shape)
        return image.cuda(), label.cuda(), self.images[idx], self.labels[idx]


class Data_Gen_3D(Dataset):
    def __init__(self, data_path, label_path, test_ids, transform=transform, mode = 'train'):
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.mode = mode
        if mode=='train':
            self.images = [img for img in os.listdir(data_path) if img[:2] not in test_ids]
            self.labels = [img for img in os.listdir(label_path) if img[:2] not in test_ids]# Print the lengths for debugging
        else:
            self.images = [img for img in os.listdir(data_path) if img[:2] in test_ids]
            self.labels = [img for img in os.listdir(label_path) if img[:2] in test_ids]# Print the lengths for debugging

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        #print(idx)
        img_name = os.path.join(self.data_path, self.images[idx])
        #print(img_name)
        label_name = os.path.join(self.label_path, self.labels[idx])
        image = tiff.imread(img_name)
        image = np.float32(image)
        image = image/255
        image = image.transpose(1,2,0)
        #print(image.max())
        label = tiff.imread(label_name)
        label = np.float32(label)
        label = label.transpose(1, 2, 0)
        #label = label.unsqueeze(0)
        #print(label.shape)


        if self.transform:
            image = self.transform(image)
            image = image.unsqueeze(0)
            label = self.transform(label)
            
            
                #label = label.permute(1,0)
            label[label==255] = 1 
            label[label ==128] = 2
            label[label<0] = 0
            #label = label.permute(1,0)
            #print(label.max(), image.max())
            #print(image.shape, label.shape)
        return image.cuda(), label.cuda(), self.images[idx], self.labels[idx]


class Data_Gen_3D_Res(Dataset):
    def __init__(self, data_path, label_path, test_ids, transform=transform, mode='train'):
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.mode = mode
        self.target_depth = 16
        self.target_height = 128
        self.target_width = 128
        if mode == 'train':
            self.images = [img for img in os.listdir(data_path) if img[:2] not in test_ids]
            self.labels = [img for img in os.listdir(label_path) if img[:2] not in test_ids]
        else:
            self.images = [img for img in os.listdir(data_path) if img[:2] in test_ids]
            self.labels = [img for img in os.listdir(label_path) if img[:2] in test_ids]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_path, self.images[idx])
        label_name = os.path.join(self.label_path, self.labels[idx])
        image = tiff.imread(img_name)
        image = np.float32(image)
        image = image / 255
        label = tiff.imread(label_name)
        label = np.float32(label)

        # Resize the image and label
        if image.shape[0] != self.target_depth:
            image = resize_3d(image, self.target_depth, self.target_height, self.target_width)
        if label.shape[0] != self.target_depth:
            label = resize_3d(label, self.target_depth, self.target_height, self.target_width)

        if self.transform:
            image = self.transform(image)
            image = image.permute(1,2,0)
            image = image.unsqueeze(0)
            label = self.transform(label)
            label = label.permute(1,2,0)

        label[label == 255] = 1
        label[label == 128] = 2
        label[label < 0] = 0

        return image.cuda(), label.cuda(), self.images[idx], self.labels[idx]
