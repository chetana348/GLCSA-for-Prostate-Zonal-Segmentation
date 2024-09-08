import torch
from torchvision import transforms
import tifffile as tiff
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os


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
            self.images = [img for img in os.listdir(data_path) if img[:3] not in test_ids]
            self.labels = [img for img in os.listdir(label_path) if img[:3] not in test_ids]# Print the lengths for debugging
        else:
            self.images = [img for img in os.listdir(data_path) if img[:3] in test_ids]
            self.labels = [img for img in os.listdir(label_path) if img[:3] in test_ids]# Print the lengths for debugging
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

