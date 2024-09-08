import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage

def dice_coeff(seg,target,smooth=0.001):
    intersection=np.sum(seg*target)
    dice=(2*intersection+smooth)/(np.sum(seg)+np.sum(target)+smooth)
    return dice


def print_log(print_string,log):
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()


def resize_3d(image, target_depth, target_height, target_width):
    depth, height, width = image.shape
    zoom_factors = (target_depth / depth, target_height / height, target_width / width)
    resized_image = scipy.ndimage.zoom(image, zoom_factors, order=1)  # order=1 for bilinear interpolation
    return resized_image


def validate(model,dataloader, slices=20):
    device='cuda'
    model.eval()
    total_dice=np.zeros(2)
    c=0
    criterion=torch.nn.CrossEntropyLoss()
    loss=0
    for batch_num,data in enumerate(dataloader):
        img,mask,mask_onehot,length=data['im'],data['mask'],data['m'],data['length']
        #print(length)
        img=img.to(device).squeeze(0)[:length[0],:,:,:]
        #print(img.shape)
        mask=mask.to(device).squeeze(0)[:length[0],:,:]
        #print(mask.shape)
        mask_onehot=mask_onehot.to(device).squeeze(0)[:length[0],:,:,:]
        pred_raw=model(img)
        pred=F.softmax(pred_raw,dim=1)

        tmp_loss=criterion(pred_raw,mask)
        loss+=tmp_loss.item()

        pred_np=pred.detach().cpu().numpy()
        mask_onehot_np=mask_onehot.detach().cpu().numpy()

        pred_np=np.moveaxis(pred_np,1,-1)
        mask_onehot_np=np.moveaxis(mask_onehot_np,1,-1)
        pred_onehot_np=np.zeros_like(pred_np)

        pred_np=np.argmax(pred_np,axis=-1)
        for i in range(128):
            for j in range(128):
                for k in range(slices):
                    pred_onehot_np[k,i,j,pred_np[k,i,j]]=1
        for i in range(2):
            total_dice[i]+=dice_coeff(pred_onehot_np[:,:,:,i:i+1],mask_onehot_np[:,:,:,i:i+1])
        c+=1

    return total_dice/c,loss/c