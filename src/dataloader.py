import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import math
import torchvision
from torch.utils.data import DataLoader, random_split
import shutil
#import albumentations as A
#from albumentations.pytorch import ToTensorV2

seed = 24
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class KvasirSEGDataset(Dataset):
    """
    Class provides with image and mask, or alternatively
    give an transformed/augmented version of these.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir) # list containing the names of the entries in the directory

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        filename = self.images[index]

        if self.mask_dir is not None:
            img_path = os.path.join(self.image_dir, filename)
            mask_path = os.path.join(self.mask_dir, filename)
            image = np.array(Image.open(img_path, "r").convert("RGB"))
            mask = np.array(Image.open(mask_path, "r").convert("L"), dtype = np.float32) # greyskale = L in PIL
            mask[mask==255.0] = 1.0 # 255 decimal code for white, change this to 1 due to sigmoid on output.

            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]

            return image, mask

        else:
            img_path = os.path.join(self.image_dir, filename)
            image = np.array(Image.open(img_path, "r").convert("RGB"))
           
            if self.transform is not None:
                augmentations = self.transform(image=image)
                image = augmentations["image"]

            return image
    

        

def move_images():
    """
    Function takes dataset containing image and corresponding mask, 
    and splits into folders for train, val and test data.

    Mask musth have equal filename as its corresponding image.
    """
    data_set = KvasirSEGDataset("/home/feliciaj/data/MediaEval/development/images/", "/home/feliciaj/data/MediaEval/development/masks/", transform=None)

    base_path = "/home/feliciaj/data/MediaEval/development/"

    dirs = [
        base_path+"train/train_images/", 
        base_path+"train/train_masks/", 
        base_path+"val/val_images/", 
        base_path+"val/val_masks/"
    ]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    train_frac = 0.8
    val_frac = 0.2

    N = len(data_set) # number of images
    perm = np.random.permutation(N)
    filenames = np.array(os.listdir(base_path + "images/"))[perm] # shuffle filenames for randomness

    for i, f in enumerate(filenames):
        if (i < train_frac * N):
            middle_path = "train/train_"
        else:
            middle_path = "val/val_"

        # copy images
        shutil.copy(
            base_path + "images/" + f, 
            base_path + middle_path + "images/" + f
        )

        # copy masks
        shutil.copy(
            base_path + "masks/" + f, 
            base_path + middle_path + "masks/" + f
        )


def data_loaders(batch_size, train_transforms, val_transforms, num_workers, pin_memory):
    """
    Get dataloaders of train, validation and test dataset. 
    Val transfomrs are also used on test data.

    """
    train_img_dir = "/home/feliciaj/data/MediaEval/development/train/train_images"
    train_mask_dir = "/home/feliciaj/data/MediaEval/development/train/train_masks"
    val_img_dir = "/home/feliciaj/data/MediaEval/development/val/val_images"
    val_mask_dir = "/home/feliciaj/data/MediaEval/development/val/val_masks"
    test_img_dir = "/home/feliciaj/data/MediaEval/test/images/"

    train_ds = KvasirSEGDataset(
        image_dir = train_img_dir,
        mask_dir = train_mask_dir,
        transform = train_transforms,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = True,
    )

    val_ds = KvasirSEGDataset(
        image_dir = val_img_dir,
        mask_dir = val_mask_dir,
        transform = val_transforms,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = False,
        shuffle = True,
    )
    
    test_ds = KvasirSEGDataset(
        image_dir = test_img_dir,
        mask_dir = None,
        transform = val_transforms
    )

    test_loader = DataLoader(
        test_ds,
        batch_size = None,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = False,
    ) 

    return train_loader, val_loader, test_loader




def convert_png_to_jpg(driectory):
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            original = os.path.join(directory, filename)
            renamed = os.path.join(directory, filename.replace(".png",".jpg"))
            os.rename(original, renamed)
            continue
        else:
            continue



def get_mean_std(loader):
    """
    Credit:
    https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_std_mean.py
    
    Works for images with and without color channels.
    
    """

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0,2,3]) # don't sum across channel dim
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches-mean**2)**0.5

    return mean, std



if __name__ == "__main__":
    #convert_png_to_jpg(r"/home/feliciaj/data/MediaEval/development/masks/") # only run this once
    #move_images() # only run this once

    train, val, test = data_loaders(None, None, None, 1, False)

    #print(torch.from_numpy(test.dataset[1].permute(2, 0, 1)).shape)

    #image = torch.from_numpy(test.dataset[0])
    #image2 = torch.swapaxes(image, 2, 0)
    #image3 = torch.swapaxes(image2, 1, 2).float()

    #torchvision.utils.save_image(image3, f"/home/feliciaj/MediaEval/ok.png")
    #print(len(train)) # 1090
    #print(len(val)) # 272
    #print(len(test)) # 200
    """
    train_loader, val_loader, test_loader = data_loaders(
        64, 
        train_transforms=A.Compose([A.Resize(height=256, width=256),A.Normalize(mean=0.0,std=1.0,max_pixel_value=255.0,),ToTensorV2()]), 
        val_transforms=None, 
        num_workers=1, 
        pin_memory=False
    )
    """

    #print(get_mean_std(train_loader)) (tensor([0.5246, 0.3068, 0.2230]), tensor([0.3328, 0.2308, 0.1945]))
    


    