import torch
import torch.nn as nn
import torchvision
import os
import time
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataloader import data_loaders



def save_and_get_time(filenames, loader, model, folder, device):
    """
    Save test images with same filname as input image. 

    Outputs filename and corresponding time for model to generate its mask.

    Args:
        filenames (list): list containing filenames, must have the same order as loader.
        loader (pytorch object): dataloader.
        model (object): model which outputs unsigmoided predictions.
        folder (string): directory to folder of where to store imgs.
        device (pytorch object): where to store images on.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    # split filenames into name and extension
    names = []
    fn_extensions = []
    for filename in filenames:
        names.append(filename.split(".",1)[0])
        fn_extensions.append(filename.split(".",1)[0]+".png")
    filenames = names

    times = []
    model.eval()
    for idx, x in enumerate(loader):
        start_time = time.time()
        x = x.to(device=device).unsqueeze(0) # add batch dim
        with torch.no_grad():
            mask_pred = torch.sigmoid(model(x)[0]) # only extract class prob
            mask_pred = (mask_pred > 0.5).float() * 255.0 # threshold
        end_time = time.time() - start_time
        times.append(end_time)
        torchvision.utils.save_image(mask_pred, f"{folder}/{filenames[idx]}.png")
        #print("%s.jpg => %f seconds" % (filenames[idx], end_time))
    data = {"Filenames": fn_extensions, "Time": times}
    df = pd.DataFrame(data)
    df.to_csv("times_per_pred.csv", index=False)
    model.train()


def save_checkpoint(epoch, state, filename):
    print("Epoch %d => Saving checkpoint" % epoch)
    torch.save(state, "/home/feliciaj/MediaEval/saved_models/"+filename)


def check_scores(loader, model, device, criterion):
    """
    Validate for one epoch. Prints accuracy, Dice/F1 and IoU score.

    Args:
        loader (object): iterable-style dataset.
        model (class): provides with a forward method.
        device (cuda object): cpu or gpu.
        criterion (function): scoring function.

    Returns:
        Mean loss over training data.
    """
    num_correct = 0
    num_pixels = 0
    dice = 0
    iou = 0
    loss = []

    model.eval()
    with torch.no_grad():
        for batch, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            prob = model(x)
            pred = torch.sigmoid(prob)
            pred = (pred > 0.5).float()
            num_correct += (pred == y).sum()
            num_pixels += torch.numel(pred)
            dice += dice_coef(pred, y)
            iou += iou_score(pred, y)
            loss.append(criterion(prob, y).item())
    
    print(f"Accuracy: {num_correct}/{num_pixels} or {num_correct/num_pixels*100:.2f} percent")
    print(f"IoU score: {iou/len(loader)}")
    print(f"Dice score: {dice/len(loader)}")
    
    model.train()

    return sum(loss)/len(loader)


class DiceLoss(nn.Module):
    """
    Args:
        weight: An array of shape [num_classes,]
        input: A tensor of shape [N, num_classes, *]
        target: A tensor of shape same with input

    Returns:
        1D tensor
    """
    def __init__(self, weight=None):
        super().__init__()
    
    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "Prediction and GT batch size do not match"
        pred = torch.sigmoid(input).view(-1)
        truth = target.view(-1)
        
        return (1 - dice_coef(pred, truth))



def dice_coef(pred, target):
    """
    Dice coefficient used as metric.

    Args:
        pred (tensor): 1D tensor containing predicted (sigmoided + thresholded) pixel values.
        target (tensor): 1D tensor contining grund truth pixel values.
    Returns:
        1D tensor
    """
    intersection = (pred*target).sum()
    union = pred.sum() + target.sum()

    return (2.0 * intersection) / (union+intersection)



def iou_score(pred, target):
    """
    IoU (Intersect over Union) used as metric.

    Args:
        pred (tensor): 1D tensor containing predicted (sigmoided + thresholded) pixel values.
        target (tensor): 1D tensor contining grund truth pixel values.
    Returns:
        1D tensor
    """
    intersection = (pred*target).double().sum()
    union = target.double().sum() + pred.double().sum() 

    return (intersection + 1) / (union + 1)

