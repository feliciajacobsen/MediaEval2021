import torch
import torch.nn as nn
import torchvision
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# local imports
from unet import UNet
from dataloader import data_loaders
from utils import save_and_get_time

#import matplotlib.pyplot as plt


class MyEnsemble(nn.Module):
    """
    Ensemble of pretrained models.

    Args:
        model A-E (.pt-file): files of saved models.
        device (string): device to get models from.

    Returns:
        mean_pred (tensor): mean predicted mask by ensemble models of size (B,C,H,W).
        variance (tensor): normalized variance tensor of predicted mask of size (B,C,H,W).
    """
    def __init__(self, modelA, modelB, modelC, modelD, modelE, device):
        super(MyEnsemble, self).__init__()

        self.modelA = modelA.to(device)
        self.modelB = modelB.to(device)
        self.modelC = modelC.to(device)
        self.modelD = modelD.to(device)
        self.modelE = modelE.to(device)
    
    def forward(self, x):
        shape = x.shape
        x1 = self.modelA(x.clone()) # pred from model A
        x2 = self.modelB(x.clone()) # pred from model B
        x3 = self.modelC(x.clone()) # pred from model C
        x4 = self.modelD(x.clone()) # pred from model D
        x5 = self.modelE(x.clone()) # pred from model E
        
        outputs = torch.stack([x1, x2, x3, x4, x5])
        
        mean = torch.mean(outputs, dim=0).double() # element wise mean from outout of ensemble models

        pred = torch.sigmoid(outputs)
        mean_pred = torch.sigmoid(mean).float() # only extract class prob

        #variance = torch.mean((pred**2 - mean_pred).pow(2), dim=0).double()

        variance = torch.mean((pred**2 - mean_pred), dim=0).double()
        
        normalized_variance = (variance - torch.mean(variance,dim=0)) / (torch.std(variance, dim=0).double())

        return mean, variance 


def validate_ensembles():
    """
    Function loads trained models and make prediction on data from loader.
    Only supports for ensemble_size=5 and model="unet" for now.

    """

    test_folder = "/home/feliciaj/MediaEval/ensembles/medico2021images"
    pred_folder = "/home/feliciaj/MediaEval/ensembles/medico2021masks/" 
    uncertainty_folder = "/home/feliciaj/MediaEval/ensembles/medico2021uncertainty" 
    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)
    if not os.path.exists(uncertainty_folder):
        os.makedirs(uncertainty_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = UNet(in_channels=3, out_channels=1)
    ensemble_size = 5

    val_transforms = A.Compose(
        [   
            A.Normalize(
                mean=[0.5568, 0.3221, 0.2368],
                std=[0.3191, 0.2220, 0.1878],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    train_loader, val_loader, test_loader = data_loaders(
        batch_size=32, 
        train_transforms=val_transforms, 
        val_transforms=val_transforms, 
        num_workers=4, 
        pin_memory=True
    )

     
    model_list = []
    for i in range(ensemble_size):
        model_list.append(model)
    
    main_path = "/home/feliciaj/MediaEval/saved_models/" # dir folder of saved models
    paths = os.listdir(main_path)[:ensemble_size] # list of elements in folder

    assert len(paths) == ensemble_size, "No. of folder elements does not match ensemble size"
    
    # load models
    for model, path in zip(model_list, paths):
        checkpoint = torch.load(main_path + path)
        model.load_state_dict(checkpoint["state_dict"])
    
    model.eval()

    model = MyEnsemble(
        model_list[0], model_list[1], model_list[2], model_list[3], model_list[4], device=device
    )

    filenames = os.listdir(test_folder)
    with torch.no_grad():
        for (idx, x), filename in zip(enumerate(test_loader), filenames):
            filename = filename.split(".",1)[0] 
            x = x.to(device=device).unsqueeze(0)
            prob, variance = model(x)
            variance = variance.cpu().detach().squeeze(0).squeeze(0)
            #plt.imsave(f"{uncertainty_folder}/{filename}.png", variance, cmap="jet")

            #img = Image.open(f"{test_folder}/{filename}.jpg", "r").convert("RGB")
            #mask = Image.open(f"{pred_folder}{filename}.png", "r").convert("RGB")
            #heatmap = Image.open(f"{uncertainty_folder}/{filename}.png", "r").convert("RGB")
            
        
            #print(img.size, heatmap.size)
            
            #res = Image.blend(img, heatmap, alpha=0.2)
            #plt.imsave(f"/home/feliciaj/MediaEval/ensembles/hei/{filename}.png", res)

    model.train()
    save_and_get_time(filenames, test_loader, model, pred_folder, device=device)


if __name__ == "__main__":
    validate_ensembles()