import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

# Local imports
from unet import UNet
from dataloader import data_loaders
from utils import (
    check_scores, 
    save_checkpoint, 
    save_preds_as_imgs, 
    DiceLoss,
)


def train_model(loader, model, device, optimizer, criterion, scheduler):
    """
    Function perform one epoch on entire dataset and prints loss for each batch.

    Args:
        loader (object): iterable-style dataset.
        model (class): provides with a forward method.
        device (cuda object): cpu or gpu.
        optimizer (torch object): optimization algorithm. 
        criterion (torch oject): loss function with backward method.

    Returns:
        None
    """           

    tqdm_loader = tqdm(loader) # make progress bar
    scaler = torch.cuda.amp.GradScaler() # scaler for loss

    model.train()
    for batch_idx, (data, targets) in enumerate(tqdm_loader):
        # move data and masks to same device as computed gradients
        data = data.to(device=device) 
        targets = targets.float().unsqueeze(1).to(device=device) # add on channel dimension
        
        # mixed precision training
        with torch.cuda.amp.autocast():
            output = model(data) 
            loss = criterion(output, targets)

        # backprop
        optimizer.zero_grad() # zero out previous gradients
        scaler.scale(loss).backward() # scale loss before backprop
        scaler.step(optimizer) # update gradients
        scaler.update() # update scale factor

        # update tqdm loop
        tqdm_loader.set_postfix(loss=loss.item())
        

def train_and_validate():
    config = dict()
    config["lr"] = 1e-4
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    config["load_model"] = False
    config["num_epochs"] = 150
    config["numcl"] = 1
    config["batch_size"] = 32
    config["pin_memory"] = True
    config["num_workers"] = 4
    config["image_height"] = 256
    config["image_width"] = 256
    config["model_name"] = "unet"

    # specify transforms on training set
    train_transforms = A.Compose([
        A.Resize(height=config["image_height"], width=config["image_width"]),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        A.Normalize(
                mean=[0.5246, 0.3068, 0.2230],
                std=[0.3328, 0.2308, 0.1945],
                max_pixel_value=255.0,
        ),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.ColorJitter(p=0.5),
        ], p=1.0),
        A.OneOf([
                A.HorizontalFlip(p=1),
                A.RandomRotate90(p=1),
                A.VerticalFlip(p=1)            
        ], p=1),
        ToTensorV2(),
    ])

    # specify transforms on validation set
    val_transforms = A.Compose(
        [   
            A.Resize(height=config["image_height"], width=config["image_width"]),
            A.Normalize(
                mean=[0.5246, 0.3068, 0.2230],
                std=[0.3328, 0.2308, 0.1945],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # get dataloaders
    train_loader, val_loader, test_loader = data_loaders(
        batch_size=config["batch_size"], 
        train_transforms=train_transforms, 
        val_transforms=val_transforms, 
        num_workers=config["num_workers"], 
        pin_memory=config["pin_memory"]
    )

    # define model
    model = UNet(in_channels=3, out_channels=config["numcl"]).to(config["device"])
    # define loss
    criterion = DiceLoss()
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    # defice learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20, min_lr=1e-6, verbose=True) 



    for epoch in range(config["num_epochs"]):
        # train on training data, prints accuracy and dice score of training data
        train_model(train_loader, model, config["device"], optimizer, criterion, scheduler)

         # check validation loss
        print("------------")
        print("At epoch %d :" % epoch)
        mean_val_loss = check_scores(val_loader, model, config["device"], criterion)

        # save model after training
        if epoch==config["num_epochs"]-1:
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "criterion" : criterion.state_dict(),
                "loss": mean_val_loss
            }
            # change name of file and run in order to save more models
            save_checkpoint(epoch, checkpoint, config["model_name"]+"_checkpoint_10.pt")

        if scheduler is not None:
            scheduler.step(mean_val_loss)

 
if __name__ == "__main__":
    train_and_validate()
