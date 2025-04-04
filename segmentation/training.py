import torch
import wandb
from tqdm import tqdm
from monai.metrics import DiceMetric
import torch.nn.functional as F


def train(model, epochs, train_dataloader, validation_loader, optimizer,loss_function,n,early_stopping, device):
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)#.to(device)
    for epoch in tqdm(range(epochs)):
        # training
        model.train()    
        epoch_loss = 0
        step = 0
        for batch_data in train_dataloader: 
            step += 1
            optimizer.zero_grad()
            outputs = model(batch_data["img"].float().to(device))
            loss = loss_function(outputs, batch_data["mask"].to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss/step

        # validation
        step = 0
        val_loss = 0
        correct_pixels = 0
        total_pixels = 0
        dice_scores = []

        ## Evaluation
        dice_metric.reset()
        model.eval()
        with torch.no_grad():
            for batch_data in validation_loader:
                step += 1

                # Forward pass
                outputs = model(batch_data['img'].float().to(device))
                loss = loss_function(outputs, batch_data['mask'].to(device))
                val_loss += loss.item()

                # Convert outputs to binary mask (threshold at 0.5)
                preds = torch.sigmoid(outputs) > 0.5  # Convert logits to binary (0 or 1)

                # Compute pixel-wise accuracy
                correct_pixels += (preds == batch_data["mask"].to(device)).sum().item()
                total_pixels += batch_data["mask"].numel()  # Total number of pixels
                
                # Dice Score (per batch, mean over batch dimension)
                dice_metric(preds.float(), batch_data['mask'].to(device))


            val_loss = val_loss / step
            accuracy = correct_pixels / total_pixels  # Compute accuracy
            mean_dice = dice_metric.aggregate().item()
            log_to_wandb(epoch, train_loss, val_loss, accuracy,mean_dice, batch_data, outputs)
            # Early stopping (regularization)
            if epoch >= 40: 
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print("🛑 Early stopping triggered.")
                    break

    return

def log_to_wandb(epoch, train_loss,val_loss,accuracy,mean_dice, batch_data, outputs):
    """ Function that logs ongoing training variables to W&B """

    # Create list of images that have segmentation masks for model output and ground truth
    log_imgs = [wandb.Image(img, masks=wandb_masks(mask_output, mask_gt)) for img, mask_output,
                mask_gt in zip(batch_data['img'], outputs, batch_data['mask'])]

    # Send epoch, losses and images to W&B
    wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'pixelaccuracy': accuracy,'mean_dice': mean_dice, 'results': log_imgs})
    
def wandb_masks(mask_output, mask_gt):
    """ Function that generates a mask dictionary in format that W&B requires """

    # Apply sigmoid to model ouput and round to nearest integer (0 or 1)
    sigmoid = torch.nn.Sigmoid()
    mask_output = sigmoid(mask_output)
    mask_output = torch.round(mask_output)

    # Transform masks to numpy arrays on CPU
    # Note: .squeeze() removes all dimensions with a size of 1 (here, it makes the tensors 2-dimensional)
    # Note: .detach() removes a tensor from the computational graph to prevent gradient computation for it
    mask_output = mask_output.squeeze().detach().cpu().numpy()
    mask_gt = mask_gt.squeeze().detach().cpu().numpy()

    # Create mask dictionary with class label and insert masks
    class_labels = {1: 'vessels'}
    masks = {
        'predictions': {'mask_data': mask_output, 'class_labels': class_labels},
        'ground truth': {'mask_data': mask_gt, 'class_labels': class_labels}
    }
    return masks

