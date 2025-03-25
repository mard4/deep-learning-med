import torch
import monai
import numpy as np
import matplotlib.pyplot as plt



# Evaluate the trained network
def visual_evaluation(sample, model, device):
    """
    Visual inspection of a sample, showing:
    - The grayscale X-ray image.
    - The ground truth segmentation mask (in green).
    - The network's predicted segmentation map (in red).
    """
    model.eval()
    inferer = monai.inferers.SlidingWindowInferer(roi_size=[256, 256])
    discrete_transform = monai.transforms.AsDiscrete(logit_thresh=0.5, threshold_values=True)
    sigmoid = torch.nn.Sigmoid()

    with torch.no_grad():
        img_tensor = sample['img'].float().to(device)  # Ensure float32 and move to GPU
        output = model(img_tensor)  # Get model output
        output = sigmoid(output).cpu().squeeze().numpy()  # Apply sigmoid, move to CPU, and convert to NumPy
        output = discrete_transform(output)  # Threshold the output

    # Extract image and mask (ensure they are 2D)
    img = sample["img"].squeeze().cpu().numpy()  # Convert image to NumPy, shape -> [512, 512]
    mask = sample['mask'].squeeze().cpu().numpy()  # Convert mask to NumPy, shape -> [512, 512]

    # Prepare overlays (ensure they are 2D)
    overlay_mask = np.ma.masked_where(mask == 0, mask)  # Remove extra dimensions
    overlay_output = np.ma.masked_where(output < 0.1, output)  # Ensure it's 2D

    # Plot results
    fig, ax = plt.subplots(1, 3, figsize=[15, 5])
    
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('X-ray Image')

    ax[1].imshow(img, cmap='gray')
    ax[1].imshow(overlay_mask, cmap='Greens', alpha=0.7)  # Green ground truth overlay
    ax[1].set_title('Ground Truth')

    ax[2].imshow(img, cmap='gray')
    ax[2].imshow(overlay_output, cmap='Reds', alpha=0.7)  # Red prediction overlay
    ax[2].set_title('Model Prediction')

    plt.show()

# Compute evaluation metrics
def compute_metric(dataloader, model, metric_fn, device):
    """
    This function computes the average value of a metric for a data set.
    
    Args:
        dataloader (monai.data.DataLoader): dataloader wrapping the dataset to evaluate.
        model (torch.nn.Module): trained model to evaluate.
        metric_fn (function): function computing the metric value from two tensors:
            - a batch of outputs,
            - the corresponding batch of ground truth masks.
        
    Returns:
        (float) the mean value of the metric
    """
    model.eval()
    inferer = monai.inferers.SlidingWindowInferer(roi_size=[256, 256])
    discrete_transform = monai.transforms.AsDiscrete(threshold=0.5)
    Sigmoid = torch.nn.Sigmoid()
    
    mean_value = []
    
    for sample in dataloader:
        with torch.no_grad():
            output = discrete_transform(Sigmoid(inferer(sample['img'].to(device), network=model).cpu()))
            metric_value = metric_fn(output, sample["mask"]) # this returns a tensor
            
            mean_value.append(metric_value) # store tensors
    
    mean_value = torch.cat(mean_value).mean().item()
    
    return mean_value


def visual_evaluation_nomask(sample, model, device):
    """
    Visual inspection of a sample, showing:
    - The grayscale X-ray image.
    - The ground truth segmentation mask (in green).
    - The network's predicted segmentation map (in red).
    """
    model.eval()
    inferer = monai.inferers.SlidingWindowInferer(roi_size=[256, 256])
    discrete_transform = monai.transforms.AsDiscrete(logit_thresh=0.5, threshold_values=True)
    sigmoid = torch.nn.Sigmoid()

    with torch.no_grad():
        img_tensor = sample['img'].float().to(device)  # Ensure float32 and move to GPU
        output = model(img_tensor)  # Get model output
        output = sigmoid(output).cpu().squeeze().numpy()  # Apply sigmoid, move to CPU, and convert to NumPy
        output = discrete_transform(output)  # Threshold the output

    # Extract image and mask (ensure they are 2D)
    img = sample["img"].squeeze().cpu().numpy()  # Convert image to NumPy, shape -> [512, 512]
    #mask = sample['mask'].squeeze().cpu().numpy()  # Convert mask to NumPy, shape -> [512, 512]

    # Prepare overlays (ensure they are 2D)
    #overlay_mask = np.ma.masked_where(mask == 0, mask)  # Remove extra dimensions
    overlay_output = np.ma.masked_where(output < 0.1, output)  # Ensure it's 2D

    # Plot results
    fig, ax = plt.subplots(1, 2, figsize=[15, 5])
    
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('X-ray Image')

    ax[1].imshow(img, cmap='gray')
    ax[1].imshow(overlay_output, cmap='Reds', alpha=0.7)  # Red prediction overlay
    ax[1].set_title('Model Prediction')

    plt.show()

def compute_test_predictions(dataloader, model, device):
    """
    This function runs inference on a test set (no ground truth masks).
    Returns a list of model predictions.
    """
    model.eval()
    inferer = monai.inferers.SlidingWindowInferer(roi_size=[256, 256])
    discrete_transform = monai.transforms.AsDiscrete(threshold=0.5)
    Sigmoid = torch.nn.Sigmoid()

    predictions = []  # Store model outputs

    for sample in dataloader:
        with torch.no_grad():
            # Run inference on the image
            output = discrete_transform(Sigmoid(inferer(sample['img'].to(device), network=model).cpu()))

            predictions.append(output)  # Store output tensor

    return predictions
