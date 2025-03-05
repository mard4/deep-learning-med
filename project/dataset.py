import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import monai
from PIL import Image
import torch
from monai.transforms import Transform


def build_dict_vessels(data_path, mode='training'):
    """
    This function returns a list of dictionaries, each dictionary containing the keys 'img' and 'mask' 
    that returns the path to the corresponding image.
    
    Args:
        data_path (str): path to the root folder of the data set.
        mode (str): subset used. Must correspond to 'train' or 'test'.
        
    Returns:
        (List[Dict[str, str]]) list of the dictionaries containing the paths of images and masks.
    """
    # test if mode is correct
    if mode not in ["training", "test"]:
        raise ValueError(f"Please choose a mode in ['train', 'test']. Current mode is {mode}.")
    
    # define empty dictionary
    dicts = []
    # list all files in directory, including the path
    paths_retina = glob.glob(os.path.join(data_path, mode, 'images', '*.tif'))
    # make a corresponding list for all the mask files
    for retina_path in paths_retina:
        # find the binary mask that belongs to the original image, based on indexing in the filename
        image_index = os.path.basename(retina_path).split('_')[0]
        # define path to mask file based on this index and add to list of mask paths
        mask_path = os.path.join(data_path, mode, '1st_manual', f'{image_index}_manual1.gif')
        if os.path.exists(mask_path):
            dicts.append({'img': retina_path, 'mask': mask_path})
    return dicts


class LoadVesselData(Transform):
    """
    This custom Monai transform loads and processes data from the rib segmentation dataset.
    It handles RGB image loading, resizing, normalization, and binary mask conversion.
    """
    def __init__(self, keys=None):
        super().__init__()
        self.keys = keys

    def __call__(self, sample):
        try:
            # Load and process the image
            image = Image.open(sample['img']).convert('RGB')
            image = image.resize((512, 512), resample=Image.Resampling.NEAREST)
            image = np.array(image)
            if image.shape[-1] == 3:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255  # Convert to tensor and normalize

            # Load and process the mask
            mask = Image.open(sample['mask']).convert('L')
            mask = mask.resize((512, 512), resample=Image.Resampling.NEAREST)
            mask = np.array(mask, dtype=np.uint8)
            mask = np.where(mask == 255, 1, 0)  # Convert mask values to binary
            mask = torch.from_numpy(mask).unsqueeze(0).float() / 255  # Convert to tensor and normalize

            # Add metadata (if needed for further processing or consistency)
            return {
                'img': image,
                'mask': mask,
                'img_meta_dict': {'affine': np.eye(2)},
                'mask_meta_dict': {'affine': np.eye(2)}
            }

        except Exception as e:
            print(f"Error processing file: {e}")
            return None

        
def visualize_vessel_sample(sample, title=None):
    # Visualize the X-ray and overlay the mask, using the dictionary as input
    import matplotlib.pyplot as plt    
    image = sample['img']
    mask = sample['mask']
    
    # Check if the data is in PyTorch tensor format, if so, convert to numpy
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()  # Convert to numpy array
        image = np.squeeze(image)  # Remove single-dimensional entries
        if image.ndim == 3 and image.shape[0] == 3:  # If it's a three-channel image
            image = np.transpose(image, (1, 2, 0))  # Reorder dimensions for plotting
    
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()  # Convert to numpy array
        mask = np.squeeze(mask)  # Remove single-dimensional entries
    
    plt.figure(figsize=[10, 7])
    plt.imshow(image, cmap='gray')  # Ensure grayscale display
    overlay_mask = np.ma.masked_where(mask == 0, mask)  # Create an overlay for the mask
    plt.imshow(overlay_mask, cmap='gray', alpha=0.7, interpolation='nearest')  # Display mask overlay
    
    if title is not None:
        plt.title(title)
    plt.show()