import numpy as np
import tarfile, gzip, shutil
import imageio
from skimage import morphology
from PIL import Image
import PIL
from skimage.measure import block_reduce
import os

def remove_small_regions(img, size):
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

def resize_img(img):
    """
    Downsamples the image by half in width & height.
    For color images, uses PIL's bilinear interpolation.
    For binary masks or single-channel, uses block_reduce with max.
    """
    if len(img.shape) == 3:
        # (H, W, C)
        # Use PIL bilinear interpolation on each channel
        # Convert to PIL Image, resize, and back to numpy
        h, w, c = img.shape
        # Convert from [0..1 float or 0..255 int] to PIL image
        pil_img = Image.fromarray((img * 255).astype(np.uint8)) if img.max() <= 1.0 else Image.fromarray(img)
        new_w, new_h = ( (w + 1) // 2, (h + 1) // 2 )
        pil_img = pil_img.resize((new_w, new_h), resample=PIL.Image.BILINEAR)
        img = np.array(pil_img, dtype=np.float32) / 255.0
    else:
        # 2D array or single-channel
        # Use block_reduce with max
        img = block_reduce(img, block_size=(2, 2), func=np.max)
    return img

def create_valid_region_mask(img_2d_or_3d):
    """
    Creates a mask that is 1 where the original region is, and 0 where padding is added.
    Adapts to your same padding & resizing steps so it aligns perfectly with the final STARE image.
    """
    # If the image is color (H,W,C), we only need shape (H,W) for the mask
    if len(img_2d_or_3d.shape) == 3:
        h, w, _ = img_2d_or_3d.shape
    else:
        h, w = img_2d_or_3d.shape
    
    # 1) Start with an all-ones mask in the unpadded region
    mask_2d = np.ones((h, w), dtype=np.float32)
    
    # 2) Apply the same pad you do in stare_read_images
    #    if color => np.pad(img,( (1,2),(2,2),(0,0) ), ...),
    #    if single-channel => np.pad(img,( (1,2),(2,2) ), ...).
    # Here we have a 2D mask => pad((1,2),(2,2)):
    mask_2d = np.pad(mask_2d, ((1,2), (2,2)), mode='constant', constant_values=0)
    
    # 3) Resize the mask the same way you resize the image (down by half, etc.)
    #    Adapt your resize logic here:
    from PIL import Image
    import PIL
    from skimage.measure import block_reduce

    def resize_img(img):
        if len(img.shape) == 2:  # (H,W)
            # Use block_reduce with max or your chosen logic
            img = block_reduce(img, block_size=(2,2), func=np.max)
        else:
            # For color, you'd do your PIL bilinear logic. For a mask, block_reduce is simpler.
            pass
        return img

    mask_2d = resize_img(mask_2d)

    # 4) Expand dims so final shape is (H, W, 1)
    mask_2d = np.expand_dims(mask_2d, axis=-1)
    return mask_2d.astype(np.float32)


def stare_read_images(tar_filename, dest_folder, do_mask=False, do_valid_mask=False):
    """
    Unzips, loads, and (optionally) calculates segmentation masks 
    and valid-region (padding) masks for the STARE dataset images.

    Args:
        tar_filename (str): Path to .tar file containing (gzip'ed) STARE images
        dest_folder (str): Where files will be extracted
        do_mask (bool): Whether to build the 'segmentation mask' 
                        that excludes small dark blobs (with remove_small_regions).
        do_valid_mask (bool): Whether to build the 'valid region' padding mask 
                              that indicates the region of the image vs. the padded area.

    Returns:
        If do_mask=False and do_valid_mask=False:
            list of images
        If do_mask=True and do_valid_mask=False:
            (list of images, list of segmentation masks)
        If do_mask=True and do_valid_mask=True:
            (list of images, list of segmentation masks, list of valid-region masks)
        If do_mask=False and do_valid_mask=True:
            (list of images, list of valid-region masks)
    """
    tar = tarfile.open(tar_filename)
    tar.extractall(dest_folder)
    tar.close()
    
    all_images = []
    all_seg_masks = []
    all_valid_masks = []
    
    for item in sorted(os.listdir(dest_folder)):
        if item.endswith('gz'):
            gz_path = dest_folder + item
            # Unzip to remove .gz
            with gzip.open(gz_path, 'rb') as f_in:
                with open(dest_folder + item[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(gz_path)
            
            # Now read image from unzipped path
            raw_path = dest_folder + item[:-3]
            img = imageio.imread(raw_path)
            
            # --- PADDING ---
            if len(img.shape) == 3:
                # (H,W,C)
                img = np.pad(img, ((1,2), (2,2), (0,0)), mode='constant')
            else:
                # (H,W)
                img = np.pad(img, ((1,2), (2,2)), mode='constant')
            
            # --- RESIZE ---
            img = resize_img(img)
            
            # --- NORMALIZE ---
            # Scale to [0,1] if not already
            if img.max() > 1.0:
                img = img / 255.0
            img = img.astype(np.float32)
            
            # If single channel, ensure shape is (H,W,1)
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)  # from (H,W) -> (H,W,1)

            all_images.append(img)

            # Build a segmentation mask if requested
            if do_mask:
                # The logic from your snippet: 
                # "mask = (1 - remove_small_regions(np.prod((img<50/255.)*1.0, axis = 2)>0.5, 1000))*1.0"
                # but note that 'img' is already float in [0..1].
                # Typically the background is the dark region, so let's replicate your code.
                bin_mask = (img < (50.0/255.0)).astype(np.float32)
                # For multi-channel, we do np.prod along channels -> shape (H,W)
                if bin_mask.ndim == 3 and bin_mask.shape[-1] > 1:
                    bin_mask = np.prod(bin_mask, axis=2)
                # remove small connected components
                bin_mask = remove_small_regions(bin_mask > 0.5, 1000).astype(np.float32)
                # invert: region=1, background=0
                seg_mask = (1.0 - bin_mask).astype(np.float32)
                # shape => (H,W) or (H,W,1)
                seg_mask = np.expand_dims(seg_mask, axis=-1)
                all_seg_masks.append(seg_mask)

            # Build a valid-region/padding mask if requested
            if do_valid_mask:
                # We'll do the same steps as the original image
                # in a helper function. The function below
                # re-creates an all-ones region, pads it, and resizes it:
                # That means 1 inside the original region, 0 in the added padding.
                valid_mask = create_valid_region_mask(imageio.imread(raw_path))
                all_valid_masks.append(valid_mask)
    
    # Return results according to flags
    if do_mask and do_valid_mask:
        return all_images, all_seg_masks, all_valid_masks
    elif do_mask:
        return all_images, all_seg_masks
    elif do_valid_mask:
        return all_images, all_valid_masks
    else:
        return all_images

def build_dict_stare(stare_images_dir, stare_masks_dir, stare_padding_masks_dir=None):
    if stare_padding_masks_dir is None:
        stare_padding_masks_dir = stare_masks_dir

    dicts = []
    for filename in sorted(os.listdir(stare_images_dir)):
        if filename.endswith(".ppm"):
            image_path = os.path.join(stare_images_dir, filename)
            seg_mask_path = os.path.join(stare_masks_dir, filename.replace(".ppm", ".ah.ppm"))
            padding_mask_path = os.path.join(stare_padding_masks_dir, filename.replace(".ppm", "_valid.ppm"))

            entry = {"img": image_path}
            if os.path.exists(seg_mask_path):
                entry["mask"] = seg_mask_path

            # If valid region mask does NOT exist yet, create & save it
            if not os.path.exists(padding_mask_path):
                raw_img = imageio.imread(image_path)
                valid_mask = create_valid_region_mask(raw_img)  # shape (H, W, 1)
                # Squeeze to (H, W) for grayscale writing
                valid_mask_2d = np.squeeze(valid_mask, axis=-1)
                # Save as uint8 grayscale
                imageio.imwrite(padding_mask_path, (valid_mask_2d * 255).astype(np.uint8))

            if os.path.exists(padding_mask_path):
                entry["valid_mask"] = padding_mask_path

            dicts.append(entry)

    return dicts
