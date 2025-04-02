import numpy as np
import tifffile as tiff
import os
import skimage.exposure as exposure
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import cv2

def apply_augmentation(train_dict_list, test_dict_list):
    # extracting the list of paths to the tif images
    print("Applying augmentations")
    train_img_list = [d.get("img") for d in train_dict_list if "img" in d]
    print('Sample from train_img_list')
    print(train_img_list[0])
    test_img_list = [d.get("img") for d in test_dict_list if "img" in d]
    print('Sample from test_img_list')
    print(test_img_list[0])

    ### 
    train_processed_list = image_preprocessing(train_img_list, 'train')

    # temporary just for testing purposes
    #first_two_img_list = train_img_list[:2]
    #print(first_two_img_list)
    #train_processed_list = image_preprocessing(first_two_img_list, 'train')

    # substituiting the images in the dictionaries
    for new_img, d in zip(train_processed_list, train_dict_list):
        d["img"] = new_img
    i = 1;   
    # checking the computation    
    for d in train_dict_list:
        if i < 41:
            i = i + 1
        else:
            #print(d)
            break

    # doing the same for test data
    test_processed_list = image_preprocessing(test_img_list, 'test')

    # substituiting the images in the dictionaries
    for new_img, d in zip(test_processed_list, test_dict_list):
        d["img"] = new_img
    
    return train_processed_list, test_processed_list

def image_preprocessing(tiff_img_paths, type):
    clahe_img_paths = apply_clahe(tiff_img_paths, type)
    #gabor_img_paths = apply_gabor_filter(clahe_img_paths, type)
    #halfwave_img_paths = apply_halwave_rectification (gabor_img_paths, type)
    #halfwave_img_paths = apply_halwave_rectification (clahe_img_paths, type)
    return clahe_img_paths
    #return halfwave_img_paths

def apply_clahe(tiff_img_paths, type):
    # define CLAHE parameters as recommended by numerous papers
    nbins = 256  # number of histogram bins
    clip_limit = 0.01  # contrast limiting
    tile_size = (8, 8)  # tile grid size (local equalization)
    
    # setting offset for output name depending on the type of(train or test)
    if type == 'train':
        offset = 20
    else:
        offset = 0
    
    # index to count the iteration number (starting from 1 since images are numerated from offset + 1
    i = 1;
    
    # creating the list to save the paths to CLAHE images
    path = []
    
    # creating the folder where the computation results are gonna be saved
    output_dir = '../datasets/CLAHE_dataset/' + type;
    ##print('Saving computation in folder ' + output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # computation
    for img_name in tiff_img_paths:
        img = tiff.imread(img_name)
        
        # since CLAHE works on single-channel images, convert image to grayscale
        if len(img.shape) == 3:
            img = rgb2gray(img)
        
        # apply CLAHE with the set parameters
        clahe_img = exposure.equalize_adapthist(img, nbins=nbins, clip_limit=clip_limit, kernel_size=tile_size)

        # convert to uint8 (0-255 range) for saving
        clahe_img = (clahe_img * 255).astype(np.uint8)
        ##print(f"Image {offset + i} is {img.shape}, {img.dtype}")
        
        # save the processed image
        clahe_path = f'clahe_{offset + i}.tif'
        output_path = os.path.join(output_dir, clahe_path)
        path.append(output_path)
        tiff.imwrite(output_path, clahe_img)

        ##print(f"Processed image : {offset + i} → Saved to {output_path}")
        i = i + 1;

    print("All images processed with customized CLAHE.")
    print(path)
    return path

# helper functions for the Gabor filter's impelementation according to the pipeline
'''
# function to compute sigma
def compute_sigma(wavelength, bandwidth):
    slratio = (1 / np.pi) * np.sqrt(np.log(2) / 2) * ((2**bandwidth + 1) / (2**bandwidth - 1))
    return slratio * wavelength

# function to create a 2D Gabor kernel
def create_gabor_kernel(wavelength, theta, phase_offset, aspect_ratio, bandwidth):
    sigma = compute_sigma(wavelength, bandwidth)
    kernel_size = int(np.floor(2.5 * sigma / aspect_ratio))  # Compute n
    kernel_size = 2 * kernel_size + 1  # Ensure odd size

    # create mesh grid
    x, y = np.meshgrid(np.linspace(-kernel_size//2, kernel_size//2, kernel_size),
                       np.linspace(-kernel_size//2, kernel_size//2, kernel_size))
    
    # flip y direction
    y = -y  

    # compute Gabor filter components
    f = 2 * np.pi / wavelength
    b = 1 / (2 * sigma**2)
    a = b / np.pi
    xp = x * np.cos(theta) + y * np.sin(theta)
    yp = -x * np.sin(theta) + y * np.cos(theta)
    
    # compute Gabor kernel
    cos_func = np.cos(f * xp - phase_offset)
    kernel = a * np.exp(-b * (xp**2 + (aspect_ratio**2) * yp**2)) * cos_func

    # normalize the kernel to have zero integral
    pos = np.sum(kernel[kernel > 0])
    neg = np.abs(np.sum(kernel[kernel < 0]))
    kernel[kernel > 0] /= pos
    kernel[kernel < 0] /= neg

    return kernel

# Gabor filter
def apply_gabor_filter(clahe_img_paths, type):
    
    # Gabor filter parameters
    #wavelengths = [9, 10, 11]
    wavelengths = [5]
    num_orientations = 24
    orientations = np.linspace(-np.pi, np.pi, num_orientations)  # 24 orientations from -π to π
    phase_offsets = np.linspace(-np.pi, np.pi, 5)  # phase offsets between -π and π
    aspect_ratio = 0.5
    bandwidth = 1
    
    # creating the folder where the computation results are gonna be saved
    output_dir = '../datasets/Gabor_dataset/' + type;
    print('Saving computation in folder ' + output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # creating the list to save the paths to Gabor images
    path = []
    
    # setting offset for output name depending on the type of(train or test)
    if type == 'train':
        offset = 20
    else:
        offset = 0
    
    # index to count the iteration number (starting from 1 since images are numerated from offset + 1
    i = 1;
    
    for img_name in clahe_img_paths:
        img = tiff.imread(img_name)
        
        # since images are already in grayscale, normalisation only
        img = img.astype(np.float32) / 255.0  
        
        # storage for the peak response
        gabor_response = np.zeros_like(img)
        
        for wavelength in wavelengths:
            for theta in orientations:
                for phase in phase_offsets:
                    # generate Gabor kernel
                    kernel = create_gabor_kernel(wavelength, theta, phase, aspect_ratio, bandwidth)
                    
                    # convolve image with Gabor filter
                    filtered_img = convolve(img, kernel)

                    # take maximum response for final Gabor response image
                    gabor_response = np.maximum(gabor_response, filtered_img)
        
        # normalize and save the output image
        gabor_response = (gabor_response - gabor_response.min()) / (gabor_response.max() - gabor_response.min())
        gabor_response = (gabor_response * 255).astype(np.uint8)
        
        # save the processed image
        gabor_path = f'gabor_{offset + i}.tif'
        output_path = os.path.join(output_dir, gabor_path)
        path.append(output_path)
        tiff.imwrite(output_path, gabor_response)
        
        print(f"Processed image : {offset + i} → Saved to {output_path}")
        i = i + 1;
        
    print("Gabor filtering completed for all images.")
    print(path)
    return path

def apply_halwave_rectification (gabor_img_paths, type):
    # halfwave percentage set to 10
    hwpercent = 10
    
    # creating the folder where the computation results are gonna be saved
    output_dir = '../datasets/Halfwave_dataset/' + type;
    print('Saving computation in folder ' + output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # creating the list to save the paths to Halfwave rectified images 
    path = []
    i = 1;
    
    # setting offset for output name depending on the type of(train or test)
    if type == 'train':
        offset = 20
    else:
        offset = 0
    
    for img_name in gabor_img_paths:
        img = tiff.imread(img_name)
        maxIntensity = np.max(img)
        perVal = maxIntensity * hwpercent/100.0
        img_rectified = np.where(img < perVal, 0, img)
        
        #save the processed image
        halfwave_path = f'halfwave_{offset + i}.tif'
        output_path = os.path.join(output_dir, halfwave_path)
        path.append(output_path)
        tiff.imwrite(output_path, img_rectified)
        
        print(f"Processed image : {offset + i} → Saved to {output_path}")
        i = i + 1;
    print("Halfwave rectification completed for all images.")
    print(path)
    return path
'''