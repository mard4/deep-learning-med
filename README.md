# Retinal Vessel Segmentation
Most of the retinal diseases (retinopathy, occlusion etc.), can be identified through changes exhibited in retinal vasculature of fundus images. Thus, segmentation of retinal blood vessels aids in detecting the alterations and hence the disease. Manual segmentation of vessels  is a very tedious and time consuming task. Employing computational approaches for this purpose would help in efficient retinal analysis.
The methodology presented here first applies a series of deep-learning models to segment the vessels, and then explores the use of generative models to synthesize additional image data.
The proposed technique is validated on images from publicly available DRIVE database.

Here, we present several strategies to tackle this challenge and their corresponding results. First, we apply data preprocessing techniques including resolution enhancement and augmentations such as CLAHE and Gabor filtering to enrich the dataset. 
Next, we address segmentation using modified network architectures and an ensemble approach. Finally, we investigate state-of-the-art generative models and showcase the promising results achieved with these methods.

## Segmentation Model
![image](https://github.com/user-attachments/assets/f4bd91cf-2e14-48e2-8f41-4943dfc979a3)

## Generative Model
Diffusion Model + SPADE 

(in development)
