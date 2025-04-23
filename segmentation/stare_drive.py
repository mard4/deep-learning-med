# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python
#     language: python
#     name: python3
# ---

# + [markdown] user_expressions=[]
# ## Building the dictionaries

# +
from dataset_stare import *
import tarfile
import os
from PIL import Image
import matplotlib.pyplot as plt
import glob  


combination=True
if combination:
    main_path = "../datasets"
    stare_images_dir = f"{main_path}/STARE_dataset/images"
    stare_masks_dir = f"{main_path}/STARE_dataset/labels-ah"
    dataset = config["dataset_paths"][config["dataset_index"]] ## Drive dataset

    stare_dicts = build_dict_stare(stare_images_dir, stare_masks_dir)
    drive_dicts = build_dict_vessels(drive_path, mode="training")
    train_dict_list = drive_dicts + stare_dicts
    test_dict_list = build_dict_vessels(dataset, mode='test')
    
# -

from monai.data import Dataset, DataLoader
train_transforms = Compose([
    LoadVesselData()
])
combined_dataset = Dataset(data=combined_dicts, transform=train_transforms)
combined_loader = DataLoader(combined_dataset, batch_size=4, shuffle=True)


for i in range(len(combined_dataset)):
    visualize_vessel_sample(combined_dataset[1], title=f"Sample {i+1}")
    visualize_vessel_sample(combined_dataset[-1], title=f"Sample {i-1}")
    break









# +
import tarfile
import os
from PIL import Image
import matplotlib.pyplot as plt
import glob  
main_path = "../datasets"

labels_tar_path = f"{main_path}/labels-ah.tar"
images_tar_path = f"{main_path}/stare-images.tar"

extract_dir = f"{main_path}/STARE_dataset"
images_dir = os.path.join(extract_dir, "images")
print(images_dir)
masks_dir = os.path.join(extract_dir, "labels-ah")

# Find a sample image and its corresponding mask
image_files = [f for f in os.listdir(images_dir) if f.endswith('.ppm')]
print(image_files)
sample_image_file = sorted(image_files)[0]
sample_mask_file = sample_image_file.replace('.ppm', '.ah.ppm')

sample_image_path = os.path.join(images_dir, sample_image_file)
sample_mask_path = os.path.join(masks_dir, sample_mask_file)

# Load image and mask
image = Image.open(sample_image_path).convert("RGB")
mask = Image.open(sample_mask_path).convert("L")

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image)
axes[0].set_title("STARE Fundus Image")
axes[0].axis("off")

axes[1].imshow(mask, cmap='gray')
axes[1].set_title("Expert Vessel Annotation (.ah)")
axes[1].axis("off")

plt.tight_layout()
plt.show()

