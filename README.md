# Retinal Vessel Segmentation

Project associated with the course of Deep Learning for 3D Medical Imaging at University of Twente.

<table>
  <tr>
    <td align="center" width="40%">
      <a href="https://github.com/user-attachments/files/19937609/presentation_dlmia.pdf">
        <img src="https://github.com/user-attachments/assets/5c6f183a-bcb6-4c22-bcd8-17b2e70109c8" width="400px"><br>
        <b>Presentation</b>
      </a>
    </td>
    <td align="center" width="40%">
      <a href="https://github.com/user-attachments/files/19937652/DeepLearningReport.pdf">
        <img src="https://github.com/user-attachments/assets/f97d34be-87d5-4057-8119-9b3d54387e18" width="400px"><br>
        <b>Report</b>
      </a>
    </td>
  </tr>
</table>


Changes in the retinal vasculature, visible in fundus images, are key indicators for diagnosing retinal diseases such as retinopathy and vascular occlusions. Accurate segmentation of retinal blood vessels plays a crucial role in early disease detection.

This project presents a methodology that first applies a series of deep learning models to segment vessels, and then explores the use of generative models to synthesize additional image data.  
The technique is validated on images from the publicly available [DRIVE database](https://drive.grand-challenge.org/).

<img src="https://github.com/user-attachments/assets/f9a522f2-895b-4588-9d81-201bbf2cd2d0" width="400px">

---

## Methodology

- **Data Preprocessing:**  
  Resolution enhancement and augmentations, including CLAHE and Gabor filtering.
  
- **Segmentation:**  
  Ensemble of UNet and Swin Transformer models.

- **Generative Modeling:**  
  Image synthesis using Diffusion Models combined with SPADE normalization.

---

## Segmentation

The overall Dice coefficient and pixel-wise accuracy used to assess how well the model predicts each pixel’s class are computed and logged alongside the ground truth masks.

<img src="https://github.com/user-attachments/assets/88cfb82b-46bc-4685-8cd7-b5c23492cc32" width="400px">

Validation results: summarized by the average Mean Dice score and illustrated below show that the predicted masks closely resemble the ground truth.  
The Mean Dice plot further highlights that models like SwinTNet exhibit more consistent performance over time compared to U-Net Dropout.
<img src="https://github.com/user-attachments/assets/35f8e85d-3043-4d00-91ae-85ce4f556e67" width="400px">

Dice Coefficient and Hausdorff Distance are employed as evaluation metrics.  
Due to the limited size of the DRIVE dataset, incorporating additional datasets such as STARE improved overall performance.

The best results are obtained using an ensemble of U-Net with Dropout and SwinTNet models.  
The highest Mean Dice Coefficient is achieved when both DRIVE and STARE datasets are used for training.

<img src="https://github.com/user-attachments/assets/22256cef-8517-4ede-9143-b4405f745f5f" width="1000px">



---

## Generative Model

We combined three different methods:

- **SPADE**: a normalization layer that uses a segmentation mask to modulate feature maps during generation. SPADE ensures semantic information (e.g., class labels per pixel) is preserved by injecting learned scale and bias parameters at each spatial location.

- **Latent Diffusion Models**: a two-stage approach where an autoencoder first compresses images into a lower-dimensional latent space, and a diffusion model is trained on these latents.

- **Semantic Diffusion**: an approach that feeds the noisy image latent into a UNet encoder and the semantic layout into the decoder via multi-layer SPADE normalization. This method leverages the mask information more effectively than simply concatenating it with the input.

<img src="https://github.com/user-attachments/assets/c78ea63c-17da-4ef9-959f-e7644c39e86b" width="600px">

---

# Repository Structure

- `datasets/` — Contains all datasets used for training and evaluation.

- `segmentation/` — Main folder with all segmentation-related files.  
  Use `config.yaml` to modify hyperparameters and dataset settings.  
  - Run `MAIN-withvalidmask-stare-drive.ipynb` to execute the segmentation pipeline.
  - Run `spade_ldm_MEDNET.ipynb` to execute the generative modeling pipeline.

- `generative/` —
Contains additional models and experiments related to generative approaches.


# Bibliography

```bibtex
@inproceedings{park2019semantic,
  title={Semantic image synthesis with spatially-adaptive normalization},
  author={Park, Taesung and Liu, Ming-Yu and Wang, Ting-Chun and Zhu, Jun-Yan},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={2337--2346},
  year={2019}
}

@article{wang2022semantic,
  title={Semantic image synthesis via diffusion models},
  author={Wang, Weilun and Bao, Jianmin and Zhou, Wengang and Chen, Dongdong and Chen, Dong and Yuan, Lu and Li, Houqiang},
  journal={arXiv preprint arXiv:2207.00050},
  year={2022}
}

@online{drive_challenge,
  title     = {DRIVE: Digital Retinal Images for Vessel Extraction},
  url       = {https://drive.grand-challenge.org/},
  urldate   = {2025-04-07}
}

@inproceedings{DIFFUSIONMODEL,
  title={High-resolution image synthesis with latent diffusion models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={10684--10695},
  year={2022}
}

@article{STARE,
   title={DPN: detail-preserving network with high resolution representation for efficient segmentation of retinal vessels},
   volume={14},
   ISSN={1868-5145},
   url={http://dx.doi.org/10.1007/s12652-021-03422-3},
   DOI={10.1007/s12652-021-03422-3},
   number={5},
   journal={Journal of Ambient Intelligence and Humanized Computing},
   publisher={Springer Science and Business Media LLC},
   author={Guo, Song},
   year={2021},
   month=aug, pages={5689–5702} }

