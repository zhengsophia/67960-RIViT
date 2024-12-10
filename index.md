# Resolution Invariant Vision Transformers

_6.7960 Final Project_

_By Kelly Cui, Andrew Woo, Sophia Zheng_

_12/10/2024_

---

## 1. Introduction

In applications like Natural Language Processing, transformers have quickly become dominant models because of their scalibility and ability to capture long-range dependencies. Vision Transformers (ViTs) adapt the transformer architecture from text to image data. While ViTs perform effectively with large datasets, they still struggle in performance compared to existing convolutional neural network (CNN) models, especially in smaller datasets.

In particular, CNNs outperform ViTs in generalizing across different image resolutions. Our project is interested in improving current ViT architecture by building **resolution invariant ViTs**. Objects naturally appear at different sizes and image resolution. Improving scale invariance in ViTs poses benefits to fields like healthcare, autonomous systems, and urban planning where tasks like image classification and semantic segmentation help with object recognition in medical scans and satellite images.

We focus on two areas of improvement: _standardizing patch embedding lengths across different resolutions_ and _experimenting with positional encodings to combat spatial locality_.

Currently, ViTs struggle with resolution scalability, particularly for high-resolution images. When processing ViTs on image input sizes much larger than those from training, model performance declines. To address this, we propose a novel model to improve ViT task accuracy in image classification. For positional encoding, we will test on absolute and relative positional encoding. We hypothesize that standardizing patch embedding lengths and introducing absolute or relative positional encoding will improve task accuracy in classification by dynamically adapting to resolution variability compared to a baseline ViT model.

## 2. Related Work

### 2.1 ViT with Any Resolution

ViTAR[^1], or ViT with Any Resolution describes a transformer architecture that uses an adaptive token merger to give fixed size token arrays to be fed into a standard transformer, with fuzzy positional encodings. The token merger uses adaptive "GridAttention" based on image resolution, which is a combination of average pooling with CrossAttention.

### 2.2 Resformer

Resformer[^2] proposes multi-resolution training for ViTs to improve resolution invariance. Our model's novelty lies in its positional encoding methods as well as its use of adaptive pooling during patch embedding to remain resolution invariant.

### 2.3 Multi-Scale Vision Longformer

Multi-Scale Vision Longformer[^3] details combining a multi-scale model structure and a Vision Longformer, with attention mechanisms adapted from standard Longformers. This stacks more ViT stages to achieve the multi-scale structure of deep CNNs. Our approach of convolution on patch embedding leverages CNN feature extraction to achieve similar benefits, hybridizing advantages of local CNN extraction and global Transformer modeling.

---

## 4. Methods

Our baseline model for comparison will be a vanilla ViT model. We will compare the performance of our novel model against the baseline for the task of image classification, using the ImageNette dataset.

To achieve resolution invariance, we propose using convolution on patch embedding outputs to reduce tokens to a fixed size. We hypothesize that learnable conv2D layers will capture more localized features with average pooling to preserve high-resolution data fed into the base transformer.
We will compare two different approaches for positional encoding: _absolute_ positional encoding, which encode solely as a function of patch position, versus _relative_ encoding, which learn relationships between distinct image patches.
We will follow ResFormer's training methodology to perform multi-resolution training.

---

## 5. Results

Relative CNN: 80.56% on validation- 80 epochs, Sinusoidal CNN: 77.93% on validation, 90 epochs (stock model size with 6 layers, 3 head, 128 neurons)

Relative: 77.89% on validation ~90 epochs, Sinusoidal: 76.56% on validation, 90 epochs

CNN took ~30 seconds per epoch on a 3090 while non CNN took 73 seconds per epoch.

---

## 6. Conclusion

[citations]: #

[^1]:
    **ViTAR: Vision Transformer with Any Resolution**
    Fan, Q., You, Q., Han, X., Liu, Y., Tao, Y., Huang, H., He, R., & Yang, H., 2024. arXiv:2403.18361.

[^2]:
    **ResFormer: Scaling ViTs with Multi-Resolution Training**
    Tian, R., Wu, Z., Dai, Q., Hu, H., Qiao, Y., & Jiang, Y., 2022. arXiv:2212.00776.

[^3]: **Multi-Scale Vision Longformer: A New Vision Transformer for High-Resolution Image Encoding**
