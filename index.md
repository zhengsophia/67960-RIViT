# Scale Invariant Vision Transformers

_6.7960 Final Project_
_By Kelly Cui, Andrew Woo, Sophia Zheng_
_12/10/2024_

---

## 1. Introduction

In applications like Natural Language Processing, transformers have quickly become dominant models because of their scalibility and ability to capture long-range dependencies. Vision Transformers (ViTs) adapt the transformer architecture from text to image data. While ViTs perform effectively with large datasets, they still struggle in performance compared to existing convolutional neural network (CNN) models, especially in smaller datasets.

In particular, CNNs outperform ViTs in generalizing across different image resolutions. Our project is interested in improving current ViT architecture by building **scale invariant ViTs**. Objects naturally appear at different sizes and image resolution. Improving scale invariance in ViTs poses benefits to fields like healthcare, autonomous systems, and urban planning where tasks like image classification and semantic segmentation help with object recognition in medical scans and satellite images.

We focus on two areas of improvement: _standardizing patch embedding lengths across different resolutions_ and _experimenting with positional encodings to combat spatial locality_.

Currently, ViTs struggle with resolution scalability, particularly for high-resolution images. When processing ViTs on image input sizes much larger than those from training, model performance declines. To address this, we propose a novel model to improve ViT task accuracy in (1) image classification and (2) semantic segmentation. For positional encoding, we will test on absolute and relative positional encoding. We hypothesize that standardizing patch embedding lengths and introducing absolute or relative positional encoding will improve task accuracy in both classification and segmentation by dynamically adapting to resolution variability compared to a baseline ViT model.

## Related Work

ViTAR[^1], or ViT with Any Resolution describes a transformer architecture that uses an adaptive token merger to give fixed size token arrays to be fed into a standard transformer, with fuzzy positional encodings. The token merger uses adaptive "GridAttention" based on image resolution, which is a combination of average pooling with CrossAttention. Resformer[^2] proposes multi-resolution training for ViTs to improve resolution invariance. Our model's novelty lies in its positional encoding methods as well as its use of adaptive pooling during patch embedding to remain resolution invariant.

---

## Methods

Our baseline model for comparison will be a vanilla ViT model. We will compare the performance of our novel model against the baseline for the following two tasks: (i) image classification, using the ImageNet dataset, and (ii) semantic segmentation, using the ADE20K dataset.

To achieve resolution invariance, we propose using convolution and adaptive pooling layers on patch embedding outputs to reduce tokens to a fixed size. We hypothesize that learnable conv2D layers will capture more localized features with average pooling to preserve high-resolution data fed into the base transformer.
We will compare two different approaches for positional encoding: \emph{absolute} positional encoding, which encode solely as a function of patch position, versus \emph{relative} encoding, which learn relationships between distinct image patches.
We will follow ResFormer's training methodology to perform multi-resolution training.

### Positional Encodings

The basic transformer architecture is permutation-invariant (with the exception of masked attention); the order of the input tokens does not impact the output of self attention layers. However, token positions can be crucial for both NLP and vision tasks: for example, the position of a word can change the meaning of a sentence, and the location of a patch in an image can correlate to the object it represents.   

Hence, we need _positional encodings_ to enable our models to learn from the positions of tokens in the input. 

We compare a few different options for positional encodings.

#### Learned Positional Encodings

Each position is mapped to a vector of parameters. This positional embedding matrix is initialized randomly and learned by the model during training.

#### Sinusoidal Position Encodings



#### Relative Positional Encodings



---

## Results

---

## Conclusion

[citations]: #

[^1]:
    **ViTAR: Vision Transformer with Any Resolution**
    Fan, Q., You, Q., Han, X., Liu, Y., Tao, Y., Huang, H., He, R., & Yang, H., 2024. arXiv:2403.18361.

[^2]:
    **ResFormer: Scaling ViTs with Multi-Resolution Training**
    Tian, R., Wu, Z., Dai, Q., Hu, H., Qiao, Y., & Jiang, Y., 2022. arXiv:2212.00776.
