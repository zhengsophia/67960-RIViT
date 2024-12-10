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

## 4. Methodology

Our baseline model for comparison will be a vanilla ViT model. We will compare the performance of our novel model against the baseline for the task of image classification, using the Imagenette dataset. We will follow ResFormer's training methodology to perform multi-resolution training.

### 4.1. Dataset

We train on Imagenette, a subset of ImageNet with 10 distinct classes (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute). Imagenette consists of ~13,000 images, with a 70/30 train/validation split.

Images from Imagenette are loaded with their shorter dimension set to 320px. For training purposes, images are cropped, resized, and normalized to obtain square images. Following Resformer's multi-resolution training method, each training image is scaled to three different resolutions: 96px, 128px, and 160px. 

#### 4.1.1. Data Augmentations

To improve the robustness of our model, we apply a few random data augmentations:

1. Random horizontal flip (p=0.5)
2. Random rotation  (-15˚ to 15˚)
3. Random color jitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

### 4.2. Convolutional Patch Embeddings

To achieve resolution invariance, we use convolution and adaptive pooling layers on patch embedding outputs to reduce tokens to a fixed size. The CNN patch embeddings downsample the input image to extract important local features with conv2D layers before creating patches. This is expected to perform better than vanilla ViTs, where images are directly split into patches, and feature learning does not occur until later on in the Transformer architecture. We hypothesize that learnable conv2D layers will capture more localized features with average pooling to preserve high-resolution data fed into the base transformer.

### 4.3. Positional Encodings

The basic transformer architecture is permutation-invariant (with the exception of masked attention); the order of the input tokens does not impact the output of self attention layers. However, token positions can be crucial for both NLP and vision tasks: for example, the position of a word can change the meaning of a sentence, and the location of a patch in an image can correlate to the object it represents.

Hence, we need _positional encodings_ to enable our models to learn from the positions of tokens in the input. The choice of positional encoding is especially important for resolution invariance; as resolutions vary, so do the number of possible positions. The positional encoding must be capable of handling different token sequence lengths.

In our models, we explore three different options for positional encodings.

#### 4.3.1. Learned Positional Encodings

Each position is mapped to a vector of parameters with the same dimension as the token embeddings, and these vectors are added to the token embeddings at each forward pass of the model. This positional encoding matrix is initialized randomly and learned by the model during training.

Learned positional encodings have the advantage of being tailored to the task at hand, but suffer the drawback of only being trained on limited resolution sizes. Hence, they may not be able to generalize effectively to unseen longer resolution sizes.

#### 4.3.2. Sinusoidal Position Encodings

Sinusoidal position encodings also add position encoding vectors to the token embeddings, but this vector is calculated based on a fixed sinusoidal function instead of learned during training.

We implement the sinusoidal encodings as follows, based on the original Transformers paper [CITE]:

$$
PE_{(pos, i)} = \begin{cases}
    \sin\left(\frac{pos}{10,000^{i/dim}}\right) & \text{for $i$ even} \\
    \cos\left(\frac{pos}{10,000^{i/dim}}\right) & \text{for $i$ odd}
\end{cases}
$$

Here, $pos$ is the position index, $i$ is the $i$-th index of positional encoding for $pos$, and $dim$ is the dimension of the token embeddings.

This function extends easily to unseen resolution lengths, and requires less memory and computation than learned positional encodings.

#### 4.3.3. Relative Positional Encodings

The above two methods are both examples of _absolute_ positional encodings, i.e. they encode information for the position alone. However, _relative_ positional encodings capture pairwise information between different positions.

Not only do relative encodings introduce information about the relationships between tokens at different positions, but they can also be more generalizable to different resolutions. By nature, absolute encodings generally limit a model to some maximum token length, while pairwise relative encodings can generalize to unseen token sequence lengths.

We use the relative encoding scheme originating from Shaw et al. and adapted by Huang et al. [CITE], which modifies self-attention to add a new relative component to the keys:

$$RelativeAttention = \text{Softmax}\left(\frac{QK^\intercal + S_{rel}}{\sqrt{dim}}\right)V$$

$Q$, $K$, $V$ refer to the typical attention query, key, and value matrices. $S_{rel}$ is calculated using $Q$:

$$S_{rel} = QR^\intercal$$

$R$ is the relative positional encoding matrix, mapping each pair of tokens to a $dim$-length vector.

### 4.4. Training

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

[^3]:
    **Multi-Scale Vision Longformer: A New Vision Transformer for High-Resolution Image Encoding**
    Zhang, P., Dai, X., Yang, J., Xiao, B., Yuan, L., Zhang, L., & Gao, J., 2021. arXiv:2103.15358.
