# 🐸 RIViT: Resolution Invariant Vision Transformer

_6.7960 Deep Learning FA24_

_kellycui, kwoo1, sjzheng_

In applications like Natural Language Processing, transformers have quickly become dominant models because of their scalibility and ability to capture long-range dependencies. Vision Transformers (ViTs) adapt the transformer architecture from text to image data. While ViTs perform effectively with large datasets, they still struggle in performance compared to existing convolutional neural network (CNN) models, especially in smaller datasets.

In particular, CNNs outperform ViTs in generalizing across different image resolutions. Our project is interested in improving current ViT architecture by building **resolution invariant ViTs**. Objects naturally appear at different sizes and image resolution. Improving resolution invariance in ViTs poses benefits to fields like healthcare, autonomous systems, and urban planning, where tasks like image classification and semantic segmentation help with object recognition in medical scans and satellite images, input images that often cover a diverse range of sizes.

We focus on two areas of improvement: _standardizing patch embedding lengths across different resolutions_ and _experimenting with positional encodings to combat spatial locality_.

Currently, ViTs struggle with resolution scalability, particularly for high-resolution images. When processing ViTs on image input sizes much larger than those from training, model performance declines. To address this, we propose a novel model to improve ViT task accuracy in image classification. For positional encoding, we will test on absolute and relative positional encoding. We hypothesize that standardizing patch embedding lengths and introducing absolute or relative positional encoding will improve task accuracy in classification by dynamically adapting to resolution variability compared to a baseline ViT model.
