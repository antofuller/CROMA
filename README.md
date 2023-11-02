# CROMA
Official repository of **CROMA**: Remote Sensing Representations with **C**ontrastive **R**adar-**O**ptical **M**asked **A**utoencoders (NeurIPS '23)

# Using Pretrained CROMA models
My goal is to make CROMA as noob-friendly as possible.

First, load Sentinel-1 & 2 images and preprocess them. CROMA's default image size is 120x120px, but this can be changed. Sentinel-1 must be 2 channels and Sentinel-2 must be 12 channels (remove the cirrus band if necessary).

```python
import torch
from main_croma import PretrainedCROMA

device = 'cuda:0'  # use a GPU
use_8_bit = True
N = 1_000  # number of samples
sentinel_1 = torch.randn(N, 2, 120, 120).to(device)  # randomly generated for demonstration
sentinel_2 = torch.randn(N, 12, 120, 120).to(device)  # randomly generated for demonstration

def normalize(x, use_8_bit):
    # taken from SatMAE and SeCo
    x = x.float()

    imgs = []
    for channel in range(x.shape[1]):
        min_value = x[:, channel, :, :].mean() - 2 * x[:, channel, :, :].std()
        max_value = x[:, channel, :, :].mean() + 2 * x[:, channel, :, :].std()

        if use_8_bit:
            img = (x[:, channel, :, :] - min_value) / (max_value - min_value) * 255.0
            img = torch.clip(img, 0, 255).unsqueeze(dim=1).to(torch.uint8)
            imgs.append(img)
        else:
            img = (x[:, channel, :, :] - min_value) / (max_value - min_value)
            img = torch.clip(img, 0, 1).unsqueeze(dim=1)
            imgs.append(img)

    return torch.cat(imgs, dim=1)

sentinel_1 = normalize(sentinel_1, use_8_bit)
sentinel_2 = normalize(sentinel_2, use_8_bit)
```

PretrainedCROMA initializes the model(s) and loads the pretrained weights.
```python
model = PretrainedCROMA(pretrained_path='CROMA_base.pt', size='base', modality='both', image_resolution=120).to(device)

if use_8_bit:
    sentinel_1 = sentinel_1.float() / 255
    sentinel_2 = sentinel_2.float() / 255

with torch.no_grad():
    outputs = model(SAR_images=sentinel_1, optical_images=sentinel_2)
"""
outputs is a dictionary with keys:
'SAR_encodings' --> output of the radar encoder, shape (batch_size, number_of_patches, dim)
'SAR_GAP' --> output of the radar FFN (after global average pooling (GAP)), shape (batch_size, dim)
'optical_encodings' --> output of the optical encoder, shape (batch_size, number_of_patches, dim)
'optical_GAP' --> output of the optical FFN (after global average pooling (GAP)), shape (batch_size, dim)
'joint_encodings' --> output of the joint radar-optical encoder, shape (batch_size, number_of_patches, dim)
'joint_GAP' --> global averaging pooling the joint_encodings, shape (batch_size, dim)
"""
```
