"""The validated membrane normalization, raster convention and eight mirrors."""
import mrcfile
import numpy as np
import torch
from monai.transforms import Compose, ToTensor
from skimage.util import img_as_float32


def get_prediction_transforms():
    return Compose([ToTensor()])


def load_data_for_inference(path, transforms, device):
    with mrcfile.open(path, permissive=True) as volume:
        data = volume.data.copy().transpose(2, 1, 0)
        header, pixel = volume.header.copy(), volume.voxel_size.copy()
    data = img_as_float32(data)
    data -= np.mean(data)
    data /= np.std(data)
    tensor = transforms(np.expand_dims(data, 0)).unsqueeze(0).to(device)
    return tensor, header, pixel


def get_mirrored_img(image, fold):
    axes = ((), (4,), (3,), (4, 3), (2,), (4, 2), (3, 2), (4, 3, 2))
    if type(fold) is not int or not 0 <= fold < 8:
        raise ValueError('Membrane mirror index must be between zero and seven.')
    return torch.flip(image, axes[fold]) if fold else image
