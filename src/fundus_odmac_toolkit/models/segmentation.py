import warnings
from functools import lru_cache
from typing import List, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as Ftv
import ttach as tta
from fundus_data_toolkit.config import get_normalization
from fundus_data_toolkit.functional import autofit_fundus_resolution, reverse_autofit_tensor
from fundus_odmac_toolkit.models.hf_hub import download_model

Architecture = Literal["unet"]
EncoderModel = Literal["resnet34"]


def segment(
    image: np.ndarray,
    arch: Architecture = "unet",
    encoder: EncoderModel = "maxvit_small_tf_512",
    image_resolution=512,
    autofit_resolution=True,
    reverse_autofit=True,
    mean=None,
    std=None,
    return_features=False,
    return_decoder_features=False,
    features_layer=3,
    device: torch.device = "cuda",
    compile: bool = False,
    use_tta: bool = False,
):
    """Segment fundus image into 5 classes: background, CTW, EX, HE, MA

    Args:
        image (np.ndarray):   Fundus image of size HxWx3
        arch (Architecture, optional): Defaults to 'unet'.
        encoder (EncoderModel, optional): Defaults to 'resnest50d'.
        image_resolution (int, optional): Defaults to 1024.
        mean (list, optional): Defaults to constants.DEFAULT_NORMALIZATION_MEAN.
        std (list, optional): Defaults to constants.DEFAULT_NORMALIZATION_STD.
        autofit_resolution (bool, optional):  Defaults to True.
        return_features (bool, optional): Defaults to False. If True, returns also the features map of the i-th encoder layer. See features_layer.
        features_layer (int, optional): Defaults to 3. If return_features is True, returns the features map of the i-th encoder layer.
        device (torch.device, optional): Defaults to "cuda".

    Returns:
        torch.Tensor: 5 channel tensor with probabilities of each class (size 5xHxW)
    """
    assert not (use_tta and return_features), "return_features is not compatible with use_tta"

    model = get_model(arch, encoder, device, compile=compile, with_ttach=use_tta)
    model.eval()
    h, w, c = image.shape
    if autofit_resolution:
        image, roi, transforms = autofit_fundus_resolution(image, image_resolution, return_roi=True)

    image = (image / 255.0).astype(np.float32)
    tensor = torch.from_numpy(image).permute((2, 0, 1)).unsqueeze(0).to(device)

    if mean is None:
        mean = get_normalization()[0]
    if std is None:
        std = get_normalization()[1]
    tensor = Ftv.normalize(tensor, mean=mean, std=std)
    with torch.inference_mode():
        if use_tta:
            pred = model(tensor)
        else:
            features = model.encoder(tensor)
            pre_segmentation_features = model.decoder(*features)
            pred = model.segmentation_head(pre_segmentation_features)
            pred = F.softmax(pred, 1)
            pred = F.interpolate(pred, (image_resolution, image_resolution), mode="bilinear", align_corners=False)
        if return_features or return_decoder_features:
            assert (
                not reverse_autofit
            ), "reverse_autofit is not compatible with return_features or return_decoder_features"
            out = [pred]
            if return_features:
                out.append(features[features_layer])
            if return_decoder_features:
                out.append(pre_segmentation_features)
            return tuple(out)

    pred = pred.squeeze(0)
    if reverse_autofit and autofit_resolution:
        pred = reverse_autofit_tensor(pred, **transforms)
        all_zeros = ~torch.any(pred, dim=0)  # Find all zeros probabilities
        pred[0, all_zeros] = 1  # Assign them to background

        pred = F.interpolate(pred.unsqueeze(0), (h, w), mode="bilinear", align_corners=False).squeeze(0)
    return pred


def batch_segment(
    batch: Union[torch.Tensor, np.ndarray],
    arch: Architecture = "unet",
    encoder: EncoderModel = "seresnext50_32x4d",
    already_normalized=False,
    mean=None,
    std=None,
    return_features=False,
    features_layer=3,
    device: torch.device = "cuda",
    compile: bool = False,
    use_tta: bool = False,
):
    """Segment batch of fundus images into 5 classes: background, CTW, EX, HE, MA

    Args:
        batch (Union[torch.Tensor, np.ndarray]): Batch of fundus images of size BxHxWx3 or Bx3xHxW
        arch (Architecture, optional): Defaults to 'unet'.
        encoder (EncoderModel, optional): Defaults to 'resnest50d'.
        weights (TrainedOn, optional):  Defaults to 'All'.
        already_normalized (bool, optional): Defaults to False.
        mean (list, optional): Defaults to constants.DEFAULT_NORMALIZATION_MEAN.
        std (list, optional): Defaults to constants.DEFAULT_NORMALIZATION_STD.
        return_features (bool, optional): Defaults to False. If True, returns also the features map of the i-th encoder layer. See features_layer.
        features_layer (int, optional): Defaults to 3. If return_features is True, returns the features map of the i-th encoder layer.
        device (torch.device, optional):  Defaults to "cuda".

    Returns:
        torch.Tensor: 5 channel tensor with probabilities of each class (size Bx5xHxW)
    """

    model = get_model(arch, encoder, device, compile=compile, with_ttach=use_tta)
    model.eval()

    # Check if batch is torch.Tensor or np.ndarray. If np.ndarray, convert to torch.Tensor
    if isinstance(batch, np.ndarray):
        batch = torch.from_numpy(batch)  # Convert to torch.Tensor

    batch = batch.to(device)

    # Check if dimensions are BxCxHxW. If not, transpose
    if batch.shape[1] != 3:
        batch = batch.permute((0, 3, 1, 2))

    if mean is None:
        mean = get_normalization()[0]
    if std is None:
        std = get_normalization()[1]

    # Check if batch is normalized. If not, normalize it
    if not already_normalized:
        batch = batch / 255.0
        batch = Ftv.normalize(batch, mean=mean, std=std)

    with torch.inference_mode():
        pred = F.softmax(model(batch), 1)

    if return_features:
        features = model.encoder(batch)
        pred = model.segmentation_head(model.decoder(features))
        return F.softmax(pred, 1), features[features_layer]

    return pred


@lru_cache(maxsize=2)
def get_model(
    arch: Architecture = "unet",
    encoder: EncoderModel = "resnest50d",
    device: torch.device = "cuda",
    compile: bool = False,
    with_ttach: bool = False,
):
    """Get segmentation model

    Args:
        arch (Architecture, optional): Defaults to 'unet'.
        encoder (EncoderModel, optional):  Defaults to 'resnest50d'.
        weights (TrainedOn, optional):  Defaults to 'All'.
        device (torch.device, optional): Defaults to "cuda".

    Returns:
        nn.Module: Torch segmentation model
    """
    model = download_model(arch, encoder).to(device=device)
    if with_ttach:
        model.segmentation_head[-1] = nn.Softmax(dim=1)
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Rotate90(angles=[0, 90, 180, 270]),
            ]
        )

        model = tta.SegmentationTTAWrapper(model, transforms, merge_mode="mean")
    if compile:
        model.eval()
        with torch.inference_mode():
            model = torch.compile(model)
    return model
