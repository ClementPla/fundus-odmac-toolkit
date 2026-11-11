from functools import lru_cache
from typing import Literal, Tuple, Union, Optional, Dict
import numpy as np
from scipy import ndimage
from skimage import measure
import torch


import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as Ftv
import ttach as tta
from fundus_data_toolkit.config import get_normalization
from fundus_data_toolkit.functional import (
    autofit_fundus_resolution,
    reverse_autofit_tensor,
)
from fundus_odmac_toolkit.models.hf_hub import DEFAULT_REVISION, download_model

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
    revision: Optional[str] = None,
):
    """Segment fundus image into 3 classes: background, optic disc and macula

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
        torch.Tensor: 3 channel tensor with probabilities of each class (size 3xHxW)
    """
    assert not (use_tta and return_features), (
        "return_features is not compatible with use_tta"
    )

    model = get_model(arch, encoder, device, compile=compile, with_ttach=use_tta, revision=revision)
    model.eval()
    h, w, c = image.shape
    if autofit_resolution:
        image, roi, transforms = autofit_fundus_resolution(
            image, image_resolution, return_roi=True
        )

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
            if autofit_fundus_resolution:
                pred = F.interpolate(
                    pred,
                    (image_resolution, image_resolution),
                    mode="bilinear",
                    align_corners=False,
                )
        if return_features or return_decoder_features:
            assert not reverse_autofit, (
                "reverse_autofit is not compatible with return_features or return_decoder_features"
            )
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

        pred = F.interpolate(
            pred.unsqueeze(0), (h, w), mode="bilinear", align_corners=False
        ).squeeze(0)
    return pred


def batch_segment(
    batch: Union[torch.Tensor, np.ndarray],
    arch: Architecture = "unet",
    encoder: EncoderModel = "maxvit_small_tf_512",
    already_normalized=False,
    mean=None,
    std=None,
    return_features=False,
    features_layer=3,
    device: torch.device = "cuda",
    compile: bool = False,
    use_tta: bool = False,
    revision: Optional[str] = None,
):
    """Segment batch of fundus images into 3 classes: background, optic disc and macula

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
        torch.Tensor: 3 channel tensor with probabilities of each class (size Bx3xHxW)
    """

    model = get_model(arch, encoder, device, compile=compile, with_ttach=use_tta, revision=revision)
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
    revision: Optional[str] = None,
):
    """Get segmentation model

    Args:
        arch (Architecture, optional): Defaults to 'unet'.
        encoder (EncoderModel, optional):  Defaults to 'resnest50d'.
        weights (TrainedOn, optional):  Defaults to 'All'.
        device (torch.device, optional): Defaults to "cuda".
        revision (str, optional): Hugging Face branch of the weights. Defaults to DEFAULT_REVISION
            ("v2", retrained in 2026); "main" gives the original 2024 weights.

    Returns:
        nn.Module: Torch segmentation model
    """
    model = download_model(arch, encoder, revision=revision or DEFAULT_REVISION).to(device=device)
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


def postprocess_od_macula(
    pred_probs: np.ndarray,
    image_shape: Tuple[int, int] = None,
    od_size_ratio: float = 0.1,
    macula_od_ratio: float = 0.1,
    horizontal_tolerance_ratio: float = 0.15,
    return_centers: bool = True,
) -> Dict:
    """
    Post-process OD and macula segmentation predictions.

    Parameters:
    -----------
    pred_probs : np.ndarray
        Probability map of shape (3, H, W) where C0=background, C1=OD, C2=macula
    image_shape : tuple, optional
        (H, W) of original image. If None, uses pred_probs shape.
    od_size_ratio : float
        Expected OD size as fraction of image (default 0.1)
    macula_od_ratio : float
        Expected macula size as fraction of OD (default 0.1)
    horizontal_tolerance_ratio : float
        Allowed vertical deviation from OD center as fraction of image height
    return_centers : bool
        If True, also return estimated centers

    Returns:
    --------
    dict with keys:
        'mask': Clean segmentation mask (H, W) with 0=bg, 1=OD, 2=macula
        'od_center': (x, y) coordinates of OD center
        'macula_center': (x, y) coordinates of macula/fovea center
        'od_valid': bool indicating if OD was found
        'macula_valid': bool indicating if macula was found
    """

    H, W = pred_probs.shape[1:]
    if image_shape is None:
        image_shape = (H, W)

    image_center = np.array([W / 2, H / 2])

    # Expected sizes (in pixels)
    image_area = H * W
    expected_od_area = (
        image_area * (od_size_ratio**2) * np.pi / 4
    )  # Circular approximation
    expected_macula_area = expected_od_area * (macula_od_ratio**2)

    # Size tolerance bounds
    od_min_area = expected_od_area * 0.1
    od_max_area = expected_od_area * 5.0
    macula_min_area = max(10, expected_macula_area * 0.05)  # At least 10 pixels
    macula_max_area = expected_macula_area * 20.0

    # Initialize output
    clean_mask = np.zeros((H, W), dtype=np.uint8)
    result = {
        "mask": clean_mask,
        "od_center": None,
        "macula_center": None,
        "od_valid": False,
        "macula_valid": False,
    }

    # Get hard predictions
    pred_argmax = pred_probs.argmax(axis=0)

    # =========================================================================
    # 1. PROCESS OPTIC DISC
    # =========================================================================

    od_mask = (pred_argmax == 1).astype(np.uint8)
    od_center, od_clean_mask = _select_best_od(
        od_mask,
        image_center,
        od_min_area,
        od_max_area,
        pred_probs[1],  # OD probability map for confidence weighting
    )

    if od_center is not None:
        clean_mask[od_clean_mask > 0] = 1
        result["od_center"] = od_center
        result["od_valid"] = True

    # =========================================================================
    # 2. PROCESS MACULA
    # =========================================================================

    macula_mask = (pred_argmax == 2).astype(np.uint8)
    macula_center, macula_clean_mask = _select_best_macula(
        macula_mask,
        od_center,
        image_center,
        H * horizontal_tolerance_ratio,
        macula_min_area,
        macula_max_area,
        pred_probs[2],  # Macula probability map
    )

    if macula_center is not None:
        clean_mask[macula_clean_mask > 0] = 2
        result["macula_center"] = macula_center
        result["macula_valid"] = True

    result["mask"] = clean_mask

    return result


def _select_best_od(
    od_mask: np.ndarray,
    image_center: np.ndarray,
    min_area: float,
    max_area: float,
    prob_map: np.ndarray = None,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Select the best OD candidate from potentially multiple connected components.

    Rules:
    - Filter by size (must be reasonable)
    - Score by: size (larger is better) + proximity to center (closer is better)
    """

    labeled_mask, num_cc = ndimage.label(od_mask)

    if num_cc == 0:
        return None, np.zeros_like(od_mask)

    regions = measure.regionprops(labeled_mask, intensity_image=prob_map)

    # Filter and score candidates
    candidates = []
    for region in regions:
        area = region.area

        # Size filter
        if area < min_area or area > max_area:
            continue

        # Centroid (regionprops gives (y, x))
        cy, cx = region.centroid
        center = np.array([cx, cy])

        # Distance to image center (normalized)
        dist_to_center = np.linalg.norm(center - image_center)
        max_dist = np.linalg.norm(image_center)  # Corner to center
        normalized_dist = dist_to_center / max_dist

        # Normalized area score (0-1, larger is better)
        normalized_area = min(area / max_area, 1.0)

        # Mean probability (confidence)
        mean_prob = region.mean_intensity if prob_map is not None else 1.0

        # Combined score: prioritize size, then proximity, then confidence
        # Lower distance is better, so we invert it
        score = 0.4 * normalized_area + 0.4 * (1 - normalized_dist) + 0.2 * mean_prob

        candidates.append(
            {
                "label": region.label,
                "center": center,
                "area": area,
                "score": score,
            }
        )

    if not candidates:
        return None, np.zeros_like(od_mask)

    # Select best candidate
    best = max(candidates, key=lambda x: x["score"])

    # Create clean mask with only the selected OD
    clean_mask = (labeled_mask == best["label"]).astype(np.uint8)

    return best["center"], clean_mask


def _select_best_macula(
    macula_mask: np.ndarray,
    od_center: Optional[np.ndarray],
    image_center: np.ndarray,
    horizontal_tolerance: float,
    min_area: float,
    max_area: float,
    prob_map: np.ndarray = None,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Select the best macula candidate.

    Rules:
    - Must be roughly horizontally aligned with OD (if OD is valid)
    - If multiple nearby CCs, fovea is likely at their center
    - Filter by size
    """

    labeled_mask, num_cc = ndimage.label(macula_mask)

    if num_cc == 0:
        return None, np.zeros_like(macula_mask)

    regions = measure.regionprops(labeled_mask, intensity_image=prob_map)

    # Extract all candidate info first
    all_candidates = []
    for region in regions:
        area = region.area
        cy, cx = region.centroid
        center = np.array([cx, cy])
        mean_prob = region.mean_intensity if prob_map is not None else 1.0

        all_candidates.append(
            {
                "label": region.label,
                "center": center,
                "area": area,
                "mean_prob": mean_prob,
            }
        )

    if not all_candidates:
        return None, np.zeros_like(macula_mask)

    # =========================================================================
    # Check for clustered components (multiple nearby detections)
    # =========================================================================

    centers = np.array([c["center"] for c in all_candidates])

    if len(centers) > 1:
        # Check if candidates are clustered
        from scipy.spatial.distance import pdist

        pairwise_dists = pdist(centers)

        # If multiple CCs are close (within ~2x expected macula diameter), merge them
        expected_macula_diameter = np.sqrt(max_area / np.pi) * 2
        clustering_threshold = expected_macula_diameter * 3

        if pairwise_dists.min() < clustering_threshold:
            # Candidates are clustered - compute weighted center
            weights = np.array([c["area"] * c["mean_prob"] for c in all_candidates])
            weights = weights / weights.sum()

            merged_center = np.average(centers, axis=0, weights=weights)

            # Create merged mask of all nearby components
            merged_mask = np.zeros_like(macula_mask)
            for c in all_candidates:
                if np.linalg.norm(c["center"] - merged_center) < clustering_threshold:
                    merged_mask[labeled_mask == c["label"]] = 1

            # Validate horizontal alignment with OD
            if od_center is not None:
                vertical_offset = abs(merged_center[1] - od_center[1])
                if vertical_offset > horizontal_tolerance:
                    # Merged center doesn't align - fall back to individual selection
                    pass
                else:
                    return merged_center, merged_mask

    # =========================================================================
    # Standard selection: filter and score individual candidates
    # =========================================================================

    valid_candidates = []
    for c in all_candidates:
        area = c["area"]
        center = c["center"]

        # Size filter
        if area < min_area or area > max_area:
            continue

        # Horizontal alignment with OD
        if od_center is not None:
            vertical_offset = abs(center[1] - od_center[1])
            if vertical_offset > horizontal_tolerance:
                continue

            # Macula should be temporal to OD (but we don't know laterality)
            # Just check it's not overlapping
            dist_to_od = np.linalg.norm(center - od_center)
            if dist_to_od < np.sqrt(max_area / np.pi):  # Too close to OD
                continue

        # Score: prefer higher probability, appropriate size
        normalized_area = 1.0 - abs(area - (min_area + max_area) / 2) / max_area
        score = 0.6 * c["mean_prob"] + 0.4 * normalized_area

        valid_candidates.append(
            {
                **c,
                "score": score,
            }
        )

    if not valid_candidates:
        # Fallback: if no valid candidates after filtering, take highest probability
        best = max(all_candidates, key=lambda x: x["mean_prob"])
        clean_mask = (labeled_mask == best["label"]).astype(np.uint8)
        return best["center"], clean_mask

    # Select best
    best = max(valid_candidates, key=lambda x: x["score"])
    clean_mask = (labeled_mask == best["label"]).astype(np.uint8)

    return best["center"], clean_mask


# =============================================================================
# BATCH PROCESSING WRAPPER
# =============================================================================


def postprocess_batch(
    pred_batch: torch.Tensor,
    return_tensors: bool = True,
) -> Dict:
    """
    Post-process a batch of predictions.

    Parameters:
    -----------
    pred_batch : torch.Tensor
        Predictions of shape (B, 3, H, W)
    return_tensors : bool
        If True, return torch tensors; otherwise numpy arrays

    Returns:
    --------
    dict with:
        'masks': (B, H, W) clean segmentation masks
        'od_centers': list of (x, y) or None for each image
        'macula_centers': list of (x, y) or None for each image
        'od_valid': (B,) boolean array
        'macula_valid': (B,) boolean array
    """

    if isinstance(pred_batch, torch.Tensor):
        pred_np = pred_batch.cpu().numpy()
    else:
        pred_np = pred_batch

    B, C, H, W = pred_np.shape

    masks = np.zeros((B, H, W), dtype=np.uint8)
    od_centers = []
    macula_centers = []
    od_valid = np.zeros(B, dtype=bool)
    macula_valid = np.zeros(B, dtype=bool)

    for i in range(B):
        result = postprocess_od_macula(pred_np[i])

        masks[i] = result["mask"]
        od_centers.append(result["od_center"])
        macula_centers.append(result["macula_center"])
        od_valid[i] = result["od_valid"]
        macula_valid[i] = result["macula_valid"]

    output = {
        "masks": torch.from_numpy(masks) if return_tensors else masks,
        "od_centers": od_centers,
        "macula_centers": macula_centers,
        "od_valid": torch.from_numpy(od_valid) if return_tensors else od_valid,
        "macula_valid": torch.from_numpy(macula_valid)
        if return_tensors
        else macula_valid,
    }

    return output
