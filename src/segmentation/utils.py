import numpy as np
import tensorflow as tf
import nibabel as nib
from scipy.ndimage import center_of_mass
def Load_nifti(data_path):
    return nib.load(data_path)

def z_score_normalize(image, mask=None):
    valid_pixels = image[mask > 0] if mask is not None and np.sum(mask > 0) > 0 else image.flatten()
    mean, std = np.mean(valid_pixels), np.std(valid_pixels)
    return (image - mean) / (std if std > 0 else 1.0)

def resize_image(image, target_shape=(128, 128, 128)):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=-1)
    image = tf.image.resize(image, target_shape[:2], method='bilinear')
    image = tf.transpose(image, [2, 0, 1, 3])
    image = tf.image.resize(image, [target_shape[2], target_shape[1]], method='bilinear')
    image = tf.transpose(image, [1, 2, 0, 3])
    return image

def preprocess_sample(t1c, t1n, t2f, t2w):
    brain_mask = t1c > np.percentile(t1c, 5)
    t1c_norm = z_score_normalize(t1c, brain_mask)
    t1n_norm = z_score_normalize(t1n, brain_mask)
    t2f_norm = z_score_normalize(t2f, brain_mask)
    t2w_norm = z_score_normalize(t2w, brain_mask)
    t1c_resized, t1n_resized = resize_image(t1c_norm), resize_image(t1n_norm)
    t2f_resized, t2w_resized = resize_image(t2f_norm), resize_image(t2w_norm)
    image_tensor = tf.stack([tf.squeeze(t) for t in [t1c_resized, t1n_resized, t2f_resized, t2w_resized]], axis=-1)
    return image_tensor

def analyze_segmentation(seg_mask_data, affine):
    voxel_volume_cm3 = np.abs(np.linalg.det(affine[:3, :3])) / 1000.0
    ncr_voxels = np.sum(seg_mask_data == 1)
    ed_voxels = np.sum(seg_mask_data == 2)
    et_voxels = np.sum(seg_mask_data == 3)
    et_volume = et_voxels * voxel_volume_cm3
    tc_volume = (ncr_voxels + et_voxels) * voxel_volume_cm3
    wt_volume = (ncr_voxels + et_voxels + ed_voxels) * voxel_volume_cm3
    whole_tumor_mask = (seg_mask_data > 0)
    centroid_mm = [0, 0, 0]
    if np.sum(whole_tumor_mask) > 0:
        centroid_voxels = center_of_mass(whole_tumor_mask)
        centroid_mm = nib.affines.apply_affine(affine, centroid_voxels)
    return {
        "whole_tumor_volume_cm3": wt_volume,
        "enhancing_tumor_volume_cm3": et_volume,
        "tumor_core_volume_cm3": tc_volume,
        "tumor_centroid_mm": [round(c, 2) for c in centroid_mm]
    }