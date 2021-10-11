#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wednesday - February 17 2021, 16:40:14

@authors:
* Michele Svanera, University of Glasgow
* Mattia Savardi, University of Brescia

Functions to augment training data.
"""

import numpy as np
from scipy.ndimage import affine_transform
from scipy.ndimage import rotate
import albumentations as albu
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
from cerebrum.config import Config
#import cerebrum


#print(dir(albu.core.transforms_interface ))

def addWeighted(src1, alpha, src2, beta, gamma):
    """ 
    Calculates the weighted sum of two arrays (cv2 replaced).

    :param src1: first input array.
    :param aplha: weight of the first array elements.
    :param src2: second input array of the same size and channel number as src1.
    :param beta: weight of the second array elements.
    :param gamma: scalar added to each sum
    :return: output array that has the same size and number of channels as the input arrays.
    """

    return src1 * alpha + src2 * beta + gamma


def augmentation_salt_and_pepper_noise(X_data, amount=10. / 1000):
    """ 
    Function to add S&P noise to the volume.

    :param X_data: input volume (3D) -> shape (x,y,z)
    :param amount: quantity of voxels affected
    :return X_data_out: augmented volume
    """

    X_data_out = X_data
    salt_vs_pepper = 0.2  # Ration between salt and pepper voxels
    n_salt_voxels = int(np.ceil(amount * np.prod(X_data_out.size) * salt_vs_pepper))
    n_pepper_voxels = int(np.ceil(amount * np.prod(X_data_out.size) * (1.0 - salt_vs_pepper)))

    # Add Salt noise
    coords = [np.random.randint(0, i - 1, int(n_salt_voxels)) for i in np.squeeze(X_data_out).shape]
    X_data_out[coords[0], coords[1], coords[2]] = np.max(X_data)

    # Add Pepper noise
    coords = [np.random.randint(0, i - 1, int(n_pepper_voxels)) for i in np.squeeze(X_data_out).shape]
    X_data_out[coords[0], coords[1], coords[2]] = np.min(X_data)

    return X_data_out


class SaltAndPepperNoiseAugment(ImageOnlyTransform):
    def apply(self, img, **params):
        return augmentation_salt_and_pepper_noise(img)


def augmentation_gaussian_noise(X_data):
    """ 
    Function to add gaussian noise to the volume.

    :param X_data: input volume (3D) -> shape (x,y,z)
    :return X_data_out: augmented volume
    """

    # Gaussian distribution parameters
    X_data_no_background = X_data
    mean = np.mean(X_data_no_background)
    var = np.var(X_data_no_background)
    sigma = var ** 0.5

    gaussian = np.random.normal(mean, sigma, X_data.shape).astype(X_data.dtype)

    # Compose the output (src1, alpha, src2, beta, gamma)
    X_data_out = addWeighted(X_data, 0.8, gaussian, 0.2, 0)

    return X_data_out


class GaussianNoiseAugment(ImageOnlyTransform):
    def apply(self, img, **params):
        return augmentation_gaussian_noise(img)


def augmentation_inhomogeneity_noise(X_data, inhom_vol):
    """ 
    Function to add inhomogeneity noise to the volume.

    :param X_data: input volume (3D) -> shape (x,y,z)
    :param inhom_vol: inhomogeneity volume (preloaded)
    :return X_data_out: augmented volume
    """

    # Randomly select a vol of the same shape of 'X_data'
    x_1 = np.random.randint(0, int(X_data.shape[0]) - 1, size=1)[0]
    x_2 = np.random.randint(0, int(X_data.shape[1]) - 1, size=1)[0]
    x_3 = np.random.randint(0, int(X_data.shape[2]) - 1, size=1)[0]
    y_1 = inhom_vol[x_1: x_1 + X_data.shape[0],
          x_2: x_2 + X_data.shape[1],
          x_3: x_3 + X_data.shape[2]]

    # Compose the output: add noise to the original vol
    X_data_out = X_data + y_1.astype(X_data.dtype)

    return X_data_out


class InhomogeneityNoiseAugment(ImageOnlyTransform):

    def __init__(self, inhom_vol: np.array, always_apply=False, p=1.0):
        super(InhomogeneityNoiseAugment, self).__init__(always_apply, p)
        self.inhom_vol = inhom_vol

    def apply(self, img, **params):
        return augmentation_inhomogeneity_noise(img, self.inhom_vol)


def translate_volume(image,
                     shift_x0: int, shift_x1: int, shift_x2: int,
                     padding_mode: str = 'nearest',
                     spline_interp_order: int = 1):
    """ 
    Function to apply volume translation to a single volume.

    :param image: input volume (3D) -> shape (x,y,z)
    :param shift_x0-shift_x1-shift_x2: shift in voxels
    :param padding_mode: the padding mode
    :param spline_interp_order: order for the affine transformation
    :return: augmented volume
    """

    # Set the affine transformation matrix
    M_t = np.eye(4)
    M_t[:-1, -1] = np.array([-shift_x0, -shift_x1, -shift_x2])

    return affine_transform(image, M_t,
                            order=spline_interp_order,
                            mode=padding_mode,
                            cval=0,
                            output_shape=image.shape)


class TranslationAugment(DualTransform):
    """ Class to deal with translation augmentation. """

    def __init__(self, max_shift: list = [20, 20, 20], always_apply=False, p=1.0):
        super(TranslationAugment, self).__init__(always_apply, p)
        self.max_shift = max_shift

    def get_params(self):

        # Randomly select parameters
        try:
            shifts = [(np.random.RandomState().randint(2 * i) - i) for i in self.max_shift]
            shift_x0, shift_x1, shift_x2 = shifts
        except:
            shift_x0, shift_x1, shift_x2 = [0] * 3

        return {"shift_x0": shift_x0, "shift_x1": shift_x1, "shift_x2": shift_x2}

    def apply(self, img, shift_x0: int = 0, shift_x1: int = 0, shift_x2: int = 0, **params):

        # Apply to image or mask
        if np.issubdtype(img.dtype, np.floating):  # image
            img_out = translate_volume(img,
                                       shift_x0, shift_x1, shift_x2,
                                       padding_mode='nearest',
                                       spline_interp_order=1)
        elif np.issubdtype(img.dtype, np.integer):  # mask
            img_out = translate_volume(img,
                                       shift_x0, shift_x1, shift_x2,
                                       padding_mode='constant',
                                       spline_interp_order=0)
        else:
            raise Exception('Error 23: type not supported.')

        return img_out


class RotationAugment(DualTransform):
    """ Class to deal with rotation augmentation. """

    def __init__(self,
                 max_angle: int = 10,
                 rot_spline_order: int = 3,
                 always_apply=False,
                 p=1.0):
        super(RotationAugment, self).__init__(always_apply, p)
        self.max_angle = max_angle
        self.rot_spline_order = rot_spline_order

    def get_params(self):

        # Randomly select parameters
        random_angle = np.random.RandomState().randint(2 * self.max_angle) - self.max_angle
        rot_axes = np.random.RandomState().permutation(range(3))[:2]  # random select the 2 rotation axes

        return {"random_angle": random_angle, "rot_axes": rot_axes}

    def apply(self, img, random_angle: int, rot_axes: int, **params):

        # Apply to image or mask
        if np.issubdtype(img.dtype, np.floating):  # image
            img_out = rotate(input=img,
                             angle=random_angle,
                             axes=rot_axes,
                             reshape=False,
                             order=self.rot_spline_order,
                             mode='nearest',
                             prefilter=True)
        elif np.issubdtype(img.dtype, np.integer):  # mask
            img_out = rotate(input=img,
                             angle=random_angle,
                             axes=rot_axes,
                             reshape=False,
                             order=0,
                             mode='constant',
                             prefilter=True)
        else:
            raise Exception('Error 24: type not supported.')

        return img_out


class GhostingAugment(ImageOnlyTransform):
    """ Class to deal with ghosting augmentation. """

    def __init__(self,
                 max_repetitions: int = 4,
                 always_apply=False,
                 p=1.0):
        super(GhostingAugment, self).__init__(always_apply, p)
        self.max_repetitions = max_repetitions

    def apply(self, img, **params):
        # Randomly select parameters
        repetitions = np.random.RandomState().choice(range(1, self.max_repetitions + 1))
        axis = np.random.RandomState().choice(range(len(img.shape)))

        img_out = img
        shift_value = 0
        for i_rep in range(1, repetitions + 1):
            # Compute the shift to apply to the data
            shift_value += int(img.shape[axis] / (i_rep + 1))

            # Shift the data and add to the out volume
            data_repetition = np.roll(img, shift_value, axis=axis)
            img_out = addWeighted(img_out, 0.85, data_repetition, 0.15, 0)

        return img_out


def get_augm_transforms(inho_vol, config: Config, volume_size: int = 256):
    """
    Get the transformations for volume (and mask) augmentation.

    :param inho_vol: inhomogeneity volume
    :param volume_size: size of the volume
    :param config: Config class (with all probabilities stored)
    :return: albumentation composition
    """

    return albu.Compose([

        # Default transformations
        albu.VerticalFlip(p = config.augment.prob_flip),  # sagittal plane
        InhomogeneityNoiseAugment(inho_vol, p = config.augment.prob_inho),  # Inhomogeneity noise

        # Geometric transformations
        albu.OneOf([
            albu.GridDistortion(num_steps = 5,
                                distort_limit = (-0.10, +0.10),
                                interpolation = 4,
                                border_mode = 1,
                                p = config.augment.prob_grid),
            albu.RandomResizedCrop(height = volume_size,
                                   width = volume_size,
                                   scale = (0.9, 1.0),
                                   ratio = (0.8, 1.20),
                                   interpolation = 4,
                                   p = config.augment.prob_resi),
            RotationAugment(p = config.augment.prob_rota),
            TranslationAugment(p = config.augment.prob_tran),
        ], p = config.augment.prob_geom),

        # Color transformations
        albu.OneOf([
            albu.Blur(blur_limit = (3, 3), p = config.augment.prob_blur),
            albu.Downscale(scale_min = 0.6, 
                           scale_max = 0.99, 
                           interpolation = 4, 
                           p = config.augment.prob_down),
            SaltAndPepperNoiseAugment(p = config.augment.prob_salt),
            GaussianNoiseAugment(p = config.augment.prob_gaus),
            GhostingAugment(p = config.augment.prob_ghos),
        ], p = config.augment.prob_colo),

    ], p = config.augment.prob_overall)
    
