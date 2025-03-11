# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import itertools
import math
import random
from typing import Any
import cv2

import cv2
import numpy as np
import torch
from scipy import special
from scipy.stats import multivariate_normal
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as F_vision
from torchvision.transforms.functional import rgb_to_grayscale


y_table = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
).T
y_table = nn.Parameter(torch.from_numpy(y_table))
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]]).T
c_table = nn.Parameter(torch.from_numpy(c_table))


def _cubic(x: float) -> float:
    """Python implements `cubic()` in Matlab

    Args:
        x (float): pixel value

    Returns:
        cubic (float): bicubic core

    """
    abs_x = torch.abs(x)
    abs_x2 = abs_x**2
    abs_x3 = abs_x**3
    cubic1 = (1.5 * abs_x3 - 2.5 * abs_x2 + 1) * ((abs_x <= 1).type_as(abs_x))
    cubic2 = (-0.5 * abs_x3 + 2.5 * abs_x2 - 4 * abs_x + 2) * (((abs_x > 1) * (abs_x <= 2)).type_as(abs_x))
    cubic = cubic1 + cubic2

    return cubic


def _mesh_grid(kernel_size: int) -> [np.ndarray, np.ndarray, np.ndarray]:
    """Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector fields
    over N-D grids, given one-dimensional coordinate arrays x1, x2,..., xn.

    Args:
        kernel_size (int): Gaussian kernel size

    Returns:
        xy (np.ndarray): Gaussian kernel with this scale ->(kernel_size, kernel_size, 2)
        xx (np.ndarray): Gaussian kernel with this scale ->(kernel_size, kernel_size)
        yy (np.ndarray): Gaussian kernel with this scale ->(kernel_size, kernel_size)

    """
    ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)), yy.reshape(kernel_size * kernel_size, 1))).reshape(
        kernel_size, kernel_size, 2
    )

    return xy, xx, yy


def _calculate_weights_indices(
    in_length: int, out_length: int, scale: float, kernel_width: int, antialiasing: bool
) -> [np.ndarray, np.ndarray, int, int]:
    """Implementation of `calculate_weights_indices()` in Matlab under Python language.

    Args:
        in_length (int): Input length.
        out_length (int): Output length.
        scale (float): Scale factor.
        kernel_width (int): Kernel width.
        antialiasing (bool): Whether to apply antialiasing when down-sampling operations.
            Caution: Bicubic down-sampling in PIL uses antialiasing by default.

    Returns:

    """
    if (scale < 1) and antialiasing:
        # Use a modified kernel (larger kernel width) to simultaneously
        # interpolate and antialiasing
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5 + scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    p = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, p) + torch.linspace(0, p - 1, p).view(1, p).expand(
        out_length, p
    )

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, p) - indices

    # apply cubic kernel
    if (scale < 1) and antialiasing:
        weights = scale * _cubic(distance_to_center * scale)
    else:
        weights = _cubic(distance_to_center)

    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, p)

    # If a column in weights is all zero, get rid of it. only consider the
    # first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, p - 2)
        weights = weights.narrow(1, 1, p - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, p - 2)
        weights = weights.narrow(1, 0, p - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1

    return weights, indices, int(sym_len_s), int(sym_len_e)


def _calculate_rotate_sigma_matrix(sigma_x: float, sigma_y: float, theta: float) -> np.ndarray:
    """Calculate rotated sigma matrix

    Args:
        sigma_x (float): Sigma value in the horizontal axis direction
        sigma_y (float): sigma value along the vertical axis
        theta (float): Radian measurement

    Returns:
        out (np.ndarray): Rotated sigma matrix

    """
    d_matrix = np.array([[sigma_x**2, 0], [0, sigma_y**2]])
    u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    out = np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))

    return out


def _calculate_probability_density(sigma_matrix: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Calculate probability density function of the bivariate Gaussian distribution

    Args:
        sigma_matrix (np.ndarray): with the shape (2, 2)
        grid (np.ndarray): generated by :func:`mesh_grid`, with the shape (K, K, 2), K is the kernel size

    Returns:
        probability_density (np.ndarray): un-normalized kernel

    """
    inverse_sigma = np.linalg.inv(sigma_matrix)
    probability_density = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))

    return probability_density


def _calculate_cumulative_density(skew_matrix: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Calculate the CDF of the standard bivariate Gaussian distribution.
    Used in skewed Gaussian distribution

    Args:
        skew_matrix (np.ndarray): skew matrix
        grid (np.ndarray): generated by :func:`mesh_grid`, with the shape (K, K, 2), K is the kernel size

    Returns:
        cumulative_density (np.ndarray): Cumulative density function

    """
    rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
    grid = np.dot(grid, skew_matrix)
    cumulative_density = rv.cdf(grid)

    return cumulative_density


def _generate_bivariate_gaussian_kernel(
    kernel_size: int, sigma_x: float, sigma_y: float, theta: float, grid: np.ndarray = None, isotropic: bool = True
) -> np.ndarray:
    """Generate a bivariate isotropic or anisotropic Gaussian kernel

    Args:
        kernel_size (int): Gaussian kernel size
        sigma_x (float): Sigma value in the horizontal axis direction
        sigma_y (float): sigma value along the vertical axis
        theta (float): Radian measurement
        grid (optional, np.ndarray): generated by :func:`mesh_grid`, with the shape (K, K, 2), K is the kernel size. Default: ``None``
        isotropic (optional, bool): Set to `True` for homosexual Gaussian kernel, set to `False` for heterosexual Gaussian kernel. Default: ``True``

    Returns:
        bivariate_gaussian_kernel (np.ndarray): Bivariate gaussian kernel

    """
    if grid is None:
        grid, _, _ = _mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sigma_x**2, 0], [0, sigma_x**2]])
    else:
        sigma_matrix = _calculate_rotate_sigma_matrix(sigma_x, sigma_y, theta)

    bivariate_gaussian_kernel = _calculate_probability_density(sigma_matrix, grid)
    bivariate_gaussian_kernel = bivariate_gaussian_kernel / np.sum(bivariate_gaussian_kernel)

    return bivariate_gaussian_kernel


def _generate_bivariate_generalized_gaussian_kernel(
    kernel_size: int,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    beta: float,
    grid: np.ndarray = None,
    isotropic: bool = True,
) -> np.ndarray:
    """Generate a bivariate generalized Gaussian kernel

    Args:
        kernel_size (int): Gaussian kernel size
        sigma_x (float): Sigma value of the horizontal axis
        sigma_y (float): Sigma value of the vertical axis
        theta (float): Radian measurement
        beta (float): shape parameter, beta = 1 is the normal distribution
        grid (optional, np.ndarray): generated by :func:`mesh_grid`, with the shape (K, K, 2), K is the kernel size. Default: None
        isotropic (optional, bool): Set to `True` for homosexual Gaussian kernel, set to `False` for heterosexual Gaussian kernel. (Default: ``True``)

    Returns:
        bivariate_generalized_gaussian_kernel (np.ndarray): bivariate generalized gaussian kernel

    """
    if grid is None:
        grid, _, _ = _mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sigma_x**2, 0], [0, sigma_x**2]])
    else:
        sigma_matrix = _calculate_rotate_sigma_matrix(sigma_x, sigma_y, theta)

    inverse_sigma = np.linalg.inv(sigma_matrix)
    bivariate_generalized_gaussian_kernel = np.exp(-0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta))
    bivariate_generalized_gaussian_kernel = bivariate_generalized_gaussian_kernel / np.sum(
        bivariate_generalized_gaussian_kernel
    )

    return bivariate_generalized_gaussian_kernel


def _generate_bivariate_plateau_gaussian_kernel(
    kernel_size: int,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    beta: float,
    grid: np.ndarray = None,
    isotropic: bool = True,
) -> np.ndarray:
    """Generate a plateau-like anisotropic kernel

    Args:
        kernel_size (int): Gaussian kernel size
        sigma_x (float): Sigma value of the horizontal axis
        sigma_y (float): Sigma value of the vertical axis
        theta (float): Radian measurement
        beta (float): shape parameter, beta = 1 is the normal distribution
        grid (optional, np.ndarray): generated by :func:`mesh_grid`, with the shape (K, K, 2), K is the kernel size. Default: None
        isotropic (optional, bool): Set to `True` for homosexual Gaussian kernel, set to `False` for heterosexual Gaussian kernel. (Default: ``True``)

    Returns:
        bivariate_plateau_gaussian_kernel (np.ndarray): bivariate plateau gaussian kernel

    """
    if grid is None:
        grid, _, _ = _mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sigma_x**2, 0], [0, sigma_x**2]])
    else:
        sigma_matrix = _calculate_rotate_sigma_matrix(sigma_x, sigma_y, theta)

    inverse_sigma = np.linalg.inv(sigma_matrix)
    bivariate_plateau_gaussian_kernel = np.reciprocal(np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta) + 1)
    bivariate_plateau_gaussian_kernel = bivariate_plateau_gaussian_kernel / np.sum(bivariate_plateau_gaussian_kernel)

    return bivariate_plateau_gaussian_kernel


def _random_bivariate_gaussian_kernel(
    kernel_size: int,
    sigma_x_range: tuple,
    sigma_y_range: tuple,
    rotation_range: tuple,
    noise_range: tuple = None,
    isotropic: bool = True,
) -> np.ndarray:
    """Randomly generate bivariate isotropic or anisotropic Gaussian kernels

    Args:
        kernel_size (int): Gaussian kernel size
        sigma_x_range (float): Sigma range along the horizontal axis
        sigma_y_range (float): Sigma range along the vertical axis
        rotation_range (tuple): Gaussian kernel rotation matrix angle range value
        noise_range(optional, tuple): multiplicative kernel noise. Default: None
        isotropic (optional, bool): Set to `True` for homosexual plateau kernel, set to `False` for heterosexual plateau kernel. (Default: ``True``)

    Returns:
        bivariate_gaussian_kernel (np.ndarray): Bivariate gaussian kernel

    """
    assert kernel_size % 2 == 1, "Kernel size must be an odd number."
    assert sigma_x_range[0] < sigma_x_range[1], "Wrong sigma_x_range."
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])

    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], "Wrong sigma_y_range."
        assert rotation_range[0] < rotation_range[1], "Wrong rotation_range."
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    bivariate_gaussian_kernel = _generate_bivariate_gaussian_kernel(
        kernel_size, sigma_x, sigma_y, rotation, isotropic=isotropic
    )

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], "Wrong noise range."
        noise = np.random.uniform(noise_range[0], noise_range[1], size=bivariate_gaussian_kernel.shape)
        bivariate_gaussian_kernel = bivariate_gaussian_kernel * noise

    bivariate_gaussian_kernel = bivariate_gaussian_kernel / np.sum(bivariate_gaussian_kernel)

    return bivariate_gaussian_kernel


def _random_bivariate_generalized_gaussian_kernel(
    kernel_size: int,
    sigma_x_range: tuple,
    sigma_y_range: tuple,
    rotation_range: tuple,
    beta_range: tuple,
    noise_range: tuple = None,
    isotropic: bool = True,
) -> np.ndarray:
    """Randomly generate bivariate generalized Gaussian kernels

    Args:
        kernel_size (int): Gaussian kernel size
        sigma_x_range (float): Sigma range along the horizontal axis
        sigma_y_range (float): Sigma range along the vertical axis
        rotation_range (tuple): Gaussian kernel rotation matrix angle range value
        beta_range (tuple): Gaussian kernel beta matrix angle range value
        noise_range(optional, tuple): multiplicative kernel noise. Default: None
        isotropic (optional, bool): Set to `True` for homosexual plateau kernel, set to `False` for heterosexual plateau kernel. (Default: ``True``)

    Returns:
        bivariate_generalized_gaussian_kernel (np.ndarray): Bivariate generalized gaussian kernel

    """
    assert kernel_size % 2 == 1, "Kernel size must be an odd number."
    assert sigma_x_range[0] < sigma_x_range[1], "Wrong sigma_x_range."
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], "Wrong sigma_y_range."
        assert rotation_range[0] < rotation_range[1], "Wrong rotation_range."
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    bivariate_generalized_gaussian_kernel = _generate_bivariate_generalized_gaussian_kernel(
        kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic
    )

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], "Wrong noise range."
        noise = np.random.uniform(noise_range[0], noise_range[1], size=bivariate_generalized_gaussian_kernel.shape)
        bivariate_generalized_gaussian_kernel = bivariate_generalized_gaussian_kernel * noise

    bivariate_generalized_gaussian_kernel = bivariate_generalized_gaussian_kernel / np.sum(
        bivariate_generalized_gaussian_kernel
    )

    return bivariate_generalized_gaussian_kernel


def _random_bivariate_plateau_gaussian_kernel(
    kernel_size: int,
    sigma_x_range: tuple,
    sigma_y_range: tuple,
    rotation_range: tuple,
    beta_range: tuple,
    noise_range: tuple = None,
    isotropic: bool = True,
) -> np.ndarray:
    """Randomly generate bivariate plateau kernels

    Args:
        kernel_size (int): Gaussian kernel size
        sigma_x_range (float): Sigma range along the horizontal axis
        sigma_y_range (float): Sigma range along the vertical axis
        rotation_range (tuple): Gaussian kernel rotation matrix angle range value
        beta_range (tuple): Gaussian kernel beta matrix angle range value
        noise_range(tuple, optional): multiplicative kernel noise. Default: None
        isotropic (bool): Set to `True` for homosexual plateau kernel, set to `False` for heterosexual plateau kernel. (Default: ``True``)

    Returns:
        bivariate_plateau_gaussian_kernel (np.ndarray): Bivariate plateau gaussian kernel

    """
    assert kernel_size % 2 == 1, "Kernel size must be an odd number."
    assert sigma_x_range[0] < sigma_x_range[1], "Wrong sigma_x_range."
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])

    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], "Wrong sigma_y_range."
        assert rotation_range[0] < rotation_range[1], "Wrong rotation_range."
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    bivariate_plateau_gaussian_kernel = _generate_bivariate_plateau_gaussian_kernel(
        kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic
    )
    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], "Wrong noise range."
        noise = np.random.uniform(noise_range[0], noise_range[1], size=bivariate_plateau_gaussian_kernel.shape)
        bivariate_plateau_gaussian_kernel = bivariate_plateau_gaussian_kernel * noise

    bivariate_plateau_gaussian_kernel = bivariate_plateau_gaussian_kernel / np.sum(bivariate_plateau_gaussian_kernel)

    return bivariate_plateau_gaussian_kernel


def random_mixed_kernels(
    kernel_type: list,
    kernel_prob: float,
    kernel_size: int,
    sigma_x_range: list,
    sigma_y_range: list,
    rotation_range: list,
    generalized_kernel_beta_range: list,
    plateau_kernel_beta_range: list,
    noise_range: tuple = None,
) -> np.ndarray:
    """Randomly generate mixed kernels

    Args:
        kernel_type (list): a list name of gaussian kernel types
        kernel_prob (float): corresponding kernel probability for each kernel type
        kernel_size (int): Gaussian kernel size
        sigma_x_range (list): Sigma range along the horizontal axis
        sigma_y_range (list): Sigma range along the vertical axis
        rotation_range (list): Gaussian kernel rotation matrix angle range value
        generalized_kernel_beta_range (list): Gaussian kernel beta matrix angle range value
        plateau_kernel_beta_range (list): Plateau Gaussian Kernel Beta Matrix Angle Range Values
        noise_range (optional, tuple): multiplicative kernel noise, [0.75, 1.25]. Default: ``None``

    Returns:
        mixed kernels (np.ndarray): Mixed kernels

    """
    kernel_type = random.choices(kernel_type, kernel_prob)[0]
    if kernel_type == "isotropic":
        mixed_kernels = _random_bivariate_gaussian_kernel(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range, True
        )
    elif kernel_type == "anisotropic":
        mixed_kernels = _random_bivariate_gaussian_kernel(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range, False
        )
    elif kernel_type == "generalized_isotropic":
        mixed_kernels = _random_bivariate_generalized_gaussian_kernel(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, generalized_kernel_beta_range, noise_range, True
        )
    elif kernel_type == "generalized_anisotropic":
        mixed_kernels = _random_bivariate_generalized_gaussian_kernel(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, generalized_kernel_beta_range, noise_range, False
        )
    elif kernel_type == "plateau_isotropic":
        mixed_kernels = _random_bivariate_plateau_gaussian_kernel(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, plateau_kernel_beta_range, None, True
        )
    elif kernel_type == "plateau_anisotropic":
        mixed_kernels = _random_bivariate_plateau_gaussian_kernel(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, plateau_kernel_beta_range, None, False
        )
    else:
        mixed_kernels = _random_bivariate_gaussian_kernel(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range, True
        )

    return mixed_kernels


def generate_sinc_kernel(cutoff: float, kernel_size: int, padding: int = 0) -> np.ndarray:
    """2D sinc filter

    Args:
        cutoff (float): Cutoff frequency in radians (pi is max)
        kernel_size (int): Horizontal and vertical size, must be odd
        padding (optional, int): Pad kernel size to desired size, must be odd or zero. Default: 0

    Returns:
        sinc_kernel (np.ndarray): Sinc kernel

    """
    assert kernel_size % 2 == 1, "Kernel size must be an odd number."

    np.seterr(divide="ignore", invalid="ignore")

    sinc_kernel = np.fromfunction(
        lambda x, y: cutoff
        * special.j1(cutoff * np.sqrt((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2))
        / (2 * np.pi * np.sqrt((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2)),
        [kernel_size, kernel_size],
    )
    sinc_kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff**2 / (4 * np.pi)
    sinc_kernel = sinc_kernel / np.sum(sinc_kernel)

    if padding > kernel_size:
        pad_size = (padding - kernel_size) // 2
        sinc_kernel = np.pad(sinc_kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    return sinc_kernel


def _generate_gaussian_noise(image: np.ndarray, sigma: float = 10.0, gray_noise: bool = False) -> np.ndarray:
    """Generate Gaussian noise (OpenCV)

    Args:
        image (np.ndarray): Input image
        sigma (float): Noise scale (measured in range 255). Default: 10.0
        gray_noise (optional, bool): Whether to add grayscale noise. Default: ``False``

    Returns:
        gaussian_noise (np.array): Gaussian noise

    """
    if gray_noise:
        gaussian_noise = np.float32(np.random.randn(*(image.shape[0:2]))) * sigma / 255.0
        gaussian_noise = np.expand_dims(gaussian_noise, axis=2).repeat(3, axis=2)
    else:
        gaussian_noise = np.float32(np.random.randn(*image.shape)) * sigma / 255.0

    return gaussian_noise


def _generate_poisson_noise(image: np.ndarray, scale: float = 1.0, gray_noise: bool = False) -> np.ndarray:
    """Generate poisson noise (OpenCV)

    Args:
        image (np.ndarray): Input image
        scale (optional, float): Noise scale value. Default: 1.0
        gray_noise (optional, bool): Whether to add grayscale noise. Default: ``False``

    Returns:
        poisson_noise (np.array): Poisson noise

    """
    if gray_noise:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Round and clip image for counting vals correctly
    image = np.clip((image * 255.0).round(), 0, 255) / 255.0
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    out = np.float32(np.random.poisson(image * vals) / float(vals))
    noise = out - image

    if gray_noise:
        noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)

    poisson_noise = noise * scale

    return poisson_noise


def _random_generate_gaussian_noise(image: np.ndarray, sigma_range: tuple = (0, 10), gray_prob: int = 0) -> np.ndarray:
    """Random generate gaussian noise (OpenCV)

    Args:
        image (np.ndarray): Input image
        sigma_range (optional, tuple): Noise range. Default: (0, 10)
        gray_prob (optional, int): Add grayscale noise probability. Default: 0

    Returns:
        gaussian_noise (np.ndarray): Gaussian noise

    """
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])

    if np.random.uniform() < gray_prob:
        gray_noise = True
    else:
        gray_noise = False

    gaussian_noise = _generate_gaussian_noise(image, sigma, gray_noise)

    return gaussian_noise


def _random_generate_poisson_noise(image: np.ndarray, scale_range: tuple = (0, 1.0), gray_prob: int = 0) -> np.ndarray:
    """Random generate poisson noise (OpenCV)

    Args:
        image (np.ndarray): Input image
        scale_range (optional, tuple): Noise scale range. Default: (0, 1.0)
        gray_prob (optional, int): Add grayscale noise probability. Default: 0

    Returns:
        poisson_noise (np.ndarray): Poisson noise

    """
    scale = np.random.uniform(scale_range[0], scale_range[1])

    if np.random.uniform() < gray_prob:
        gray_noise = True
    else:
        gray_noise = False

    poisson_noise = _generate_poisson_noise(image, scale, gray_noise)

    return poisson_noise


def _add_gaussian_noise(
    image: np.ndarray, sigma: float = 10.0, clip: bool = True, rounds: bool = False, gray_noise: bool = False
):
    """Add gaussian noise to image (OpenCV)

    Args:
        image (np.ndarray): Input image
        sigma (optional, float): Noise scale (measured in range 255). Default: 10.0
        clip (optional, bool): Whether to clip image pixel. If `True`, clip image pixel to [0, 1] or [0, 255]. Default: ``True``
        rounds (optional, bool): Gaussian noise rounds scale. Default: ``False``
        gray_noise (optional, bool): Whether to add grayscale noise. Default: ``False``

    Returns:
        out (np.ndarray): Add gaussian noise to image

    """
    noise = _generate_gaussian_noise(image, sigma, gray_noise)
    out = image + noise

    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.0
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.0

    return out


def _add_poisson_noise(
    image: np.ndarray, scale: float = 1.0, clip: bool = True, rounds: bool = False, gray_noise: bool = False
):
    """Add poisson noise to image (OpenCV)

    Args:
        image (np.ndarray): Input image
        scale (optional, float): Noise scale value. Default: 1.0
        clip (optional, bool): Whether to clip image pixel. If `True`, clip image pixel to [0, 1] or [0, 255]. Default: ``True``
        rounds (optional, bool): Gaussian noise rounds scale. Default: ``False``
        gray_noise (optional, bool): Whether to add grayscale noise. Default: ``False``

    Returns:
        out (np.ndarray): Add poisson noise to image

    """
    noise = _generate_poisson_noise(image, scale, gray_noise)
    out = image + noise

    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.0
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.0

    return out


def _random_add_gaussian_noise(
    image: np.ndarray, sigma_range: tuple = (0, 1.0), gray_prob: int = 0, clip: bool = True, rounds: bool = False
) -> np.ndarray:
    """Random add gaussian noise to image (OpenCV)

    Args:
        image (np.ndarray): Input image
        sigma_range (optional, tuple): Noise range. Default: (0, 1.0)
        gray_prob (optional, int): Add grayscale noise probability. Default: 0
        clip (optional, bool): Whether to clip image pixel. If `True`, clip image pixel to [0, 1] or [0, 255]. Default: ``True``
        rounds (optional, bool): Noise rounds scale. Default: ``False``

    Returns:
        out (np.ndarray): Add gaussian noise to image

    """
    noise = _random_generate_gaussian_noise(image, sigma_range, gray_prob)
    out = image + noise

    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.0
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.0

    return out


def _random_add_poisson_noise(
    image: np.ndarray, scale_range: tuple = (0, 1.0), gray_prob: int = 0, clip: bool = True, rounds: bool = False
) -> np.ndarray:
    """Random add gaussian noise to image (OpenCV)

    Args:
        image (np.ndarray): Input image
        scale_range (optional, tuple): Noise scale range. Default: (0, 1.0)
        gray_prob (optional, int): Add grayscale noise probability. Default: 0
        clip (optional, bool): Whether to clip image pixel. If `True`, clip image pixel to [0, 1] or [0, 255]. Default: ``True``
        rounds (optional, bool): Noise rounds scale. Default: ``False``

    Returns:
        out (np.ndarray): Add poisson noise to image

    """
    noise = _random_generate_poisson_noise(image, scale_range, gray_prob)
    out = image + noise

    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.0
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.0

    return out


def _generate_gaussian_noise_torch(image, sigma=10.0, gray_noise=0):
    """Generate Gaussian noise (PyTorch)
    Args:
        image (torch.Tensor): Input image
        sigma (float): Noise scale (measured in range 255). Default: 10.0
        gray_noise (optional, torch.Tensor): Whether to add grayscale noise. Default: 0
    Returns:
        gaussian_noise (torch.Tensor): Gaussian noise

    """
    b, _, h, w = image.size()
    if not isinstance(sigma, (float, int)):
        sigma = sigma.view(image.size(0), 1, 1, 1)
    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0
    if cal_gray_noise:
        noise_gray = torch.randn(*image.size()[2:4], dtype=image.dtype, device=image.device) * sigma / 255.0
        noise_gray = noise_gray.view(b, 1, h, w)
    # always calculate color noise
    noise = torch.randn(*image.size(), dtype=image.dtype, device=image.device) * sigma / 255.0
    if cal_gray_noise:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    return noise


def _generate_poisson_noise_torch(image, scale=1.0, gray_noise=0):
    """Generate poisson noise (PyTorch)
    Args:
        image (torch.Tensor): Input image
        scale (optional, float): Noise scale value. Default: 1.0
        gray_noise (optional, torch.Tensor): Whether to add grayscale noise. Default: 0
    Returns:
        poisson_noise (torch.Tensor): Poisson noise
    """
    b, c, h, w = image.size()
    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0
    if cal_gray_noise:
        if c == 3:
            img_gray = rgb_to_grayscale(image, num_output_channels=1)
        elif c == 4:
            tmp = torch.stack([image[:, 0, :, :], torch.mean(image[:, 1:3, :, :], dim=1), image[:, 3, :, :]], dim=1)
            img_gray = rgb_to_grayscale(tmp, num_output_channels=1)
        else:
            raise NotImplementedError("_generate_poisson_noise_torch requires 3 or 4 channel inputs")
        # round and clip image for counting vals correctly
        img_gray = torch.clamp((img_gray * 255.0).round(), 0, 255) / 255.0
        # use for-loop to get the unique values for each sample
        vals_list = [len(torch.unique(img_gray[i, :, :, :])) for i in range(b)]
        vals_list = [2 ** np.ceil(np.log2(vals)) for vals in vals_list]
        vals = img_gray.new_tensor(vals_list).view(b, 1, 1, 1)
        out = torch.poisson(img_gray * vals) / vals
        noise_gray = out - img_gray
        noise_gray = noise_gray.expand(b, c, h, w)
    # always calculate color noise
    # round and clip image for counting vals correctly
    image = torch.clamp((image * 255.0).round(), 0, 255) / 255.0
    # use for-loop to get the unique values for each sample
    vals_list = [len(torch.unique(image[i, :, :, :])) for i in range(b)]
    vals_list = [2 ** np.ceil(np.log2(vals)) for vals in vals_list]
    vals = image.new_tensor(vals_list).view(b, 1, 1, 1)
    out = torch.poisson(image * vals) / vals
    noise = out - image
    if cal_gray_noise:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    if not isinstance(scale, (float, int)):
        scale = scale.view(b, 1, 1, 1)
    poisson_noise = noise * scale
    return poisson_noise


def _random_generate_gaussian_noise_torch(image, sigma_range=(0, 10), gray_prob=0):
    """Random generate gaussian noise (PyTorch)
    Args:
        image (torch.Tensor): Input image
        sigma_range (optional, tuple): Noise range. Default: (0, 10)
        gray_prob (optional, int): Add grayscale noise probability. Default: 0
    Returns:
        gaussian_noise (torch.Tensor): Gaussian noise
    """
    sigma = (
        torch.rand(image.size(0), dtype=image.dtype, device=image.device) * (sigma_range[1] - sigma_range[0])
        + sigma_range[0]
    )
    gray_noise = torch.rand(image.size(0), dtype=image.dtype, device=image.device)
    gray_noise = (gray_noise < gray_prob).float()
    gaussian_noise = _generate_gaussian_noise_torch(image, sigma, gray_noise)
    return gaussian_noise


def _random_generate_poisson_noise_torch(image, scale_range=(0, 1.0), gray_prob=0):
    """Random generate poisson noise (PyTorch)
    Args:
        image (torch.Tensor): Input image
        scale_range (optional, tuple): Noise scale range. Default: (0, 1.0)
        gray_prob (optional, int): Add grayscale noise probability. Default: 0
    Returns:
        poisson_noise (torch.Tensor): Poisson noise
    """
    scale = (
        torch.rand(image.size(0), dtype=image.dtype, device=image.device) * (scale_range[1] - scale_range[0])
        + scale_range[0]
    )
    gray_noise = torch.rand(image.size(0), dtype=image.dtype, device=image.device)
    gray_noise = (gray_noise < gray_prob).float()
    poisson_noise = _generate_poisson_noise_torch(image, scale, gray_noise)
    return poisson_noise


def _add_gaussian_noise_torch(
    image: torch.Tensor, sigma: float = 10.0, clip: bool = True, rounds: bool = False, gray_noise: torch.Tensor = 0
):
    """Add gaussian noise to image (PyTorch)

    Args:
        image (torch.Tensor): Input image
        sigma (optional, float): Noise scale (measured in range 255). Default: 10.0
        clip (optional, bool): Whether to clip image pixel. If `True`, clip image pixel to [0, 1] or [0, 255]. Default: ``True``
        rounds (optional, bool): Gaussian noise rounds scale. Default: ``False``
        gray_noise (optional, torch.Tensor): Whether to add grayscale noise. Default: 0

    Returns:
        out (torch.Tensor): Add gaussian noise to image

    """
    noise = _generate_gaussian_noise_torch(image, sigma, gray_noise)
    out = image + noise

    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.0
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.0

    return out


def _add_poisson_noise_torch(
    image: torch.Tensor, scale: float = 1.0, clip: bool = True, rounds: bool = False, gray_noise: torch.Tensor = 0
) -> torch.Tensor:
    """Add poisson noise to image (PyTorch)

    Args:
        image (torch.Tensor): Input image
        scale (optional, float): Noise scale value. Default: 1.0
        clip (optional, bool): Whether to clip image pixel. If `True`, clip image pixel to [0, 1] or [0, 255]. Default: ``True``
        rounds (optional, bool): Gaussian noise rounds scale. Default: ``False``
        gray_noise (optional, torch.Tensor): Whether to add grayscale noise. Default: 0

    Returns:
        out (torch.Tensor): Add poisson noise to image

    """
    noise = _generate_poisson_noise_torch(image, scale, gray_noise)
    out = image + noise

    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.0
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.0

    return out


def random_add_gaussian_noise_torch(image, sigma_range=(0, 1.0), gray_prob=0):
    """Random add gaussian noise to image (PyTorch)
    Args:
        image (torch.Tensor): Input image
        sigma_range (optional, tuple): Noise range. Default: (0, 1.0)
        gray_prob (optional, int): Add grayscale noise probability. Default: 0
    Returns:
        out (torch.Tensor): Add gaussian noise to image
    """
    out = image + _random_generate_gaussian_noise_torch(image, sigma_range, gray_prob)
    out = torch.clamp(out, 0, 1)
    return out


def random_add_poisson_noise_torch(image, scale_range=(0, 1.0), gray_prob=0):
    """Random add gaussian noise to image (PyTorch)
    Args:
        image (torch.Tensor): Input image
        scale_range (optional, tuple): Noise scale range. Default: (0, 1.0)
        gray_prob (optional, int): Add grayscale noise probability. Default: 0
    Returns:
        out (torch.Tensor): Add poisson noise to image
    """
    out = image + _random_generate_poisson_noise_torch(image, scale_range, gray_prob)
    out = torch.clamp(out, 0, 1)
    return out


def filter2d_torch(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """PyTorch implements `cv2.filter2D()`

    Args:
        image (torch.Tensor): Image data, PyTorch data stream format
        kernel (torch.Tensor): Blur kernel data, PyTorch data stream format

    Returns:
        out (torch.Tensor): Image processed with a specific filter on the image

    """
    k = kernel.size(-1)
    b, c, h, w = image.size()

    if k % 2 == 1:
        image = F.pad(image, (k // 2, k // 2, k // 2, k // 2), mode="reflect")
    else:
        raise ValueError("Wrong kernel size.")

    ph, pw = image.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        image = image.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(image, kernel, padding=0).view(b, c, h, w)
    else:
        image = image.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)

    out = F.conv2d(image, kernel, groups=b * c).view(b, c, h, w)

    return out


def _calculate_quality_factor(quality: int) -> float:
    """Calculate factor corresponding to quality

    Args:
        quality (float): Quality for jpeg compression

    Returns:
        quality_factor (float): Compression factor value

    """
    if quality < 50:
        quality = 5000.0 / quality
    else:
        quality = 200.0 - quality * 2

    quality_factor = quality / 100.0

    return quality_factor


def _add_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    """Add JPEG compression (OpenCV)

    Args:
        image (np.ndarray): Input image
        quality (float): JPEG compression quality

    Returns:
        jpeg_image (np.ndarray): JPEG processed image

    """
    image = np.clip(image, 0, 1)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encode_image = cv2.imencode(".jpg", image * 255.0, encode_params)
    jpeg_image = np.float32(cv2.imdecode(encode_image, 1)) / 255.0

    return jpeg_image


def _random_add_jpg_compression(image: np.ndarray, quality_range: tuple) -> np.ndarray:
    """Random add JPEG compression (OpenCV)

    Args:
        image (np.ndarray): Input image
        quality_range (tuple): JPEG compression quality range

    Returns:
        image (np.ndarray): JPEG processed image

    """
    quality = np.random.uniform(quality_range[0], quality_range[1])
    jpeg_image = _add_jpeg_compression(image, quality)

    return jpeg_image


def _jpeg_diff_round(x: torch.Tensor) -> torch.Tensor:
    """JPEG differentiable

    Args:
        x (torch.Tensor): None.

    Returns:
        jpeg_differentiable (torch.Tensor): None

    """
    jpeg_differentiable = torch.round(x) + (x - torch.round(x)) ** 3

    return jpeg_differentiable


class _RGBToYCbCr(nn.Module):
    def __init__(self) -> None:
        super(_RGBToYCbCr, self).__init__()
        matrix = np.array(
            [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]], dtype=np.float32
        ).T
        self.shift = nn.Parameter(torch.tensor([0.0, 128.0, 128.0]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        out = torch.tensordot(x, self.matrix, dims=1) + self.shift
        out = out.view(x.shape)

        return out


class _ChromaSubsampling(nn.Module):
    def __init__(self) -> None:
        super(_ChromaSubsampling, self).__init__()

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        image = x.permute(0, 3, 1, 2).clone()
        cb = F.avg_pool2d(image[:, 1, :, :].unsqueeze(1), (2, 2), (2, 2), count_include_pad=False)
        cr = F.avg_pool2d(image[:, 2, :, :].unsqueeze(1), (2, 2), (2, 2), count_include_pad=False)
        cb = cb.permute(0, 2, 3, 1)
        cr = cr.permute(0, 2, 3, 1)
        out = x[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)

        return out


class _BlockSplitting(nn.Module):
    def __init__(self) -> None:
        super(_BlockSplitting, self).__init__()
        self.k = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, _ = x.shape[1:3]
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, height // self.k, self.k, -1, self.k)
        x_transposed = x_reshaped.permute(0, 1, 3, 2, 4)
        out = x_transposed.contiguous().view(batch_size, -1, self.k, self.k)

        return out


class _DCT8x8(nn.Module):
    def __init__(self) -> None:
        super(_DCT8x8, self).__init__()
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1.0 / np.sqrt(2)] + [1] * 7)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())
        self.scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha) * 0.25).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - 128
        out = self.scale * torch.tensordot(x, self.tensor, dims=2)
        out = out.view(x.shape)

        return out


class _YQuantize(nn.Module):
    def __init__(self, rounding: torch.round or _jpeg_diff_round) -> None:
        super(_YQuantize, self).__init__()
        self.rounding = rounding
        self.y_table = y_table

    def forward(self, x: torch.Tensor, factor: int or float or torch.Tensor) -> torch.Tensor:
        if isinstance(factor, (int, float)):
            out = x.float() / (self.y_table * factor)
        else:
            b = factor.size(0)
            table = self.y_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
            out = x.float() / table

        out = self.rounding(out)

        return out


class _CQuantize(nn.Module):
    def __init__(self, rounding: torch.round or _jpeg_diff_round) -> None:
        super(_CQuantize, self).__init__()
        self.rounding = rounding
        self.c_table = c_table

    def forward(self, x: torch.Tensor, factor: int or float or torch.Tensor) -> torch.Tensor:
        if isinstance(factor, (int, float)):
            out = x.float() / (self.c_table * factor)
        else:
            b = factor.size(0)
            table = self.c_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
            out = x.float() / table
        out = self.rounding(out)

        return out


class _CompressJPEG(nn.Module):
    def __init__(self, rounding: torch.round or _jpeg_diff_round) -> None:
        super(_CompressJPEG, self).__init__()
        self.l1 = nn.Sequential(_RGBToYCbCr(), _ChromaSubsampling())
        self.l2 = nn.Sequential(_BlockSplitting(), _DCT8x8())
        self.c_quantize = _CQuantize(rounding)
        self.y_quantize = _YQuantize(rounding)

    def forward(self, x: torch.Tensor, factor: torch.Tensor) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        y, cb, cr = self.l1(x * 255)
        components = {"y": y, "cb": cb, "cr": cr}
        for k in components.keys():
            comp = self.l2(components[k])
            if k in ("cb", "cr"):
                comp = self.c_quantize(comp, factor)
            else:
                comp = self.y_quantize(comp, factor)

            components[k] = comp

        out = components["y"], components["cb"], components["cr"]

        return out


class _YDeQuantize(nn.Module):
    def __init__(self) -> None:
        super(_YDeQuantize, self).__init__()
        self.y_table = y_table

    def forward(self, x: torch.Tensor, factor: int or float or torch.Tensor) -> torch.Tensor:
        if isinstance(factor, (int, float)):
            out = x * (self.y_table * factor)
        else:
            b = factor.size(0)
            table = self.y_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
            out = x * table

        return out


class _CDeQuantize(nn.Module):
    def __init__(self) -> None:
        super(_CDeQuantize, self).__init__()
        self.c_table = c_table

    def forward(self, x: torch.Tensor, factor: int or float or torch.Tensor) -> torch.Tensor:
        if isinstance(factor, (int, float)):
            out = x * (self.c_table * factor)
        else:
            b = factor.size(0)
            table = self.c_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
            out = x * table

        return out


class _DeDCT8x8(nn.Module):
    def __init__(self) -> None:
        super(_DeDCT8x8, self).__init__()
        alpha = np.array([1.0 / np.sqrt(2)] + [1] * 7)
        self.alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float())
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos((2 * v + 1) * y * np.pi / 16)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.alpha
        out = 0.25 * torch.tensordot(x, self.tensor, dims=2) + 128
        out.view(x.shape)

        return out


class _DeBlockSplitting(nn.Module):
    def __init__(self) -> None:
        super(_DeBlockSplitting, self).__init__()

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        k = 8
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, height // k, width // k, k, k)
        x_transposed = x_reshaped.permute(0, 1, 3, 2, 4)
        out = x_transposed.contiguous().view(batch_size, height, width)

        return out


class _DeChromaSubsampling(nn.Module):
    def __init__(self) -> None:
        super(_DeChromaSubsampling, self).__init__()

    def forward(self, y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor) -> torch.Tensor:
        def repeat(x, k=2):
            height, width = x.shape[1:3]
            x = x.unsqueeze(-1)
            x = x.repeat(1, 1, k, k)
            x = x.view(-1, height * k, width * k)
            return x

        cb = repeat(cb)
        cr = repeat(cr)

        out = torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], dim=3)

        return out


class _YCbCrToRGB(nn.Module):
    def __init__(self) -> None:
        super(_YCbCrToRGB, self).__init__()
        matrix = np.array([[1.0, 0.0, 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]], dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0, -128.0, -128.0]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.tensordot(x + self.shift, self.matrix, dims=1)
        out = out.view(x.shape).permute(0, 3, 1, 2)

        return out


class _DeCompressJPEG(nn.Module):
    def __init__(self) -> None:
        super(_DeCompressJPEG, self).__init__()
        self.c_de_quantize = _CDeQuantize()
        self.y_de_quantize = _YDeQuantize()
        self.de_dct = _DeDCT8x8()
        self.de_block_splitting = _DeBlockSplitting()
        self.de_chroma_subsampling = _DeChromaSubsampling()
        self.ycbcr_to_rgb = _YCbCrToRGB()

    def forward(
        self, y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor, image_height: int, image_width: int, factor: int
    ) -> torch.Tensor:
        components = {"y": y, "cb": cb, "cr": cr}
        for k in components.keys():
            if k in ("cb", "cr"):
                comp = self.c_de_quantize(components[k], factor)
                height, width = int(image_height / 2), int(image_width / 2)
            else:
                comp = self.y_de_quantize(components[k], factor)
                height, width = image_height, image_width
            comp = self.de_dct(comp)
            components[k] = self.de_block_splitting(comp, height, width)

        out = self.de_chroma_subsampling(components["y"], components["cb"], components["cr"])
        out = self.ycbcr_to_rgb(out)

        out = torch.min(255 * torch.ones_like(out), torch.max(torch.zeros_like(out), out))

        out /= 255

        return out


class DiffJPEG(nn.Module):
    def __init__(self, differentiable: bool) -> None:
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = _jpeg_diff_round
        else:
            rounding = torch.round

        self.compress = _CompressJPEG(rounding)
        self.decompress = _DeCompressJPEG()

    def forward(self, x: torch.Tensor, quality: int or float or torch.Tensor) -> torch.Tensor:
        factor = quality
        if isinstance(factor, (int, float)):
            factor = _calculate_quality_factor(factor)
        else:
            for i in range(factor.size(0)):
                factor[i] = _calculate_quality_factor(factor[i])
        h, w = x.size()[-2:]
        h_pad, w_pad = 0, 0

        if h % 16 != 0:
            h_pad = 16 - h % 16
        if w % 16 != 0:
            w_pad = 16 - w % 16

        x = F.pad(x, (0, w_pad, 0, h_pad), mode="constant", value=0)

        y, cb, cr = self.compress(x, factor)
        out = self.decompress(y, cb, cr, (h + h_pad), (w + w_pad), factor)
        out = out[:, :, 0:h, 0:w]

        return out


def _usm_sharp(image: np.ndarray, weight: float, radius: int, threshold: int) -> np.ndarray:
    if radius % 2 == 0:
        radius += 1

    blur = cv2.GaussianBlur(image, (radius, radius), 0)
    residual = image - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype("float32")
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    out = image + weight * residual
    out = np.clip(out, 0, 1)
    out = soft_mask * out + (1 - soft_mask) * image

    return out


# class USMSharp(nn.Module):

#     def __init__(self, radius=50, sigma=0):
#         super(USMSharp, self).__init__()
#         if radius % 2 == 0:
#             radius += 1

#         self.radius = radius
#         kernel = cv2.getGaussianKernel(radius, sigma)
#         kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
#         self.register_buffer("kernel", kernel)

#     def forward(self, x, weight=0.5, threshold=10):
#         usm_blur = filter2d_torch(x, self.kernel)
#         residual = x - usm_blur

#         mask = torch.abs(residual) * 255 > threshold
#         mask = mask.float()
#         soft_mask = filter2d_torch(mask, self.kernel)
#         out = x + weight * residual
#         out = torch.clip(out, 0, 1)
#         out = soft_mask * out + (1 - soft_mask) * x

#         return out


class USMSharp(nn.Module):
    def __init__(self, radius=50, sigma=2):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1

        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer("kernel", kernel)

    def forward(self, x, weight=2):
        usm_blur = filter2d_torch(x, self.kernel)
        result = x + (x - usm_blur) * weight
        return result


def image_to_tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type

    Returns:
        tensor (torch.Tensor): Data types supported by PyTorch

    Examples:
        >>> example_image = cv2.imread("example_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, False, False)

    """
    # Convert image data type to Tensor data type
    tensor = F_vision.to_tensor(image)

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor


def tensor_to_image(tensor: torch.Tensor, range_norm: bool, half: bool) -> Any:
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type

    Args:
        tensor (torch.Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV

    Examples:
        >>> example_tensor = torch.randn([1,3, 256, 256], dtype=torch.float)
        >>> example_image = tensor_to_image(example_tensor, False, False)

    """
    # Scale the image data from [-1, 1] to [0, 1]
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    return image


def image_resize(
    image: np.ndarray or torch.Tensor, scale_factor: float, antialiasing: bool = True
) -> np.ndarray or torch.Tensor:
    """Implementation of `imresize()` function in Matlab under Python language.

    Args:
        image: The input image.
        scale_factor (float): Scale factor. The same scale applies for both height and width.
        antialiasing (bool): Whether to apply antialiasing when down-sampling operations.
            Caution: Bicubic down-sampling in `PIL` uses antialiasing by default. Default: ``True``.

    Returns:
        out_2 (np.ndarray or torch.Tensor): Output image with shape (c, h, w), [0, 1] range, w/o round.

    """
    squeeze_flag = False
    if type(image).__module__ == np.__name__:  # numpy type
        numpy_type = True
        if image.ndim == 2:
            image = image[:, :, None]
            squeeze_flag = True
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
    else:
        numpy_type = False
        if image.ndim == 2:
            image = image.unsqueeze(0)
            squeeze_flag = True

    in_c, in_h, in_w = image.size()
    out_h, out_w = math.ceil(in_h * scale_factor), math.ceil(in_w * scale_factor)
    kernel_width = 4

    # get weights and indices
    weights_h, indices_h, sym_len_hs, sym_len_he = _calculate_weights_indices(
        in_h, out_h, scale_factor, kernel_width, antialiasing
    )
    weights_w, indices_w, sym_len_ws, sym_len_we = _calculate_weights_indices(
        in_w, out_w, scale_factor, kernel_width, antialiasing
    )
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_c, in_h + sym_len_hs + sym_len_he, in_w)
    img_aug.narrow(1, sym_len_hs, in_h).copy_(image)

    sym_patch = image[:, :sym_len_hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_hs).copy_(sym_patch_inv)

    sym_patch = image[:, -sym_len_he:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_hs + in_h, sym_len_he).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_c, out_h, in_w)
    kernel_width = weights_h.size(1)
    for i in range(out_h):
        idx = int(indices_h[i][0])
        for j in range(in_c):
            out_1[j, i, :] = img_aug[j, idx : idx + kernel_width, :].transpose(0, 1).mv(weights_h[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_c, out_h, in_w + sym_len_ws + sym_len_we)
    out_1_aug.narrow(2, sym_len_ws, in_w).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_we:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_ws + in_w, sym_len_we).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_c, out_h, out_w)
    kernel_width = weights_w.size(1)
    for i in range(out_w):
        idx = int(indices_w[i][0])
        for j in range(in_c):
            out_2[j, :, i] = out_1_aug[j, :, idx : idx + kernel_width].mv(weights_w[i])

    if squeeze_flag:
        out_2 = out_2.squeeze(0)
    if numpy_type:
        out_2 = out_2.numpy()
        if not squeeze_flag:
            out_2 = out_2.transpose(1, 2, 0)

    return out_2


def expand_y(image: np.ndarray) -> np.ndarray:
    """Convert BGR channel to YCbCr format, and expand Y channel data in YCbCr, from HW to HWC

    Args:
        image (np.ndarray): Y channel image data

    Returns:
        y_image (np.ndarray): Y-channel image data in HWC

    """
    # Normalize image data to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Convert BGR to YCbCr, and extract only Y channel
    y_image = bgr2ycbcr(image, True)

    # Expand Y channel
    y_image = y_image[..., None]

    # Normalize the image data to [0, 255]
    y_image = y_image.astype(np.float64) * 255.0

    return y_image


def rgb2ycbcr(image: np.ndarray, only_use_y_channel: bool) -> np.ndarray:
    """Implementation of `rgb2ycbcr()` in Matlab under Python language

    Formula reference: `https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        image (np.ndarray): Image input in RGB format
        only_use_y_channel (bool): Extract Y channel separately

    Returns:
        image (np.ndarray): YCbCr image array data

    """
    if only_use_y_channel:
        image = np.dot(image, [65.481, 128.553, 24.966]) + 16.0
    else:
        image = np.matmul(image, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]) + [
            16,
            128,
            128,
        ]

    image /= 255.0
    image = image.astype(np.float32)

    return image


def bgr2ycbcr(image: np.ndarray, only_use_y_channel: bool) -> np.ndarray:
    """Implementation of `bgr2ycbcr()` in Matlab under Python language

    Formula reference: `https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        image (np.ndarray): Image input in BGR format
        only_use_y_channel (bool): Extract Y channel separately

    Returns:
        image (np.ndarray): YCbCr image array data

    """
    if only_use_y_channel:
        image = np.dot(image, [24.966, 128.553, 65.481]) + 16.0
    else:
        image = np.matmul(image, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [
            16,
            128,
            128,
        ]

    image /= 255.0
    image = image.astype(np.float32)

    return image


def ycbcr2rgb(image: np.ndarray) -> np.ndarray:
    """Implementation of `ycbcr2rgb()` in Matlab under Python language.

    Formula reference: `https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        image (np.ndarray): Image input in YCbCr format

    Returns:
        image (np.ndarray): RGB image array data

    """
    image_dtype = image.dtype
    image *= 255.0

    image = np.matmul(
        image, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071], [0.00625893, -0.00318811, 0]]
    ) * 255.0 + [-222.921, 135.576, -276.836]

    image /= 255.0
    image = image.astype(image_dtype)

    return image


def ycbcr2bgr(image: np.ndarray) -> np.ndarray:
    """Implementation of `ycbcr2bgr()` in Matlab under Python language.

    Formula reference: `https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        image (np.ndarray): Image input in YCbCr format.

    Returns:
        image (np.ndarray): BGR image array data

    """
    image_dtype = image.dtype
    image *= 255.0

    image = np.matmul(
        image, [[0.00456621, 0.00456621, 0.00456621], [0.00791071, -0.00153632, 0], [0, -0.00318811, 0.00625893]]
    ) * 255.0 + [-276.836, 135.576, -222.921]

    image /= 255.0
    image = image.astype(image_dtype)

    return image


def rgb2ycbcr_torch(tensor: torch.Tensor, only_use_y_channel: bool) -> torch.Tensor:
    """Implementation of `rgb2ycbcr()` in Matlab under PyTorch

    Formula reference: `https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        tensor (torch.Tensor): Image data in PyTorch format
        only_use_y_channel (bool): Extract only Y channel

    Returns:
        tensor (torch.Tensor): YCbCr image data in PyTorch format

    """
    if only_use_y_channel:
        weight = torch.Tensor([[65.481], [128.553], [24.966]]).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.Tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(
            tensor
        )
        bias = torch.Tensor([16, 128, 128]).view(1, 3, 1, 1).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    tensor /= 255.0

    return tensor


def bgr2ycbcr_torch(tensor: torch.Tensor, only_use_y_channel: bool) -> torch.Tensor:
    """Implementation of `bgr2ycbcr()` in Matlab under PyTorch

    Formula reference: `https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        tensor (torch.Tensor): Image data in PyTorch format
        only_use_y_channel (bool): Extract only Y channel

    Returns:
        tensor (torch.Tensor): YCbCr image data in PyTorch format

    """
    if only_use_y_channel:
        weight = torch.Tensor([[24.966], [128.553], [65.481]]).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.Tensor([[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]).to(
            tensor
        )
        bias = torch.Tensor([16, 128, 128]).view(1, 3, 1, 1).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    tensor /= 255.0
    return tensor


def center_crop(image: np.ndarray, image_size: int) -> np.ndarray:
    """Crop small image patches from one image center area.

    Args:
        image (np.ndarray): The input image for `OpenCV.imread`.
        image_size (int): The size of the captured image area.

    Returns:
        patch_image (np.ndarray): Small patch image

    """
    image_height, image_width = image.shape[:2]

    # Just need to find the top and left coordinates of the image
    top = max(0, (image_height - image_size) // 2)
    left = max(0, (image_width - image_size) // 2)

    # Crop image patch
    patch_image = image[top : top + image_size, left : left + image_size, ...]

    return patch_image


def center_crop_torch_batch(image: torch.Tensor, image_size: int) -> torch.Tensor:
    image_height, image_width = image.shape[2], image.shape[3]
    top = max(0, (image_height - image_size) // 2)
    left = max(0, (image_width - image_size) // 2)
    patch_image = image[:, :, top : top + image_size, left : left + image_size]
    return patch_image


def random_crop(image: np.ndarray, image_size: int) -> np.ndarray:
    """Crop small image patches from one image.

    Args:
        image (np.ndarray): The input image for `OpenCV.imread`.
        image_size (int): The size of the captured image area.

    Returns:
        patch_image (np.ndarray): Small patch image

    """
    image_height, image_width = image.shape[:2]

    # Just need to find the top and left coordinates of the image
    top = random.randint(0, image_height - image_size)
    left = random.randint(0, image_width - image_size)

    # Crop image patch
    patch_image = image[top : top + image_size, left : left + image_size, ...]

    return patch_image


def random_crop_pytorch_batch(image: np.ndarray, image_size: int) -> np.ndarray:
    image_height, image_width = image.shape[2:]
    top = random.randint(0, image_height - image_size)
    left = random.randint(0, image_width - image_size)
    patch_image = image[:, :, top : top + image_size, left : left + image_size]
    return patch_image


def random_rotate(image, angles, center=None, scale_factor=1.0):
    """Rotate an image by a random angle

    Args:
        image (np.ndarray): Image read with OpenCV
        angles (list): Rotation angle range
        center (optional, tuple[int, int]): High resolution image selection center point. Default: ``None``
        scale_factor (optional, float): scaling factor. Default: 1.0

    Returns:
        rotated_image (np.ndarray): image after rotation

    """
    image_height, image_width = image.shape[:2]

    if center is None:
        center = (image_width // 2, image_height // 2)

    # Random select specific angle
    angle = random.choice(angles)
    matrix = cv2.getRotationMatrix2D(center, angle, scale_factor)
    rotated_image = cv2.warpAffine(image, matrix, (image_width, image_height))

    return rotated_image


def random_horizontally_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Flip the image upside down randomly

    Args:
        image (np.ndarray): Image read with OpenCV
        p (optional, float): Horizontally flip probability. Default: 0.5

    Returns:
        horizontally_flip_image (np.ndarray): image after horizontally flip

    """
    if random.random() < p:
        horizontally_flip_image = cv2.flip(image, 1)
    else:
        horizontally_flip_image = image

    return horizontally_flip_image


def random_vertically_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Flip an image horizontally randomly

    Args:
        image (np.ndarray): Image read with OpenCV
        p (optional, float): Vertically flip probability. Default: 0.5

    Returns:
        vertically_flip_image (np.ndarray): image after vertically flip

    """
    if random.random() < p:
        vertically_flip_image = cv2.flip(image, 0)
    else:
        vertically_flip_image = image

    return vertically_flip_image


def vignetting_torch(batch_img, batch_sigma_ratio, batch_keep_ratio):
    """
    :param batch_img: [bs, c h, w]
    :param radius_sigma_ratio: control how large the sigma is of the gaussian kernel
    :param keep_ratio: keep how many percent of pixels (spherically centered in the image center) to its original brightness
    """
    res = batch_img.clone()
    for curr_b_id, (img, sigma_ratio, keep_ratio) in enumerate(zip(batch_img, batch_sigma_ratio, batch_keep_ratio)):
        _, h, w = img.shape
        kernel_x = cv2.getGaussianKernel(w, w * sigma_ratio)
        kernel_y = cv2.getGaussianKernel(h, w * sigma_ratio)
        kernel = kernel_y * kernel_x.T

        norm_value = kernel.flatten()
        norm_value = norm_value[np.argsort(norm_value)[int(-norm_value.shape[0] * keep_ratio)]]
        mask = kernel / norm_value
        mask = np.clip(mask, 0, 1)
        mask = cv2.GaussianBlur(mask, (h - 1, w - 1), 0)
        mask = torch.FloatTensor(mask).to(img.device)

        res[curr_b_id] = img * mask.unsqueeze(0)
    return res


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.clip(sharpened, 0, 1)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened
