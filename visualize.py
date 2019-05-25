import torch
from torch import nn
from torch.nn import functional as F
import math


def gaussian_kernel(sigma=3):
    kernel_size = 5 * sigma

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return gaussian_kernel


def fill_mask(mask, m, k):
    q = 0
    for i in range(m[0]):
        for j in range(m[1]):
            u = k // 2
            mask[q, :, u + (i * k) - 1:u + (i * k) + 2, u + (j * k) - 1:u + (j * k) + 2] = 1
            q += 1
    return mask


def generate_mask(size, k, sigma, downsample=1, contrast=1):
    m = (size[0] // k, size[1] // k)
    mask = torch.zeros((m[0] * m[1], 1, size[0] // downsample, size[1] // downsample), dtype=torch.long)
    mask = fill_mask(mask, m, k // downsample)
    mask = torch.tensor(mask, dtype=torch.float)
    kernel = gaussian_kernel(sigma=sigma // downsample)[None, None, :, :]
    convolved = F.conv2d(mask, kernel, groups=1, padding=(kernel.shape[-1] - 1) // 2)

    mask = F.interpolate(convolved, size, mode='bilinear')


    # TODO: solve NAN problem (why is max zero?)
    m, _ = mask.view(mask.size(0), -1).max(1)
    m = m[:,None,None,None]
    m = torch.where(m>0, m, torch.tensor(1.0))
    mask = mask / m

    if contrast > 1:
        mask = torch.clamp(contrast * mask, 0, 1)
    return mask


def blur_image(img, sigma, downsample):
    original_shape = img.shape[2:]
    channels = img.shape[1]
    size = (img.shape[2] // downsample, img.shape[3] // downsample)
    img = F.interpolate(img, size, mode='nearest')
    kernel = gaussian_kernel(sigma=sigma // downsample)[None, None, :, :].repeat(channels, 1, 1, 1)
    img = F.conv2d(img, kernel, groups=channels, padding=(kernel.shape[-1] - 1) // 2)
    img = F.interpolate(img, original_shape, mode='bilinear')
    return img


def generate_candidate_frames(img, spacing = 16, sigma = 15, blur_sigma = 9, contrast = 1.2, downsample = 8):

    masks = generate_mask(img.shape[2:], spacing, sigma, downsample=downsample, contrast=contrast)

    blurred_img = blur_image(img.float(), blur_sigma, downsample)

    outputs = (masks) * blurred_img + (1 - masks) * img.float()

    return outputs, masks


def saliency_map(state, policy, output_shape, spacing=8, sigma=9, blur_sigma=15, contrast=1.2, downsample=8):

    state = state.float()

    candidates, masks = generate_candidate_frames(state, spacing=spacing, sigma=sigma, blur_sigma=blur_sigma, contrast=contrast, downsample=downsample)

    frames = torch.cat([state, candidates], 0)

    values = policy.value(frames).sum(1)

    L, Ls = values[:1], values[1:]

    m = int(math.sqrt(Ls.shape[0]))
    Ls = Ls.view(m,m)

    Sign = (Ls > L.squeeze()).float()
    S = 0.5 * (Ls - L.squeeze())**2

    S = F.interpolate(S[None,None,:,:], output_shape, mode='nearest')
    Sign = F.interpolate(Sign[None, None, :, :], output_shape, mode='nearest')


    S = S - S.min()

    #smax = S.max()
    S = S / S.max()

    #S = blur_image(S, 3, 1)
    #S = S * smax

    return S, Sign






