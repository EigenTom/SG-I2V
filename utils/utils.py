import importlib
import numpy as np
import cv2
import torch
import torch.distributed as dist
import torch.fft as fft
import math


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def check_istarget(name, para_list):
    """ 
    name: full name of source para
    para_list: partial name of target para 
    """
    istarget=False
    for para in para_list:
        if para in name:
            return True
    return istarget


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_npz_from_dir(data_dir):
    data = [np.load(os.path.join(data_dir, data_name))['arr_0'] for data_name in os.listdir(data_dir)]
    data = np.concatenate(data, axis=0)
    return data


def load_npz_from_paths(data_paths):
    data = [np.load(data_path)['arr_0'] for data_path in data_paths]
    data = np.concatenate(data, axis=0)
    return data   


def resize_numpy_image(image, max_resolution=512 * 512, resize_short_edge=None):
    h, w = image.shape[:2]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def setup_dist(args):
    if dist.is_initialized():
        return
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )
    
    
# >>> custom functions
def freq_mix_3d(x, noise, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))

    # frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = noise_freq * HPF
    x_freq_mixed = x_freq_low + noise_freq_high # mix in freq domain

    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real

    return x_mixed
    
def get_freq_filter(shape, device, filter_type, n, d_s, d_t):
    """
    Form the frequency filter for noise reinitialization.

    Args:
        shape: shape of latent (B, C, T, H, W)
        filter_type: type of the freq filter
        n: (only for butterworth) order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    if filter_type == "gaussian":
        return gaussian_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "ideal":
        return ideal_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "box":
        return box_low_pass_filter(shape=shape, d_s=d_s, d_t=d_t).to(device)
    elif filter_type == "butterworth":
        return butterworth_low_pass_filter(shape=shape, n=n, d_s=d_s, d_t=d_t).to(device)
    else:
        raise NotImplementedError
    
def gaussian_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the gaussian low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = math.exp(-1/(2*d_s**2) * d_square)
    return mask


def butterworth_low_pass_filter(shape, n=4, d_s=0.25, d_t=0.25):
    """
    Compute the butterworth low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        n: order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] = 1 / (1 + (d_square / d_s**2)**n)
    return mask


def ideal_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask
    for t in range(T):
        for h in range(H):
            for w in range(W):
                d_square = (((d_s/d_t)*(2*t/T-1))**2 + (2*h/H-1)**2 + (2*w/W-1)**2)
                mask[..., t,h,w] =  1 if d_square <= d_s*2 else 0
    return mask


def box_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Compute the ideal low pass filter mask (approximated version).

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    mask = torch.zeros(shape)
    if d_s==0 or d_t==0:
        return mask

    threshold_s = round(int(H // 2) * d_s)
    threshold_t = round(T // 2 * d_t)

    cframe, crow, ccol = T // 2, H // 2, W //2
    mask[..., cframe - threshold_t:cframe + threshold_t, crow - threshold_s:crow + threshold_s, ccol - threshold_s:ccol + threshold_s] = 1.0

    return mask