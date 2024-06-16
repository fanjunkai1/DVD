import cv2
import random
import torch
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms
from utils import imwrite


def mod_crop(img, scale):
    """Mod crop images, used during testing.
    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.
    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img


def paired_random_crop(img_cfs, 
                       img_hfs, 
                       cf_patch_resize, 
                       cf_patch_cropsize, 
                       scale, 
                       phase,
                       cf_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.
    It crops lists of lq and gt images with corresponding locations.
    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.
    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_cfs, list):
        img_cfs = [img_cfs]
    if not isinstance(img_hfs, list):
        img_hfs = [img_hfs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_cfs[0]) else 'Numpy'

    # if input_type == 'Tensor':
    #     h_hf, w_hf = img_hfs[0].size()[-2:]
    #     h_cf, w_cf = img_cfs[0].size()[-2:]
    # else:
    #     h_hf, w_hf = img_hfs[0].shape[0:2]
    #     h_cf, w_cf = img_cfs[0].shape[0:2]

    hf_patch_resize = cf_patch_resize // scale
    hf_patch_cropsize = cf_patch_cropsize // scale

    # if h_cf != h_hf * scale or w_cf != w_hf * scale:
    #     raise ValueError(f'Scale mismatches. CF ({h_cf}, {w_cf}) is not {scale}x ',
    #                      f'multiplication of HF ({h_hf}, {w_hf}).')
    
    # if h_hf < hf_patch_resize or w_hf < hf_patch_resize:
    #     raise ValueError(f'LQ ({h_hf}, {w_hf}) is smaller than patch size '
    #                      f'({hf_patch_resize}, {hf_patch_resize}). '
    #                      f'Please remove {cf_path}.')



    if phase == 'train':

        # crop img
        # randomly choose top and left coordinates for hf patch
        top_cf = random.randint(0, cf_patch_resize-cf_patch_cropsize)
        left_cf = random.randint(0, cf_patch_resize-cf_patch_cropsize)

        Resize = transforms.Resize((cf_patch_resize, cf_patch_resize), 
                            interpolation=InterpolationMode.BICUBIC)


        # resize img
        if input_type == 'Tensor':
            img_cfs = [Resize(v) for v in img_cfs]
        else:
            img_cfs = [cv2.resize(v, (cf_patch_resize, cf_patch_resize), interpolation=cv2.INTER_CUBIC) for v in img_cfs]

        if input_type == 'Tensor':
            img_cfs = [v[:, :, top_cf:top_cf + cf_patch_cropsize, left_cf:left_cf + cf_patch_cropsize] 
                    for v in img_cfs]
        else:
            img_cfs = [v[top_cf:top_cf + cf_patch_cropsize, left_cf:left_cf + cf_patch_cropsize, ...] 
                    for v in img_cfs]


        # crop corresponding gt patch
        top_hf, left_hf = int(top_cf // scale), int(left_cf // scale)

        # resize img
        if input_type == 'Tensor':
            img_hfs = [Resize(v) for v in img_hfs]
        else:
            img_hfs = [cv2.resize(v, (hf_patch_resize, hf_patch_resize), interpolation=cv2.INTER_CUBIC) for v in img_hfs]

        
        # crop hf patch
        if input_type == 'Tensor':
            img_hfs = [v[:, :,top_hf:top_hf + hf_patch_cropsize, left_hf:left_hf + hf_patch_cropsize] 
                    for v in img_hfs]
        else:
            img_hfs = [v[top_hf:top_hf + hf_patch_cropsize, left_hf:left_hf + hf_patch_cropsize, ...] 
                    for v in img_hfs]

    else:
        Resize = transforms.Resize((cf_patch_cropsize, cf_patch_cropsize), 
                            interpolation=InterpolationMode.BICUBIC)
        # resize img
        if input_type == 'Tensor':
            img_hfs = [Resize(v) for v in img_hfs]
        else:
            img_hfs = [cv2.resize(v, (hf_patch_cropsize, hf_patch_cropsize)) for v in img_hfs]

        # resize img
        if input_type == 'Tensor':
            img_cfs = [Resize(v) for v in img_cfs]
        else:
            img_cfs = [cv2.resize(v, (cf_patch_cropsize, cf_patch_cropsize)) for v in img_cfs]


    return img_hfs, img_cfs

    

def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).
    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.
    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.
    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.
    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.
    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img