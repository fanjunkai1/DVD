import torch
import numpy as np
from PIL import Image
import cv2
import glob
import os
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
from network.dehaze_net import ResnetGenerator
from network.dehaze_net import DCPDehazeGenerator
from torchvision.transforms import InterpolationMode


def __patch_instance_norm_state_dict(state_dict, module, keys, i=0):
    """
    Fix InstanceNorm checkpoints incompatibility (prior to 0.4)
    
    """
    key = keys[i]
    if i + 1 == len(keys): # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
            (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
    else:
        __patch_instance_norm_state_dict(state_dict, 
                                         getattr(module, key), keys, i + 1)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', 
                        default='input_img', 
                        help='input test image folder')
    
    parser.add_argument('--output', 
                        type=str, 
                        default='output_img', 
                        help='output folder')
    
    parser.add_argument('--model_path',
                        type=str,
                        default=  
                        'models/remove_hazy_model.pth')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ResnetGenerator(input_nc=3, 
                            output_nc=3, 
                            norm_layer=nn.InstanceNorm2d)

    state_dict = (torch.load(args.model_path, map_location=device))

    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    # patch InstanceNorm checkpoints prior to 0.4
    for key in list(state_dict.keys()):  
        __patch_instance_norm_state_dict(state_dict, model, key.split('.'))

    model.load_state_dict(state_dict)
    model.eval()

    DCP = DCPDehazeGenerator().to(device)

    model = model.to(device)

    os.makedirs(args.output, exist_ok=True)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)

        # read image
        img = Image.open(path).convert('RGB')

        transform_list = [
            transforms.ToTensor(),
            transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))
        ]

        transform = transforms.Compose(transform_list)

        img = transform(img)
        img = img.unsqueeze(0).to(device)

        # print(img.shape)

        # inference
        try:
            with torch.no_grad():
                img = DCP(img)
                output = model(img)
                output = (output + 1) / 2
        except Exception as error:
            print('Error', error, imgname)
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.output, f'{imgname}_dehazing.png'), output)


if __name__ == "__main__":
    main()

