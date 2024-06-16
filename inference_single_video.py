import argparse
import cv2
import glob
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from archs.DVD_arch import NSDNGAN
from data.data_util import read_img_seq
from utils.img_util import tensor2img, img2tensor,imwrite
from pre_dehazing.network.dehaze_net import ResnetGenerator
from pre_dehazing.network.dehaze_net import DCPDehazeGenerator

from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--visual_enhance_model_path', 
        type=str, 
        default='checkpoint/net_g_latest.pth')
    
    parser.add_argument(
        '--input_path', 
        type=str, 
        default='input_video_frame', 
        help='input test image folder')
    
    parser.add_argument(
        '--dehazing_model_path', 
        type=str,
        default= 'pre_dehazing/models/remove_hazy_model_256x256.pth')
    
    parser.add_argument(
        '--save_path', 
        type=str, 
        default='output_video_frame', 
        help='save image path')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # predehazing model
    dehazing_model = ResnetGenerator(input_nc=3, output_nc=3, norm_layer=nn.InstanceNorm2d)
    state_dict = (torch.load(args.dehazing_model_path, map_location=device))
    dehazing_model.load_state_dict(state_dict)
    dehazing_model.eval()
    DCP = DCPDehazeGenerator().to(device)
    dehazing_model = dehazing_model.to(device)

    # video model
    visual_enhance_model = NSDNGAN(num_feat=64)
    visual_enhance_model.load_state_dict(torch.load(args.visual_enhance_model_path)['params'], strict=True)
    visual_enhance_model.eval()
    visual_enhance_model = visual_enhance_model.to(device)

    os.makedirs(args.save_path, exist_ok=True)

    if os.path.isfile(args.input_path):
        imgs_list = [args.input_path]
    else:
        imgs_list = sorted(glob.glob(os.path.join(args.input_path, '*')))

    pbar = tqdm(total=len(imgs_list), unit='image')

    for idx in range(0, len(imgs_list)):
        if idx == 0:
            img_paths = [imgs_list[0], imgs_list[0]]
        else:
            img_paths = imgs_list[idx - 1:idx + 1]

        Imgs_list = []
        for img in img_paths:
            img = cv2.imread(img).astype(np.float32) 
            img = cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC) / 255.
            img = img2tensor(img, bgr2rgb=True, float32=True).unsqueeze(dim=0).to(device)

            with torch.no_grad():
                dcp_img = DCP(img)
                dehazing_img = dehazing_model(dcp_img)
                dehazing_img = (dehazing_img + 1) / 2
            Imgs_list.append(dehazing_img)
        Imgs = torch.concat(Imgs_list, dim=0)

        frame_name = os.path.splitext(os.path.split(img_paths[-1])[-1])[0]
        Imgs = Imgs.unsqueeze(0).to(device)
        output, _, _ , _, _, = visual_enhance_model(Imgs)
        output = tensor2img(output)
        cv2.imwrite(os.path.join(args.save_path, '{}_DVD.png'.format(frame_name)), output)

        pbar.update(1)
    pbar.close()



if __name__ == '__main__':
    main()