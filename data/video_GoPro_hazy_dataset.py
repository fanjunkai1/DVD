from pathlib import Path
import os, random
import cv2
import numpy as np
import torch
from torch.utils import data as data
from utils.registry import DATASET_REGISTRY
from .transforms import paired_random_crop, augment
from utils import get_root_logger, imfrombytes, img2tensor, tensor2img
from utils import imwrite
import torch.nn as nn
from pre_dehazing.network.dehaze_net import ResnetGenerator, DCPDehazeGenerator





@DATASET_REGISTRY.register()
class VideoGoProHazyDataset(data.Dataset):

    """
    Vimeo90K dataset for training.
    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt
    Each line contains the following items, separated by a white space.
    1. clip name;
    2. frame number;
    3. image shape
    Examples:
    ::
        00001/0001 7 (256,448,3)
        00001/0002 7 (256,448,3)
    - Key examples: "00001/0001"
    - CF (cf): clear frames;
    - HF (hf): hazy frames, e.g., low-resolution/blurry/noisy/compressed frames.
    The neighboring frame list for different num_frame:
    ::
        num_frame | frame list
                1 | 4
                3 | 3,4,5
                5 | 2,3,4,5,6
                7 | 1,2,3,4,5,6,7
    Args:
        opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        num_frame (int): Window size for input frames.
        gt_size (int): Cropped patched size for gt patches.
        random_reverse (bool): Random reverse input frames.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h 
                        and w for implementation).
        scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(VideoGoProHazyDataset, self).__init__()
        self.opt = opt

        self.cf_folder = Path(opt['dataroot_cf'])
        self.hf_folder = Path(opt['dataroot_hf'])

        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line for line in fin]

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ###################### one stage -----> predehazing ####################
        self.dehaze_model = ResnetGenerator(input_nc=3, 
                        output_nc=3, 
                        norm_layer=nn.InstanceNorm2d)
        self.DCP = DCPDehazeGenerator()
        state_dict = (torch.load(self.opt['predehazing_model_path']))
        self.dehaze_model.load_state_dict(state_dict)
        self.dehaze_model.eval()
        # self.model = self.model.to(self.device)
        #######################################################################

        # ####################### optical flow estimation #######################
        # self.flow_model = torch.nn.DataParallel(RAFT())
        # # print(self.opt['raft_model_path'])
        # self.flow_model.load_state_dict(torch.load(self.opt['raft_model_path']))
        # self.flow_model = self.flow_model.module
        # self.flow_model.eval()
        

        
    def __getitem__(self, index):
        
        cf_resize = self.opt['cf_resize']
        cf_cropsize = self.opt['cf_cropsize']
        scale = self.opt['scale']
        key = self.keys[index]


        clip, seq = key.split('/')
        seq = seq.rstrip('\n')


        # get the hazy frame (e.g., frame_1_hazy.jpg)
        hazyframe_path = self.hf_folder/clip/seq

        hazyframe_list = sorted(os.listdir(hazyframe_path),key=lambda x:int(x.split('_')[1]))
        # print(hazyframe_list)
        curr_frame_path = os.path.join(hazyframe_path, hazyframe_list[-1])

        img_hfs = []
        img_cfs = []

        for per_hazyframe_name in hazyframe_list:
            per_hazyframe_path = os.path.join(hazyframe_path, per_hazyframe_name)
            input_hazyframe_img = cv2.imread(per_hazyframe_path).astype(np.float32) / 255.0
            img_hfs.append(input_hazyframe_img)

        # obtain the path of clearframe
        clear_clip = clip.split('_')[0] + '_clearframe'
        clearframe_path = self.cf_folder/clear_clip/seq
        clearframe_name_list = sorted(os.listdir(clearframe_path))

        clearframe_path_list = []
        for per_clearframe_name in clearframe_name_list:
            per_clearframe_path = os.path.join(clearframe_path, per_clearframe_name)
            clearframe_path_list.append(per_clearframe_path)
            ref_clearframe_img = cv2.imread(per_clearframe_path).astype(np.float32) / 255.0
            img_cfs.append(ref_clearframe_img)

        phase = self.opt['phase']
        # randomly resize and crop
        img_hfs, img_cfs = paired_random_crop(img_cfs, img_hfs, cf_resize, cf_cropsize, 
                                            scale, phase, clearframe_path_list)

        # augmentation - flip, rotate
        img_hfs.extend(img_cfs)
        img_results = augment(img_hfs, self.opt['use_hflip'], self.opt['use_rot'])

        # imwrite(img_results[0]*255.0, './results/img1.jpg')
        # imwrite(img_results[1]*255.0, './results/img2.jpg')
        # imwrite(img_results[2]*255.0, './results/img3.jpg')
        # imwrite(img_results[3]*255.0, './results/img4.jpg')

        img_results = img2tensor(img_results)
        img_hfs = torch.stack(img_results[0:-2], dim=0)


        ################ one stage -----> predehazing ##############
        t, c, h, w = img_hfs.shape

        predehazing_hfs = []
        for i in range(0, t):
            try:
                with torch.no_grad():
                    img_frame = img_hfs[i, :, :, :].unsqueeze(dim=0)
                    img_frame = self.DCP(img_frame)
                    output = self.dehaze_model(img_frame)
                    output = (output + 1) / 2
                    # imwrite(tensor2img(output), './results/img3.jpg')
                    predehazing_hfs.append(output.squeeze(dim=0))
            except Exception as error:
                print('Error', error, key)

        img_predehazing_hfs = torch.stack(predehazing_hfs, dim=0)
        # print(img_predehazing_hfs.shape)
        # imwrite(tensor2img(img_predehazing_hfs[0,:, :, :].unsqueeze(dim=0)), './results/img1.jpg')
        # imwrite(tensor2img(img_predehazing_hfs[1,:, :, :].unsqueeze(dim=0)), './results/img2.jpg')

        ############################################################

        # ################ optical flow estimation ###################
        # # current frame
        # optical_flow = []
        # curr_frame = img_predehazing_hfs[t-1,:, :, :].unsqueeze(dim=0)
        
        # for i in range(0, t):
        #     try:
        #         with torch.no_grad():
        #             img_frame = img_predehazing_hfs[i, :, :, :].unsqueeze(dim=0)
        #             padder = InputPadder(curr_frame.shape)
        #             curr_frame = curr_frame * 255.0
        #             img_frame = img_frame * 255.0
        #             # print(curr_frame, img_frame)
        #             curr_frame, img_frame = padder.pad(curr_frame, img_frame)
        #             flow_low, flow_up = self.flow_model(curr_frame, 
        #                                                 img_frame, 
        #                                                 iters=20, 
        #                                                 test_mode=True)
        #             # print(flow_up.shape)
        #             optical_flow.append(flow_up)
        #             # 
        #             # viz(curr_frame, flow_up)
            
        #     except Exception as error:
        #         print('Error', error, key)

        # optical_flow = torch.stack(optical_flow, dim=1).squeeze(dim=0)
        # # print(optical_flow.shape)
        

        img_ref_matched_frame = img_results[-2]
        img_ref_next_frame = img_results[-1]

        # print(img_ref_matched_frame.shape)

        # img_lqs: (t, c, h, w)
        # img_ref_matched_frame: (c, h, w)
        # img_ref_next_frame: (c, h, w)
        # key: str

        # print(img_hfs.shape)
        # print(curr_frame_path)

        return {
                'hfs': img_hfs,
                'predehazing_hfs': img_predehazing_hfs, 
                'cf_ref_curr': img_ref_matched_frame, 
                'cf_ref_next': img_ref_next_frame, 
                # 'optical_flow': optical_flow,
                'curr_frame_path': curr_frame_path,
                'key': key.rstrip('\n')}



    def __len__(self):
        return len(self.keys)
    
