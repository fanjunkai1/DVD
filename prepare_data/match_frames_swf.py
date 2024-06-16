from PIL import Image
import numpy as np
import os
import re
import torch
import torch.nn as nn
import argparse
import math
from torchvision.models import vgg16
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode
# from torch.nn.functional import cosine_similarity


class Compute_cos_similarity(nn.Module):
    def __init__(self, vgg_model):
        super(Compute_cos_similarity, self).__init__()

        self.vgg_layers = vgg_model
        self.select_layers = {
        '0': "conv_1_1",
        '28': "conv_5_3"
    }
        self.weight_list = [0.2, 0.8]

    def cosine_similarity(self, img1, img2, dim=1, eps=1e-8):
        # cos_sim = np.dot(img1, img2) / (np.linalg.norm(img1) * np.linalg.norm(img2))
        w12 = torch.sum(img1 * img2, dim)
        w1 = torch.norm(img1, 2, dim)
        w2 = torch.norm(img2, 2, dim)
        return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
    
    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():

            x = module(x)
            if name in self.select_layers:
                output[self.select_layers[name]] = x

        return list(output.values())

    def forward(self, img1, img2):

        cos_sim_list = []

        img1_feature_list = self.output_features(img1)
        img2_feature_list = self.output_features(img2)

        for weight, img1_fea, img2_fea in zip(self.weight_list, img1_feature_list, 
                                              img2_feature_list):
            
            img1_flatten = img1_fea.reshape(1, -1)
            img2_flatten = img2_fea.reshape(1, -1)

            similarity_scores = self.cosine_similarity(img1_flatten, img2_flatten)

            cos_sim_list.append(similarity_scores * weight)
            
        return sum(cos_sim_list) / len(cos_sim_list)
    


def OneFrameToMultiFramesMatch(MatchFrameImg_path, SWF_FramesImg_path_list):

    ## load pretrain vgg16 model for feature extraction
    model = vgg16(pretrained=True).features[:31]

    ## init Compute_cos_similarity class
    compute_cos_simi = Compute_cos_similarity(model)


    ## define preprocess data operation
    transform_list = [
        transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
    transform = transforms.Compose(transform_list)

    # hazy_frames_list = os.listdir(hazy_frames_folder)
    # clear_frames_list = os.listdir(clear_frames_folder)

    # match_frames_record = {}

    # current image for match frames
    MatchFrameImg = Image.open(MatchFrameImg_path).convert('RGB')
    
    # preprocess data by using the define transform
    MatchFrameImg = transform(MatchFrameImg)

    match_score_dict = {}
    sorted_match_score = {}

    ## Match the current hazy image with the clearest image that is most similar
    for MatchingFrameImg_path in SWF_FramesImg_path_list:

        MatchingFrameImg = Image.open(MatchingFrameImg_path).convert('RGB')

        MatchingFrameImg = transform(MatchingFrameImg)
        
        # compute similarity score
        similarity_score = compute_cos_simi(MatchFrameImg, MatchingFrameImg)

        match_score_dict[MatchingFrameImg_path] = similarity_score

        sorted_match_score = sorted(match_score_dict.items(), 
                                    key=lambda x : x[1],
                                    reverse=True)

    MatchedFrameImg_path = sorted_match_score[0][0]
    MatchedIndex = sorted_match_score[0][0].split('/')[-1].split('_')[1]

    # print(match_frames_record)
    return MatchedFrameImg_path, int(MatchedIndex)



def Write_TXT(file_content, save_path):

    with open(save_path, 'w+') as file:
        for hazy_frame, clear_frame in file_content.items():
            file.write(hazy_frame + '|' + clear_frame + "\n")
    file.close()



def Adaptive_sliding_window_matchframes(hazy_frames_folder_path, clear_frames_folder_path):

    HazyFramesImg_filename_list = os.listdir(hazy_frames_folder_path)
    ClearFramesImg_filename_list = os.listdir(clear_frames_folder_path)


    ## sorted for list, 0 frame ---> N frame
    HazyFramesImg_filename_list = sorted(HazyFramesImg_filename_list, 
                                key=lambda x : int(re.search(r'\d+', x).group()))
    
    ClearFramesImg_filename_list = sorted(ClearFramesImg_filename_list, 
                                key=lambda x : int(re.search(r'\d+', x).group()))

    ## computing the length of video frames 
    hazy_frames_length = len(HazyFramesImg_filename_list)
    clear_frames_length = len(ClearFramesImg_filename_list)

    MatchedFramesRecord = {}
    MatchedIndex_list = []

    ## Generating sliding window frame sequences from long video frames
    # if hazy_frames_length < clear_frames_length:

        # set sliding_window_size to 2N
        # N ----> the length difference between a hazy video and a clear video

    #     init_sliding_window_size = clear_frames_length - hazy_frames_length

    #     if init_sliding_window_size <= 2: 

    #         init_sliding_window_size = 4 * (init_sliding_window_size + 1) 

    #     StartIndex = 0
    #     EndIndex = StartIndex + init_sliding_window_size

    #     OneQuarter_sw_size = math.ceil(init_sliding_window_size / 2)
        

    #     for HazyFramesImg_filename in HazyFramesImg_filename_list:

    #         # index prior
    #         hazy_img_index = HazyFramesImg_filename.split('_')[1]

    #         print(HazyFramesImg_filename)

    #         MatchFrameImg_path = os.path.join(hazy_frames_folder_path, 
    #                                              HazyFramesImg_filename)

    #         SWF_FramesImg_filename_list = ClearFramesImg_filename_list[StartIndex:EndIndex]

    #         print(SWF_FramesImg_filename_list)

    #         SWF_FramesImg_path_list = []

    #         for PerFrameImg_filename in SWF_FramesImg_filename_list:
    #             PerFrameImg_path = os.path.join(clear_frames_folder_path, PerFrameImg_filename)
    #             SWF_FramesImg_path_list.append(PerFrameImg_path)

    #         MatchedFrameImg_path, MatchedIndex = OneFrameToMultiFramesMatch(
    #                                                             MatchFrameImg_path,
    #                                                             SWF_FramesImg_path_list)
    #         MatchedIndex_list.append(MatchedIndex)
    #         print(MatchedIndex)


    #         if len(MatchedIndex_list) < 2:
    #             EstiNextMatchedIndex = MatchedIndex_list[-1] + hazy_img_index
    #             next_sliding_window_size = init_sliding_window_size
    #         else:
    #             EstiNextMatchedIndex = MatchedIndex + np.abs(MatchedIndex_list[-1] - MatchedIndex_list[-2])
    #             if np.abs(MatchedIndex_list[-1] - MatchedIndex_list[-2]) == 0:
    #                 next_sliding_window_size = init_sliding_window_size
    #             else:
    #                 next_sliding_window_size = 2 * np.abs(MatchedIndex_list[-1] - MatchedIndex_list[-2]) + 1

    #         if EstiNextMatchedIndex - OneQuarter_sw_size < 0:
    #             StartIndex = 0
    #         else:
    #             StartIndex = EstiNextMatchedIndex - OneQuarter_sw_size

    #             if StartIndex < MatchedIndex_list[-1]:
    #                 StartIndex = MatchedIndex_list[-1] + 1

    #         EndIndex = StartIndex + next_sliding_window_size

    #         if EndIndex > clear_frames_length:
    #             StartIndex = clear_frames_length - next_sliding_window_size
    #             EndIndex = clear_frames_length

    #         MatchedFramesRecord[MatchFrameImg_path] = MatchedFrameImg_path

    #         OneQuarter_sw_size = math.ceil(next_sliding_window_size / 2)
    # else:
    #     # set sliding_window_size to N
    #     init_sliding_window_size = math.ceil((hazy_frames_length - clear_frames_length) / 2)

    #     if sliding_window_size <= 1:
    #         sliding_window_size = 6 * (init_sliding_window_size + 1)

    #     StartIndex = 0
    #     EndIndex = StartIndex + init_sliding_window_size

    #     for HazyFramesImg_filename in HazyFramesImg_filename_list:

    #         print(HazyFramesImg_filename)

    #         MatchFrameImg_path = os.path.join(hazy_frames_folder_path, 
    #                                     HazyFramesImg_filename)
            
            
    #         SWF_FramesImg_filename_list = ClearFramesImg_filename_list[StartIndex:EndIndex]

    #         print(SWF_FramesImg_filename_list)

    #         SWF_FramesImg_path_list = []

    #         for PerFrameImg_filename in SWF_FramesImg_filename_list:
    #             PerFrameImg_path = os.path.join(clear_frames_folder_path, PerFrameImg_filename)
    #             SWF_FramesImg_path_list.append(PerFrameImg_path)

    #         MatchedFrameImg_path, MatchedIndex = OneFrameToMultiFramesMatch(
    #                                                 MatchFrameImg_path,
    #                                                 SWF_FramesImg_path_list)
    #         MatchedIndex_list.append(MatchedIndex)
    #         print(MatchedIndex)

    #         # OneQuarter_sw_size = math.ceil(sliding_window_size / 2)

    #         if len(MatchedIndex_list) < 2:
    #             EstiNextMatchedIndex = 2 * MatchedIndex + 1 
    #         else:
    #             EstiNextMatchedIndex = MatchedIndex + np.abs(MatchedIndex_list[-1]-MatchedIndex_list[-2])

    #         if EstiNextMatchedIndex - OneQuarter_sw_size < 0:
    #             StartIndex = 0
    #         else:
    #             StartIndex = EstiNextMatchedIndex - OneQuarter_sw_size

    #             if StartIndex < MatchedIndex_list[-1]:
    #                 StartIndex = MatchedIndex_list[-1] - 2
            
    #         next_sliding_window_size = 2 * np.abs(MatchedIndex_list[-1] - MatchedIndex_list[-2])

    #         EndIndex = StartIndex + next_sliding_window_size

    #         if EndIndex > clear_frames_length:
    #             StartIndex = clear_frames_length - next_sliding_window_size
    #             EndIndex = clear_frames_length

    #         MatchedFramesRecord[MatchFrameImg_path] = MatchedFrameImg_path
    #         OneQuarter_sw_size = math.ceil(next_sliding_window_size / 2)
    # Generating sliding window frame sequences from long video frames
    if hazy_frames_length < clear_frames_length:

        # set sliding_window_size to 2N
        # N ----> the length difference between a hazy video and a clear video

        min_SWF_size = math.ceil((clear_frames_length - hazy_frames_length) / 2)

        if min_SWF_size <= 2: 

            min_SWF_size = 2 * min_SWF_size + 1

        StartIndex = 0
        EndIndex = StartIndex + min_SWF_size
        # OneQuarter_sw_size = math.ceil(min_SWF_size / 2)

        for HazyFramesImg_filename in HazyFramesImg_filename_list:
            # index proir of video frames
            Hazy_Img_index = int(HazyFramesImg_filename.split('_')[1])
            print(HazyFramesImg_filename)

            MatchFrameImg_path = os.path.join(hazy_frames_folder_path, 
                                                 HazyFramesImg_filename)

            SWF_FramesImg_filename_list = ClearFramesImg_filename_list[StartIndex:EndIndex]

            print(SWF_FramesImg_filename_list)

            SWF_FramesImg_path_list = []

            for PerFrameImg_filename in SWF_FramesImg_filename_list:
                PerFrameImg_path = os.path.join(clear_frames_folder_path, PerFrameImg_filename)
                SWF_FramesImg_path_list.append(PerFrameImg_path)

            MatchedFrameImg_path, MatchedIndex = OneFrameToMultiFramesMatch(
                                                                MatchFrameImg_path,
                                                                SWF_FramesImg_path_list)
            MatchedIndex_list.append(MatchedIndex)
            print(MatchedIndex)

            if len(MatchedIndex_list) <= 1:
                if MatchedIndex_list[-1] > 0:
                    EstiNextMatchedIndex = 2 * MatchedIndex_list[-1]

                    if EstiNextMatchedIndex >= Hazy_Img_index:
                        EndIndex = EstiNextMatchedIndex + (MatchedIndex_list[-1] - 0)
                        StartIndex = Hazy_Img_index - (MatchedIndex_list[-1] - 0)
                        if StartIndex < MatchedIndex_list[-1]:
                            StartIndex = MatchedIndex_list[-1]
                    else:
                        EndIndex = Hazy_Img_index + (MatchedIndex_list[-1] - 0)
                        StartIndex = EstiNextMatchedIndex - (MatchedIndex_list[-1] - 0)
                        if StartIndex < MatchedIndex_list[-1]:
                            StartIndex = MatchedIndex_list[-1]
                else:
                    StartIndex = 0
                    EndIndex = min_SWF_size
            
            else:
                Est_SW_size = np.abs(MatchedIndex_list[-1] - MatchedIndex_list[-2])
                if Est_SW_size == 0:
                    Est_SW_size = min_SWF_size

                # half_SW_size = math.ceil(Est_SW_size / 2)
                EstiNextMatchedIndex = MatchedIndex_list[-1] + Est_SW_size
                if EstiNextMatchedIndex >= Hazy_Img_index:
                    EndIndex = EstiNextMatchedIndex + Est_SW_size
                    StartIndex = Hazy_Img_index - Est_SW_size
                    if StartIndex < MatchedIndex_list[-1]:
                        StartIndex = MatchedIndex_list[-1]
                    elif Est_SW_size == 0:
                        StartIndex = MatchedIndex_list[-1] + 1
                    
                else:
                    EndIndex = Hazy_Img_index + Est_SW_size
                    StartIndex = EstiNextMatchedIndex - Est_SW_size
                    if StartIndex < MatchedIndex_list[-1]:
                        StartIndex = MatchedIndex_list[-1]
                    elif Est_SW_size == 0:
                        StartIndex = MatchedIndex_list[-1] + 1

            if EndIndex > clear_frames_length:
                EndIndex = clear_frames_length

            if (EndIndex - StartIndex) < min_SWF_size:
                EndIndex = StartIndex + min_SWF_size

            MatchedFramesRecord[MatchFrameImg_path] = MatchedFrameImg_path
    
    else:
        # set sliding_window_size to 2N
        # N ----> the length difference between a hazy video and a clear video

        min_SWF_size = math.ceil((hazy_frames_length - clear_frames_length) / 4)

        if min_SWF_size <= 2: 

            min_SWF_size = 2 * min_SWF_size + 1

        StartIndex = 0
        EndIndex = StartIndex + min_SWF_size
        # OneQuarter_sw_size = math.ceil(min_SWF_size / 2)

        for HazyFramesImg_filename in HazyFramesImg_filename_list:
            # index proir of video frames
            Hazy_Img_index = int(HazyFramesImg_filename.split('_')[1])
            print(HazyFramesImg_filename)

            MatchFrameImg_path = os.path.join(hazy_frames_folder_path, 
                                                 HazyFramesImg_filename)

            SWF_FramesImg_filename_list = ClearFramesImg_filename_list[StartIndex:EndIndex]

            print(SWF_FramesImg_filename_list)

            SWF_FramesImg_path_list = []

            for PerFrameImg_filename in SWF_FramesImg_filename_list:
                PerFrameImg_path = os.path.join(clear_frames_folder_path, PerFrameImg_filename)
                SWF_FramesImg_path_list.append(PerFrameImg_path)

            MatchedFrameImg_path, MatchedIndex = OneFrameToMultiFramesMatch(
                                                                MatchFrameImg_path,
                                                                SWF_FramesImg_path_list)
            MatchedIndex_list.append(MatchedIndex)
            print(MatchedIndex)

            if len(MatchedIndex_list) <= 1:
                if MatchedIndex_list[-1] > 0:
                    EstiNextMatchedIndex = 2 * MatchedIndex_list[-1] 

                    if EstiNextMatchedIndex >= Hazy_Img_index:
                        EndIndex = EstiNextMatchedIndex + (MatchedIndex_list[-1] - 0)
                        StartIndex = Hazy_Img_index - (MatchedIndex_list[-1] - 0)
                        if StartIndex < MatchedIndex_list[-1]:
                            StartIndex = MatchedIndex_list[-1]
                    else:
                        EndIndex = Hazy_Img_index + (MatchedIndex_list[-1] - 0)
                        StartIndex = EstiNextMatchedIndex - (MatchedIndex_list[-1] - 0)
                        if StartIndex < MatchedIndex_list[-1]:
                            StartIndex = MatchedIndex_list[-1]
                else:
                    StartIndex = 0
                    EndIndex = min_SWF_size
            else:
                Est_SW_size = np.abs(MatchedIndex_list[-1] - MatchedIndex_list[-2])
                if Est_SW_size == 0:
                    Est_SW_size = min_SWF_size
                half_SW_size = math.ceil(Est_SW_size / 2)
                EstiNextMatchedIndex = MatchedIndex_list[-1] + half_SW_size
                if EstiNextMatchedIndex >= Hazy_Img_index:
                    EndIndex = EstiNextMatchedIndex + half_SW_size
                    StartIndex = Hazy_Img_index - half_SW_size
                    if StartIndex < MatchedIndex_list[-1]:
                        StartIndex = MatchedIndex_list[-1]
                    # elif StartIndex < MatchedIndex_list[-1] and Est_SW_size == 0:
                    #     StartIndex = MatchedIndex_list[-1] + 1
                    
                else:
                    EndIndex = Hazy_Img_index + half_SW_size
                    StartIndex = EstiNextMatchedIndex - half_SW_size
                    if StartIndex < MatchedIndex_list[-1]:
                        StartIndex = MatchedIndex_list[-1]

            if EndIndex > clear_frames_length:
                EndIndex = clear_frames_length

            if (EndIndex - StartIndex) < min_SWF_size:
                EndIndex = StartIndex + min_SWF_size
                
            MatchedFramesRecord[MatchFrameImg_path] = MatchedFrameImg_path

    return MatchedFramesRecord



def main(hazy_frames_path, clear_frames_path, save_matched_txt_path):

    hazy_frames_folder_list = os.listdir(hazy_frames_path)
    # clear_frames_folder_list = os.listdir(clear_frames_path)

    for hazy_frames_folder in hazy_frames_folder_list:

        index = hazy_frames_folder.split('_')[0]
        clear_frames_folder = '{}_clear_frames'.format(index)

        hazy_frames_folder_path = os.path.join(hazy_frames_path, hazy_frames_folder)
        clear_frames_folder_path = os.path.join(clear_frames_path, clear_frames_folder)

        match_reuslts = Adaptive_sliding_window_matchframes(hazy_frames_folder_path,
                                                            clear_frames_folder_path)

        Write_TXT(match_reuslts, os.path.join(save_matched_txt_path, 
                                            '{}_hazy&clear_frames.txt'.format(index)))
        
def main_test_single_video(hazy_frames_path, clear_frames_path, save_matched_txt_path):


    index = hazy_frames_path.split('/')[-2].split('_')[0]
    match_reuslts = Adaptive_sliding_window_matchframes(hazy_frames_path,
                                                        clear_frames_path)

    Write_TXT(match_reuslts, os.path.join(save_matched_txt_path, 
                                        '{}_hazy&clear_frames.txt'.format(index)))
        

# def _concat_all_matchedTXT(save_matched_txt_path, save_concat_all_machedTXT_path):

#     matched_resultTXT_list = os.listdir(save_matched_txt_path)
#     save_TXTfile_path = os.path.join(save_concat_all_machedTXT_path, 
#                                  'meta_info_GoproHazy_val_all_frames.txt')

#     f = open(save_TXTfile_path, 'w+')

#     for per_video_matchedTXT in matched_resultTXT_list:

#         per_video_matchedTXT_path = os.path.join(save_matched_txt_path, 
#                                             per_video_matchedTXT)
        
#         for line in open(per_video_matchedTXT_path):
#             f.writelines(line)
    
#     f.close()




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hazy_frames_path", 
        default="./datasets/foggy_video/train_video/HazyVideoFrames/23_hazy_frames/", 
        help="the path of hazy video frames"
    )

    parser.add_argument(
        "--clear_frames_path", 
        default="./datasets/foggy_video/train_video/ClearVideoFrames/23_clear_frames/", 
        help="the path of clear video frames"
    )

    parser.add_argument(
        "--save_matched_txt_path", 
        default="./datasets/foggy_video/train_video/TrainMatchFrames/", 
        help="the save path of mached TXT record"
    )

    # parser.add_argument(
    #     "--save_concat_all_machedTXT_path", 
    #     default="./data/meta_info/", 
    #     help="the save path of mached TXT record"
    # )

    args = parser.parse_args()

    main_test_single_video(args.hazy_frames_path, 
         args.clear_frames_path, 
         args.save_matched_txt_path)
    
    # _concat_all_matchedTXT( args.save_matched_txt_path,
    #                         args.save_concat_all_machedTXT_path)


