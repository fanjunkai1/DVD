from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
import argparse
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
    


def Match_frames(hazy_frames_folder, clear_frames_folder):

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

    hazy_frames_list = os.listdir(hazy_frames_folder)
    clear_frames_list = os.listdir(clear_frames_folder)

    match_frames_record = {}

    for hazy_frames in hazy_frames_list:

        # current image for match frames
        curr_hazy_frame_path = os.path.join(hazy_frames_folder, hazy_frames)
        curr_hazy_frame = Image.open(curr_hazy_frame_path).convert('RGB')
        
        # preprocess data by using the define transform
        curr_hazy_frame = transform(curr_hazy_frame)

        match_score_dict = {}

        ## Match the current hazy image with the clearest image that is most similar
        for clear_frames in clear_frames_list:

            clear_frames_path = os.path.join(clear_frames_folder, clear_frames)
            curr_clear_frame = Image.open(clear_frames_path).convert('RGB')

            curr_clear_frame = transform(curr_clear_frame)
            
            # compute similarity score
            similarity_score = compute_cos_simi(curr_hazy_frame, curr_clear_frame)

            match_score_dict[clear_frames] = similarity_score

            sorted_match_score = sorted(match_score_dict.items(), 
                                        key=lambda x : x[1],
                                        reverse=True)

        match_frames_record[curr_hazy_frame_path] = os.path.join( 
                                                clear_frames_folder, 
                                                sorted_match_score[0][0])
        # print(match_frames_record)
    return match_frames_record



def Write_TXT(file_content, save_path):

    with open(save_path, 'w+') as file:
        for hazy_frame, clear_frame in file_content.items():
            file.write(hazy_frame + '|' + clear_frame + "\n")
    file.close()




# def sliding_window_frames():



def main(hazy_frames_path, clear_frames_path, save_matched_txt):

    hazy_frames_folder_list = os.listdir(hazy_frames_path)
    # clear_frames_folder_list = os.listdir(clear_frames_path)

    for hazy_frames_folder in hazy_frames_folder_list:

        index = hazy_frames_folder.split('_')[0]
        clear_frames_folder = '{}_clear_frames'.format(index)

        hazy_frames_folder_path = os.path.join(hazy_frames_path, hazy_frames_folder)
        clear_frames_folder_path = os.path.join(clear_frames_path, clear_frames_folder)

        match_reuslts = Match_frames(hazy_frames_folder_path, clear_frames_folder_path)

        Write_TXT(match_reuslts, os.path.join(save_matched_txt, 
                                            '{}_hazy&clear_frames.txt'.format(index)))




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hazy_frames_path", 
        default="./datasets/foggy_video/test_sliding_window/hazyVideoFrames/", 
        help="the path of hazy video frames"
    )

    parser.add_argument(
        "--clear_frames_path", 
        default="./datasets/foggy_video/test_sliding_window/clearVideoFrames/", 
        help="the path of clear video frames"
    )

    parser.add_argument(
        "--save_matched_txt_path", 
        default="./datasets/foggy_video/test_sliding_window/test_MatchFrames/", 
        help="the save path of mached TXT record"
    )

    args = parser.parse_args()

    main(args.hazy_frames_path, 
         args.clear_frames_path, 
         args.save_matched_txt_path)


