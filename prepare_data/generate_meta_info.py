import argparse, os
from os import path as osp

def generate_meta_info_gopro(args):

    folder_list = sorted(os.listdir(args.hazyframe_path))

    with open(args.save_meta_info_path, 'w+') as f:
        for folder in folder_list:

            hazyframe_seq = sorted(os.listdir(osp.join(args.hazyframe_path, folder)))

            for per_seq in hazyframe_seq:
                info = f"{folder}/{per_seq}"
                f.write(f'{info}\n')
    f.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hazyframe_path", 
        default = './datasets/foggy_video/train_video/TrainClipsFrames/hazyframe_seq/', 
        help="the path of input hazy frame."
    )

    parser.add_argument(
        "--clearframe_path", 
        default = './datasets/foggy_video/train_video/TrainClipsFrames/clearframe/', 
        help="the path of clear frame."
    )

    parser.add_argument(
        "--save_meta_info_path", 
        default = './data/meta_info/meta_info_GoPro_train_frames_seq.txt', 
        help="the path of clear frame."
    )

    args = parser.parse_args()

    generate_meta_info_gopro(args)