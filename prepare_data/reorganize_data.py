import argparse
import glob, os
import shutil


def main(args):

    match_results_path = glob.glob(os.path.join(args.index_txt_path, '*.txt'))

    for per_txt_path in match_results_path:

        video_index = int(per_txt_path.split('/')[-1].split('_')[0])
        # print(video_index)
        hazy_video_seq_name = f"{video_index:05d}"
        hazy_video_seq_path = os.path.join(args.save_hazyframe_seq_path, 
                        '{}_hazyframe_seq'.format(hazy_video_seq_name))
        
        if not os.path.exists(hazy_video_seq_path):
            os.makedirs(hazy_video_seq_path)
        
        corr_clearframe_path = os.path.join(args.save_clearframe_path, 
                        '{}_clearframe'.format(hazy_video_seq_name))
        
        if not os.path.exists(corr_clearframe_path):
            os.makedirs(corr_clearframe_path)

        with open(per_txt_path, 'r') as f:
            for line in f.read().splitlines():
                # about hazyframe
                hazyframe_path, clearframe_path = line.split('|')
                hazyframe_folder = '/'.join(hazyframe_path.split('/')[:-1])
                hazy_filename = hazyframe_path.split('/')[-1]
                currframe_index = int(hazy_filename.split('_')[1])     
                hazy_domain_name = hazy_filename.split('.')[1]

                # about clearframe
                clearframe_folder = '/'.join(clearframe_path.split('/')[:-1])
                clearframe_len = len(os.listdir(clearframe_folder))
                clear_filename = clearframe_path.split('/')[-1]
                matchedframe_index = int(clear_filename.split('_')[1]) 
                clear_domain_name = clear_filename.split('.')[1]

                inputframe_path_list = []
                if currframe_index+1 >= args.input_frames_num and matchedframe_index<clearframe_len-2:

                    for i in range(0, args.input_frames_num):

                        inputframe_path = os.path.join(hazyframe_folder, 
                                    'frame_{}_hazy.{}'.format(currframe_index-i, hazy_domain_name))
                        
                        inputframe_path_list.append(inputframe_path)

                    hazy_clips_name = f"{currframe_index:05d}"
                    hazy_video_clips_path = os.path.join(hazy_video_seq_path, hazy_clips_name)

                    if not os.path.exists(hazy_video_clips_path):
                        os.makedirs(hazy_video_clips_path)

                    clear_video_clips_path = os.path.join(corr_clearframe_path, hazy_clips_name)

                    if not os.path.exists(clear_video_clips_path):
                        os.makedirs(clear_video_clips_path)

                    for per_frame_path in inputframe_path_list:
                        shutil.copy(per_frame_path, hazy_video_clips_path)

                    next_clearframe_path = os.path.join(clearframe_folder, 
                                    'frame_{}_clear.{}'.format(matchedframe_index+1, clear_domain_name))
                    
                    shutil.copy(clearframe_path, clear_video_clips_path)
                    shutil.copy(next_clearframe_path, clear_video_clips_path)

        print("prepare %d data done!" % (video_index))
                
        f.close()
        
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_frames_num", 
        default = 2, 
        help="the number of input frames."
    )

    parser.add_argument(
        "--index_txt_path", 
        default = './datasets/foggy_video/train_video/TrainMatchFrames/', 
        help="reorganize data structure by using matched frames"
    )

    parser.add_argument(
        "--save_hazyframe_seq_path", 
        default = './datasets/foggy_video/train_video/TrainClipsFrames/hazyframe_seq', 
        help="save path for reorganized data"
    )

    parser.add_argument(
        "--save_clearframe_path", 
        default = './datasets/foggy_video/train_video/TrainClipsFrames/clearframe', 
        help="save path for reorganized data"
    )

    args = parser.parse_args()

    main(args)