import argparse
import cv2
import os


############################## video ----> frame ##############################

def hazy_video2hazy_frame(args, video_path, save_frame_path):

    # load video
    video = cv2.VideoCapture(video_path)
    # output_save_path = './video_dataset/hazy_video2frame/2_hazy_video/'

    # Define video frame rate and frames per second (FPS) for saving.
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))

    save_rate = args.save_rate # Saved frame rate

    # Define the file naming format for the output frames
    output_filename_format = 'frame_{}_{}.jpg'

    # Initialize counter
    frame_count = 0
    save_count = 0

    # traverse through video frame by frame
    while True:
        # read next frame
        ret, frame = video.read()

        # If the video has finished loading, then exit the loop
        if not ret:
            break

        if frame_count % (frame_rate * save_rate) == 0:
            # define the filename of output, and save frame
            output_filename = output_filename_format.format(save_count, args.data_type)
            cv2.imwrite(os.path.join(save_frame_path, output_filename), frame)
            save_count += 1

        # Update frame counter
        frame_count += 1

    # Release video object
    video.release()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video_rootpath", default="./videoes/train_video/hazy_video/", 
        help="folder with input hazy_video"
    )

    parser.add_argument(
        "--save_frame_rootpath",
        default="./datasets/foggy_video/train_video/hazyVideoFrames/",
        help="folder for output image frames",
    )

    parser.add_argument(
        "--save_rate",
        default=0.1,
        help="save frame rate",
    )

    parser.add_argument(
        "--data_type",
        default= 'hazy',
        help="data type of video, | hazy | or | clear |",
    )

    args = parser.parse_args()

    # video_rootpath = './video_dataset/hazy_video/'
    video_filename_list = os.listdir(args.video_rootpath)

    # print(video_filename_list)

    # save_frame_rootpath = './video_dataset/hazy_video2frame/'

    for per_video_filename in video_filename_list:

        video_id = per_video_filename.split('_')[0]

        new_frame_filename = '{}_{}_frames'.format(video_id, args.data_type)

        save_frame_path = os.path.join(args.save_frame_rootpath, new_frame_filename)

        if not os.path.exists(save_frame_path):
            os.makedirs(save_frame_path)

        per_video_path = os.path.join(args.video_rootpath, per_video_filename)

        # print(per_video_path, save_frame_path)
        print('processing %s' % (per_video_path))

        hazy_video2hazy_frame(args, per_video_path, save_frame_path)


    