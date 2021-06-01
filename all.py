import argparse
import os

def get_list_32_opt():
    video_list = open(VIDEO_LIST_DIR).readlines()
    video_list = [item.strip() for item in video_list]
    video_len = len(video_list)
    for i, video in enumerate(video_list):
        #print(video)
        os.system('python get_32_opt.py --VIDEO_DIR ' + VIDEO_DIR + ' --VIDEO_NAME ' + video.split(' ')[0] +
          ' --OUTPUT_DIR ' + OUTPUT_DIR )
        progress = round(i*100/len(video_list), 2)
        print(str(progress)+"%...")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--VIDEO_DIR', dest='VIDEO_DIR', type=str, default='./temp')
    parser.add_argument('--VIDEO_LIST_DIR', dest='VIDEO_LIST_DIR', type=str, default='./temp')
    parser.add_argument('--OUTPUT_DIR', dest='OUTPUT_DIR', type=str, default='./temp')
    args = parser.parse_args()
    params = vars(args)

    VIDEO_DIR = params['VIDEO_DIR']
    VIDEO_LIST_DIR = params['VIDEO_LIST_DIR']
    OUTPUT_DIR = params['OUTPUT_DIR']
    get_list_32_opt()
    
    