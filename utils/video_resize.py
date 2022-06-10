import os
import time
import sys
import subprocess
from multiprocessing import Pool, Value

folder_path = './dataset/ego4d/'
output_path = './dataset/ego4d_256/'

def videos_resize(videoinfos):
    global count

    videoid, videoname = videoinfos

    if os.path.exists(os.path.join(output_path, videoname)):
        print(f'{videoname} is resized.')
        return

    inname = folder_path + '/' + videoname
    outname = output_path + '/' + videoname

    cmd = "ffmpeg -y -i {} -filter:v scale=\"trunc(oh*a/2)*2:256\" -c:a copy {}".format( inname, outname)
    subprocess.call(cmd, shell=True)

    return

if __name__ == "__main__":
    file_list = []
    mp4_list = [item for item in os.listdir(folder_path) if item.endswith('.mp4')]
    for id, video in enumerate(mp4_list):
        file_list.append([id, video])

    for file in file_list:
        videos_resize(file)