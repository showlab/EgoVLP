import os
import cv2
import shutil
import json
import sys
from multiprocessing import Pool
import subprocess
import pandas as pd
from csv import reader, writer

video_dir = 'dataset/ego4d_256/'
output_dir = 'dataset/ego4d_chunked/'

save_dir = os.path.join(output_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

dur_limit = 600

def video2segments(infos):
    global count
    index, uid, dur = infos[0], infos[1], infos[2]
    input_path = os.path.join(video_dir, uid + '.mp4')

    output_uid_dir = os.path.join(output_dir, uid)
    if not os.path.exists(output_uid_dir):
        os.makedirs(output_uid_dir)

    if index % num_partition != partition:
        return

    assert os.path.exists(input_path)

    cap = cv2.VideoCapture(input_path)
    rate = cap.get(5)
    frame_num = cap.get(7)
    duration = frame_num / rate

    if duration <= dur_limit:
        shutil.copyfile(input_path, os.path.join(output_uid_dir, '0.mp4'))
        return

    num_seg = duration // dur_limit

    s1_time = 0;
    s2_time = dur_limit
    num_finished = 0
    while num_finished <= num_seg:
        output_mp4_path = os.path.join(output_uid_dir, str(num_finished) + '.mp4')

        cmd = 'ffmpeg -y -i {} -ss {} -to {} -async 1 {}'.format(input_path, s1_time, s2_time,
                                                                                 output_mp4_path)
        # print(cmd)
        subprocess.call(cmd, shell=True)

        # Update for next steps
        s1_time = s2_time
        s2_time = min(s1_time + dur_limit, duration)
        num_finished += 1
    return

if __name__ == "__main__":
    with open('dataset/manifest.csv', 'r') as csv:
        csv_reader = list(reader(csv))[1:]

    downloaded = os.listdir('dataset/ego4d')

    uid_list = []
    infos_list = []
    num_valid = 0
    for i_item, item in enumerate(csv_reader):
        uid, dur = item[0], float(item[2])
        existed = uid + '.mp4' in downloaded

        if not existed:
            continue

        uid_list.append(uid)
        infos_list.append([num_valid, uid, dur])
        num_valid += 1

    # for infos in infos_list:
    #     video2segments(infos)

    pool = Pool(32)
    pool.map(video2segments, tuple(infos_list))