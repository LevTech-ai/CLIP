# Created by Angelo Stekardis 2/2/2023
from os import listdir, walk
from os.path import isfile, join
import csv
import skimage
import random 
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

from collections import OrderedDict
import numpy as np
import torch
from pkg_resources import packaging
import clip

def main():
    num_to_check = 50

    basedir = '/home/angelo/projects/lev_tech/guardian/out/crops/unknown'
    videos_path = '/home/angelo/projects/lev_tech/guardian/data/rucd_samples'
    out_basedir = '/home/angelo/projects/lev_tech/guardian/out/crops/'
    video_to_school = {}
    video_to_school['Camera_7-20221207-112439.mp4'] = 'Vernal_1'
    video_to_school['Camera_9-20221121-124338.mp4'] = 'Price_1'
    video_to_school['Camera_3-20221130-113801-1669833481.mp4'] = 'Monroe'
    video_to_school['V1_Outside_Camera 3-20221208-110342.mp4'] = 'Vernal_1'

    correct_count = 0
    incorrect_count = 0
    unknown_count = 0
    cnt = -1
    for dirpath, dirnames, filenames in walk(basedir):
        for fname in [f for f in filenames if f.endswith(".csv")]:
            cnt += 1
            # Load the video corresponding to this csv
            school_name = dirpath.split('/')[-2]
            video_filename = f'{fname.split(".")[0]}.mp4'

            school_name = video_to_school[video_filename]
            print(join(videos_path, school_name, video_filename))
            cap = cv2.VideoCapture(join(videos_path, school_name, video_filename))

            out_csv = join(out_basedir, fname)

            with open(join(dirpath, fname), mode ='r')as filename:
                # reading the CSV file
                # data = csv.reader(filename)
                
                num_rows = sum(1 for row in filename)
                for i in range(num_to_check):
                    offset = random.randrange(num_rows)
                    filename.seek(offset)
                    filename.readline()
                    line = filename.readline()
                    
                    line = line.split(',')

                    frame_num = int(line[0])
                    cls, x, y, w, h, conf = [float(l) for l in line[1:7]]
                    classification = line[7]
                    class_conf = line[8]

                    # Crop the image from the frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)
                        
                    ret, frame = cap.read()
                    if ret:
                        height, width, chan = frame.shape

                        # Scale x, y, w, h
                        x *= width
                        y *= height
                        w *= width
                        h *= height

                        # crop = frame[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

                        cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (255,0,0), 1)
                        pressed = False
                        print(classification)
                        if classification == 'boy' or classification == 'girl':
                            classification = 'child'
                        else:
                            classification = 'adult'

                        conf = 100
                        # while True:
                        out_basedir
                        cv2.imwrite(f'{join(out_basedir, classification)}/{i + cnt * num_to_check}.png', frame)
                        # cv2.imshow(classification, frame)
                        # cv2.waitKey(0)
                        # cv2.destroyWindow(classification)
                        # print()
                        # true_class = input('a for adult, c for child, e for unknown: ')
                        # if true_class == 'a' or true_class == 'A':
                        #     true_class = 'adult'
                        # elif true_class == 'c' or true_class == 'c':
                        #     true_class = 'child'
                        # elif true_class == 'e' or true_class == 'E':
                        #     true_class = 'unknown'
                        
                        #     # break
                        #     # if k == ord('a') or k == ord('A'):
                        #     #     true_class = 'adult'
                        #     #     break
                        #     # elif k == ord('c') or k == ord('C'):
                        #     #     true_class = 'child'
                        #     #     break
                        #     # elif k == ord('e') or k == ord('E'):
                        #     #     true_class = 'unknown'
                        #     #     break
                        # print(true_class)
                        # if true_class == 'unknown':
                        #     unknown_count += 1
                        # elif true_class == classification:
                        #     correct_count += 1
                        # else:
                        #     incorrect_count +=1


    print(f'Percetage correct: {100 * correct_count / (correct_count + incorrect_count)}')
    print(f'Correct: {correct_count}, incorrect: {incorrect_count}, unknown: {unknown_count}')
                

if __name__ == "__main__":
    main()