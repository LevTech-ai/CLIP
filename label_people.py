# Created by Angelo Stekardis 1/19/2023
from os import listdir, walk
from os.path import isfile, join
import csv
import skimage
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
    # Load the model
    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    # Specify the text to be used
    classes = ['boy', 'girl', 'man', 'woman']
    text_descriptions = [f"This is a photo of a {label}" for label in classes]
    text_tokens = clip.tokenize(text_descriptions).cuda()

    # @todo @astekardis get these paths throught the command line
    basedir = '/home/angelo/projects/lev_tech/guardian/out/labels'
    videos_path = '/home/angelo/projects/lev_tech/guardian/data/rucd_samples'
    out_basedir = '/home/angelo/projects/lev_tech/guardian/out/crops/unknown'
    for dirpath, dirnames, filenames in walk(basedir):
        for fname in [f for f in filenames if f.endswith(".csv")]:
            # Load the video corresponding to this csv
            school_name = dirpath.split('/')[-1]
            video_filename = f'{fname.split(".")[0]}.mp4'

            print(join(videos_path, school_name, video_filename))
            cap = cv2.VideoCapture(join(videos_path, school_name, video_filename))

            out_csv = join(out_basedir, fname)

            with open(join(dirpath, fname), mode ='r')as filename:
                # reading the CSV file
                data = csv.reader(filename)
                
                # prev_frame_num = -1
                # bboxes = np.array([])
                images = []
                original_images = []
                original_lines = []
                prev_frame = -1
                do_inference = False
                cnt = 0
                for line in data:
                    if int(line[1]) == 0: # Person class
                        frame_num = int(line[0])
                        if prev_frame != frame_num:
                            do_inference = True
                            images = []

                        original_lines.append(line[2:7])
                        x, y, w, h, conf = [float(l) for l in line[2:7]]

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

                            crop = frame[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
                            # print(crop.shape)
                            # cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (255,0,0), 1)

                            
                            converted = cv2.cvtColor(crop,cv2.COLOR_BGR2RGB) # @todo @astekardis is this needed?
                            pil_im = Image.fromarray(crop) # @ Getting slightly different performance between this and the notebook - why??
                            images.append(preprocess(pil_im.convert("RGB")))
                            original_images.append(crop)

                            if do_inference:
                                image_input = torch.tensor(np.stack(images)).cuda()
                                with torch.no_grad():
                                    image_features = model.encode_image(image_input).float()
                                    text_features = model.encode_text(text_tokens).float()
                                    text_features /= text_features.norm(dim=-1, keepdim=True)

                                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                                top_probs, top_labels = text_probs.cpu().topk(min(5, len(classes)), dim=-1)

                                # Write all results for this frame line by line
                                for i, img in enumerate(original_images):
                                    class_name = classes[top_labels[i].numpy()[0]]
                                    class_conf = top_probs[i].numpy()[0]
                                    line = (str(frame_num), '0', *original_lines[i], class_name, str(class_conf))

                                    with open(out_csv, 'a') as f:
                                        f.write(('%s,' * len(line)).rstrip() % line + '\n')

                                    # print([classes[index] for index in top_labels[i].numpy()])
                                    # print([prob for prob in top_probs[i].numpy()])
                                    # open_cv_image = np.array(img) 
                                    # Convert RGB to BGR 
                                    # open_cv_image = open_cv_image[:, :, ::-1].copy() 
                                    # cv2.imshow(classes[top_labels[i].numpy()[0]], img)
                                    # cv2.waitKey()
                                    # cv2.imwrite(f'/home/angelo/projects/lev_tech/guardian/out/crops/unknown/out_frame_{cnt}_obj_{i}_{classes[top_labels[i].numpy()[0]]}.png', img)

                                images = []
                                original_images = []
                                original_lines = []

                                # cv2.imshow("frame", frame)
                                # cv2.imshow("cropped", crop)
                                # cv2.waitKey()
                            
                            prev_frame = frame_num
                            # cnt+=1
                            # if cnt > 5:
                            #     break

    return

if __name__ == "__main__":
    main()