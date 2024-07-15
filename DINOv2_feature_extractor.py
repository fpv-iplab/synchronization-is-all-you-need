import os
import glob
video_path = os.listdir('./videos')
print(video_path)

import lmdb
import cv2
from PIL import Image
import torchvision.transforms as T
import torch
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print('device:',device)
import sys
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14') #1024
dinov2_vitl14.to(device)

sys.path.append("path_to.../DINOv2/dinov2")

env = lmdb.open('lmdb_feature_path.../rgb', map_size = 1099511627776)

with env.begin(write = True) as txn:
    for index, video in enumerate(video_path):
        print(index,'/',len(video_path))
        print('Start',video)
        cap = cv2.VideoCapture('./videos/' + video)
        print('video:', video)
        if not cap.isOpened():
            print("Video error")
            exit()
        
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            transform = T.Compose([ T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5]), ])
            img = transform(img)[:3].unsqueeze(0)
            img = img.to(device)
            
            number = i+1
            number_format = str(number).zfill(10)
            key = video.split('.')[0] + '_' + number_format + '.jpg'

            if txn.get(key.encode('utf-8')) is not None:
                print("Key ", key, "already in lmdb.")
                i+=1
                continue

            with torch.no_grad(): 
                features = dinov2_vitl14(img)

            features = features.cpu()
            features = features.numpy()
            txn.put(key.encode(), features.tobytes())

            if i%10000 == 0:
                print(i+1)
                print('example key:', key)

            i+=1
        cap.release()
        print('End',video)
