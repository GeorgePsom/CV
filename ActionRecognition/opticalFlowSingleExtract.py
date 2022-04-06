import keras.regularizers
import wget
import unzip
import cv2
import numpy as np
from helpfuncs import*
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
import sklearn.model_selection as sk_ms
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Add, Input, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu
from PIL import Image
from sklearn import utils
from moviepy.editor import *

set_1_indices = [[2,14,15,16,18,19,20,21,24,25,26,27,28,32,40,41,42,43,44,45,46,47,48,49,50],
                 [1,6,7,8,9,10,11,12,13,23,24,25,27,28,29,30,31,32,33,34,35,44,45,47,48],
                 [2,3,4,11,12,15,16,17,18,20,21,27,29,30,31,32,33,34,35,36,42,44,46,49,50],
                 [1,7,8,9,10,11,12,13,14,16,17,18,22,23,24,26,29,31,35,36,38,39,40,41,42]]
set_2_indices = [[1,3,4,5,6,7,8,9,10,11,12,13,17,22,23,29,30,31,33,34,35,36,37,38,39],
                 [2,3,4,5,14,15,16,17,18,19,20,21,22,26,36,37,38,39,40,41,42,43,46,49,50],
                 [1,5,6,7,8,9,10,13,14,19,22,23,24,25,26,28,37,38,39,40,41,43,45,47,48],
                 [2,3,4,5,6,15,19,20,21,25,27,28,30,32,33,34,37,43,44,45,46,47,48,49,50]]
classes = ['handShake', 'highFive', 'hug', 'kiss']  # we ignore the negative class

# test set
set_1 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_1_indices[c]]
set_1_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_1_indices[c]]
print(f'Set 1 to be used for test ({len(set_1)}):\n\t{set_1}')
print(f'Set 1 labels ({len(set_1_label)}):\n\t{set_1_label}\n')

# training set
set_2 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_2_indices[c]]
set_2_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_2_indices[c]]
print(f'Set 2 to be used for train and validation ({len(set_2)}):\n\t{set_2}')
print(f'Set 2 labels ({len(set_2_label)}):\n\t{set_2_label}')

#clip=VideoFileClip(f'TV-HI/tv_human_interactions_videos/{set_2[video_no]}')
#print(f'\n\nA video with the label - {set_2_label[video_no]}\n')
#clip.ipython_display(width=380)

mode = "train"
video_no = 26  # change this to a number between [0, 100] and you can see a different training video from Set 2

cap = cv2.VideoCapture(f'TV-HI/tv_human_interactions_videos/{set_2[video_no]}')

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

step_frame = int((frames)/16)

counter = 1
step_count = 1;

while(1):

    if counter==frames-2:
        break;

    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    if counter % step_frame == 1 or step_frame == 1:
        line = 'TV-HI/'+mode+'/'+ set_2[video_no][0:-4] + '-' +  str(step_count) +  '.png'
        cv2.imwrite('TV-HI/'+mode+'/'+ set_2[video_no][0:-4] + '-' +  str(step_count) +  '.png', rgb)
        step_count += 1

    counter += 1
    prvs = next

cap.release()
cv2.destroyAllWindows()