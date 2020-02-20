import os
import sys
import cv2
import glob
import numpy as np
import threading
import shutil
from tqdm i
mport tqdm_notebook,tnrange

def touch_dir(dir):
    if not os.path.exists(dir):
        print('making dir : ', dir)
        os.makedirs(dir)
        
def clean_and_touch_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
	
def work_on_imgs(num_from, num_to):
    for k in tnrange(num_from, num_to, desc='img {:d} to {:d}'.format(num_from, num_to)):
        folder_num = int(k / 2000)
        num_str = '{:05d}'.format(k)

        img_path = os.path.join(img_dir, str(k) + '.jpg')
        image = cv2.imread(img_path)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)

        mask = np.zeros((512, 512), dtype='uint8')
        for idx, label in enumerate(label_list):
            filename = os.path.join(mask_dir, str(folder_num), num_str + '_' + label + '.png')
            if (os.path.exists(filename)):
                im=cv2.imread(filename)
                im = im[:, :, 0]
                for i in range(512):
                    for j in range(512):
                        if im[i][j] != 0:
                            mask[i][j] = (idx + 1)

        total_mask_path = os.path.join(total_mask_dir, str(k) + '.png')
        cv2.imwrite(total_mask_path, mask)

        one_mask = np.uint8(mask>0)
        inverse_mask = 1-one_mask

        bg = np.ones((512, 512, 3), dtype='uint8') * 255

        masked = cv2.bitwise_and(image, image, mask=one_mask)
        bg_masked = cv2.bitwise_and(bg, bg, mask = inverse_mask)
        composed = cv2.add(masked, bg_masked)

        target_path = os.path.join(target_dir, 'masked_' + num_str + '.png')
        cv2.imwrite(target_path, composed)
		
import math

label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

data_root = os.path.join('C:/Users/Administrator/Desktop/deeplab/', 'CelebAMask-HQ') # modify your dataset path here
example_num = 30000
img_dir = os.path.join(data_root, 'CelebA-HQ-img')
mask_dir = os.path.join(data_root, 'CelebAMask-HQ-mask-anno')
total_mask_dir = os.path.join(data_root, 'CelebAMask-total-mask')
target_dir = os.path.join(data_root, 'CelebAMask-target')

clean_and_touch_dir(total_mask_dir)
clean_and_touch_dir(target_dir)

parallel_num = 20
batch_num = math.ceil(example_num / parallel_num)
threadings = []

for i in range(parallel_num):
    start = batch_num * i
    end = min(batch_num*(i+1), example_num)
    print("pending batch examples from {} to {}".format(start, end))
    t = threading.Thread(target=work_on_imgs, args=(start,end, ))
    threadings.append(t)
    t.start()
	
for t in threadings:
    t.join(