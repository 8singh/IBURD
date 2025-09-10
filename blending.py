import scipy.sparse
from scipy.sparse.linalg import spsolve
import cv2
import numpy as np
import pdb
import torch
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
from skimage.io import imsave
from utils import *
import json
import random
import os
import math
import argparse
from pycocotools.coco import COCO

#Input arguments--------------------------------------------------------------------------
parser = argparse.ArgumentParser()
#parser.add_argument('--src')    #path for folder containing images with objects that need to be blended
#parser.add_argument('--target')     #path for background image folder
#parser.add_argument('--output')     #path for folder whee final images need to be stored
parser.add_argument('--annotation', default = 'val_annotation.json')     #source annotation file
#parser.add_argument('--target_size', default = 512)     #size of final image
#parser.add_argument('--grid_size', default = 2)     #n denotes nxn grid
#parser.add_argument('--num_blend_obj', default = 1)      #number of objects that need to be blended in a single image
#parser.add_argument('--num_output_img', default = 1)     #total number of output images    
#parser.add_argument('--source_sizes', default = [192,256])     #possible image sizes for souce object image
#parser.add_argument('--source_rot', default = [0,90,180,270])      #possible rotations for source object image
par = parser.parse_args()

# save_dir = par.output
# ts = par.target_size
# num_obj = par.num_blend_obj
# count_img = par.num_output_img
# target_path = par.target
# source_path = par.src
# s_grid = par.grid_size
# ss_array = par.source_sizes
# rot_array = par.source_rot


# Load annotations
source_ann = COCO(par.annotation)
filename_image_map = {img['file_name']: img for img in source_ann.loadImgs(source_ann.getImgIds())}

# Annotation for generated images
ann_data = {"info":{
    "description": "Trash blending dataset",
    "url":"",
    "version":"",
    "year": 2025,
    "contributor":"IRVLab",
    "date_created":""
},
"licenses":[{}],
"images":[],
"annotations":[],
"categories":source_ann.loadCats(source_ann.getCatIds())
}

bg_array = os.listdir(target_path)
filename_array = os.listdir(source_path)
for i in range(1,4):    
    ss_array.append(ts//(i*s_grid))

annotation_id = 1     #initialize annotation id 
for count in range(1, count_img):
    #create a list of possible locations on the background 
    location = []
    ss_box = s_grid//2
    for j in range(s_grid):
        x = ss_box*j
        for k in range(s_grid):
            y = ss_box*k
            if x<ts-ss_box and y<ts-ss_box:
                location.append([x,y])

    for obj_num in range(num_obj):
        #selecting parameters and images
        filename = random.choice(filename_array)
        ss = random.choice(ss_array)
        l = len(bg_array)
        pool_no = random.randint(0,l-1)
        bg = bg_array[pool_no]
        rot = random.choice(rot_array)
        save_filename = "item_"+str(count)    #initilize name of save file
        target_file = target_path+bg
        source_file = source_path+filename
        temp_source = cv2.imread(source_file)
        source = cv2.resize(temp_source,(ss,ss))
        target = cv2.resize(cv2.imread(target_file),(ts,ts))
        factor = ss/temp_source.shape[0]

        [x_start,y_start] = random.choice(location)
        location.remove([x_start,y_start])
        offset = (x_start,y_start)


        img_info = filename_image_map[filename]
        ann_ids = source_ann.getAnnIds(imgIds=img_info['id'])
        anns_source = source_ann.loadAnns(ann_ids)
        seg = anns_source["segmentation"]
        
        seg_rotate = rotate_seg(seg_resized,rot,ss,factor)
        seg_mask = [0]*len(seg_rotate)
                
        for i in range(0,len(seg_rotate),2):
            seg_mask[i] = seg_rotate[i] + x_start
            seg_mask[i+1] = seg_rotate[i+1] + y_start

        create_mask(seg_rotate,source,save_dir)
        mask = cv2.resize(cv2.imread(save_dir+"mask.png"),(ss,ss))
        target = poisson_blending(source,target,mask,offset)
        
        # Store new annotations
        annotation_ann_data = {"id":annotation_id,
                                        "bbox":seg_to_box(seg_mask),
                                        "segmentation":[seg_mask],
                                        "area":anns_source['area'],
                                        "image_id":count,
                                        "iscrowd":0,
                                        "category_id":anns_source['category_id']}
        ann_data["annotations"].append(annotation_ann_data)
        annotation_is += 1

    #save first pass image and annotations
    cv2.imwrite(save_dir+"/poisson/"+save_filename+"_firstpass.png", target)

    images_ann_data = {"id":count,
                        "file_name":save_filename+".png",
                        "width": ts,
                        "height":ts}
    ann_data["images"].append(images_ann_data)

#--------------Syle transfer------------------
    orig = cv2.imread(target_file)
    h,w, r = orig.shape
    orig = cv2.resize(orig, (h,500))
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    mean = blur(gray, size=60)
    if mean >= 40:
        style_weight = 30000
    elif mean >= 10 and mean<40:
        style_weight = 15000
    elif mean >=0 and mean <10:
        style_weight = 1500
    else:
        style_weight = 800

    content_weight = 1; tv_weight = 10e-6
    ss = 512; ts = 512
    first_pass_img_file = save_dir+"/poisson/"+save_filename+"_firstpass.png"
    gpu_id = 0

    first_pass_img = np.array(Image.open(first_pass_img_file).convert('RGB').resize((ss, ss)))
    target_img = np.array(Image.open(target_file).convert('RGB').resize((ts, ts)))
    first_pass_img = torch.from_numpy(first_pass_img).unsqueeze(0).transpose(1,3).transpose(2,3).contiguous().float().to(gpu_id)
    #first_pass_img = torch.from_numpy(first_pass_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)
    source = np.array(Image.open(first_pass_img_file).convert('RGB').resize((ss, ss)))
    source = torch.from_numpy(source).unsqueeze(0).transpose(1,3).transpose(2,3).contiguous().float().to(gpu_id)
    target_img = torch.from_numpy(target_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)

    # Define LBFGS optimizer
    def get_input_optimizer(first_pass_img):
        optimizer = optim.LBFGS([first_pass_img.requires_grad_()])
        return optimizer

    optimizer = get_input_optimizer(first_pass_img)
    # Define Loss Functions
    mse = torch.nn.MSELoss()

    mean_shift = MeanShift(gpu_id)
    vgg = Vgg16().to(gpu_id)

    print('Optimizing...')
    run = [0]
    while run[0] <= num_steps:
    
        def closure():
              
            # Compute Loss Loss    
            target_features_style = vgg(mean_shift(target_img))
            target_gram_style = [gram_matrix(y) for y in target_features_style]
            blend_features_style = vgg(mean_shift(first_pass_img))
            blend_gram_style = [gram_matrix(y) for y in blend_features_style]
            style_loss = 0
            for layer in range(len(blend_gram_style)):
                style_loss += mse(blend_gram_style[layer], target_gram_style[layer])
            style_loss /= len(blend_gram_style)  
            style_loss *= style_weight        
        
            # Compute Content Loss
            content_features = vgg(mean_shift(first_pass_img))
            source_features_blend = vgg(mean_shift(source))
            content_loss = content_weight * mse(content_features.relu2_2, source_features_blend.relu2_2)

            # Compute TV Reg Loss
            tv_loss = torch.sum(torch.abs(first_pass_img[:, :, :, :-1] - first_pass_img[:, :, :, 1:])) + \
                       torch.sum(torch.abs(first_pass_img[:, :, :-1, :] - first_pass_img[:, :, 1:, :]))
            tv_loss *= tv_weight
        
            # Compute Total Loss and Update Image
            loss = style_loss + content_loss + tv_loss
            optimizer.zero_grad()
            loss.backward()
        
            # Print Loss
            if run[0] % 1 == 0:
                print("run {}:".format(run))
                print(' style : {:4f}'.format(\
                          style_loss.item()
                        ))
                print()
        
            run[0] += 1
            return loss
    
        optimizer.step(closure)

    # clamp the pixels range into 0 ~ 255
    first_pass_img.data.clamp_(0, 255)

    # Make the Final Blended Image
    input_img_np = first_pass_img.transpose(1,3).transpose(1,2).cpu().data.numpy()[0]
    plt.imsave(save_dir+"/style/"+save_filename+".png", input_img_np.astype(np.uint8))
    #plt.imsave(save_dir+"/style/"+"trial.png", input_img_np.astype(np.uint8))

f = open(save_dir+"/annotations_"+str(num_obj)+"_object.json","w")
json.dump(ann_data,f)
f.close()