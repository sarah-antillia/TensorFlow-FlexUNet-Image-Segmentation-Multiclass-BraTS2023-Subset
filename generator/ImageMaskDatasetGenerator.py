# Copyright 2025 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2025/11/28 ImageMaskDatasetGenerator.py

import os
import io
import sys
import shutil
import glob
import nibabel as nib
import numpy as np
from PIL import Image, ImageOps
import traceback

import cv2

#RESIZE = 512

class ImageMaskDatasetGenerator:

  def __init__(self, input_dir="./BraTS21/", type="flair", 
               output_dir="./BraTS21-master", angle=90, 
               limit_subdirs = True,
               resize=256):
    self.input_dir = input_dir 

    if not os.path.exists(self.input_dir):
      raise Exception("Not found " + input_dir)   

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)  
    os.makedirs(output_dir)

    self.output_images_dir = os.path.join(output_dir, "images")
    self.output_masks_dir  = os.path.join(output_dir, "masks")

    
    os.makedirs(self.output_images_dir)

    os.makedirs(self.output_masks_dir)

    self.angle = angle

    self.limit_subdirs = limit_subdirs

    self.SEG_EXT   = "-seg.nii"
    self.IMG_EXT = "-" + type + ".nii"
  
    self.RESIZE    = (resize, resize)
    self.file_format = ".png"

  def generate(self):
    subdirs = os.listdir(self.input_dir)
    subdirs = sorted(subdirs)

    #Limiting the number of sudirs 
    if self.limit_subdirs:
      # one fifth
      num_limited = int(len(subdirs)/5)
      subdirs = subdirs[:num_limited]
  
    index = 10000
    for subdir in subdirs:
      index += 1
      subdir_fullpath = os.path.join(self.input_dir, subdir)
      print("=== subdir {}".format(subdir))
      seg_file = glob.glob(subdir_fullpath + "/*" + self.SEG_EXT)[0]
      img_file = glob.glob(subdir_fullpath + "/*" + self.IMG_EXT)[0]
      print("seg_file {}, img_file {}".format(seg_file, img_file))
      self.generate_mask_files(seg_file    , index) 
      self.generate_image_files(img_file , index) 
    
  def normalize(self, image):
    min = np.min(image)/255.0
    max = np.max(image)/255.0
    scale = (max - min)
    image = (image -min) / scale
    image = image.astype('uint8') 
    return image
  
  # Modified to save plt-image to BytesIO() not to a file.
  def generate_image_files(self, nii_file, index):
    print(">>>> generate_image_files {}".format(nii_file))
    nii = nib.load(nii_file)
    fdata  = nii.get_fdata()
   
    w, h, d = fdata.shape
    print("shape {}".format(fdata.shape))
    for i in range(d):
      img = fdata[:,:, i]
      filename  = str(index) + "_" + str(i) + self.file_format
      filepath  = os.path.join(self.output_images_dir, filename)
      corresponding_mask_file = os.path.join(self.output_masks_dir, filename)
      if os.path.exists(corresponding_mask_file):
        img   = self.normalize(img)
        image = Image.fromarray(img)
        image = image.convert("RGB")
        image = image.resize(self.RESIZE)
        if self.angle>0:
          image = image.rotate(self.angle)
        image.save(filepath)
        print("=== Saved {}".format(filepath))
      else:
        pass
        #print("--- Skipped")

  def colorize_mask(self, mask):
    h, w = mask.shape[:2]
    colorized = np.zeros((h, w, 3), dtype=np.uint8)

    #whole tumor (WT), tumor core (TC), and enhancing tumor (ET), 
    #      BGR color
    BLUE    = (255,   0,   0)     #2: blue   
    GREEN   = (  0, 255,   0)     #3: green    
    RED     = (  0,   0,  255)    #1: red

    colorized[np.equal(mask, 1)] = BLUE
    colorized[np.equal(mask, 2)] = GREEN
    colorized[np.equal(mask, 3)] = RED

    return colorized

  def generate_mask_files(self, nii_file, index ):
    nii = nib.load(nii_file)
    fdata  = nii.get_fdata()
   
    w, h, d = fdata.shape
    print("shape {}".format(fdata.shape))
    for i in range(d):
      img = fdata[:,:, i]
    
      if img.any() >0:
        img = self.colorize_mask(img)
        img = img.astype('uint8')

        image = Image.fromarray(img)
        image = image.convert("RGB")
        image = image.resize(self.RESIZE)
        if self.angle >0:
          image = image.rotate(self.angle)
      
        filename  = str(index) + "_" + str(i) + self.file_format
        filepath  = os.path.join(self.output_masks_dir, filename)
        image.save(filepath)
        print("--- Saved {}".format(filepath))
      else:
        pass
        #print("--- Skipped")


if __name__ == "__main__":
  try:
    type       = "t2f"  # "t1c", "t1n", "t2f", t2w"
    if len(sys.argv) == 2:
      type = sys.argv[1]
    input_dir  = "./BRATS2023_PART_1"
    types       = ["t1c", "t1n", "t2f", "t2w"]
    if not (type in types):
      raise Exception ("Error: Invalid type " + type)
      
    output_dir = "./BraTS2023-" + type.upper() + "-Subset-master" 
    limit_subdirs = True
    generator = ImageMaskDatasetGenerator(input_dir = input_dir, 
                                          output_dir= output_dir,
                                          type  = type, 
                                          limit_subdirs = limit_subdirs,
                                          angle = 90)
    generator.generate()
  except:
    traceback.print_exc()

 
