import subprocess
import os
import json
import random
import shutil
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import cv2

def extract_images(video_path, output_directory, output_format='jpg', frame_rate=1):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Construct the FFmpeg command
    ffmpeg_cmd = [
        'ffmpeg',
        '-hide_banner',
        '-i', video_path,
        '-vf', f'fps={frame_rate}',
        os.path.join(output_directory, f'{video_path.split(".")[0].split("/")[1]}_frame_%04d.{output_format}')
    ]

    # Run the FFmpeg command
    subprocess.call(ffmpeg_cmd)

    print(f"Images extracted and saved to {output_directory}")

def train_val_test_split(json_file_path, output_dir, img_path):
    train_dir = os.path.join(output_dir, "train")
    validation_dir = os.path.join(output_dir, "validation")
    test_dir = os.path.join(output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    with open(json_file_path, 'r') as json_file:
        coco_data = json.load(json_file)

    image_ids = [img['id'] for img in coco_data['images']]

    random.seed(42)

    random.shuffle(image_ids)

    train_size = int(0.7 * len(image_ids))
    validation_size = int(0.15 * len(image_ids))
    test_size = int(0.15 * len(image_ids))
    
    train_set = image_ids[:train_size]
    validation_set = image_ids[train_size:train_size + validation_size]
    test_set = image_ids[train_size + validation_size:]

    for img_id in train_set:
        img_info = next(img for img in coco_data['images'] if img['id'] == img_id)
        img_file = img_info['file_name']
        src_img_path = os.path.join(img_path, img_file)
        dst_img_path = os.path.join(train_dir, img_file)
        shutil.copy(src_img_path, dst_img_path)


    for img_id in validation_set:
        img_info = next(img for img in coco_data['images'] if img['id'] == img_id)
        img_file = img_info['file_name']
        src_img_path = os.path.join(img_path, img_file)
        dst_img_path = os.path.join(validation_dir, img_file)
        shutil.copy(src_img_path, dst_img_path)
    
    
    for img_id in test_set:
        img_info = next(img for img in coco_data['images'] if img['id'] == img_id)
        img_file = img_info['file_name']
        src_img_path = os.path.join(img_path, img_file)
        dst_img_path = os.path.join(test_dir, img_file)
        shutil.copy(src_img_path, dst_img_path)

    train_data = {
    "info": coco_data["info"],
    "categories": coco_data["categories"],
    "images": [img for img in coco_data['images'] if img['id'] in train_set],
    "annotations": [ann for ann in coco_data['annotations'] if ann['image_id'] in train_set]
    }

    validation_data = {
        "info": coco_data["info"],
        "categories": coco_data["categories"],
        "images": [img for img in coco_data['images'] if img['id'] in validation_set],
        "annotations": [ann for ann in coco_data['annotations'] if ann['image_id'] in validation_set]
    }

    test_data = {
        "info": coco_data["info"],
        "categories": coco_data["categories"],
        "images": [img for img in coco_data['images'] if img['id'] in test_set],
        "annotations": [ann for ann in coco_data['annotations'] if ann['image_id'] in test_set]
    }

    train_json_path = os.path.join(train_dir, "coco_annotation_train.json")
    validation_json_path = os.path.join(validation_dir, "coco_annotation_validation.json")
    test_json_path = os.path.join(test_dir, "coco_annotation_test.json")


    with open(train_json_path, 'w') as train_file:
        json.dump(train_data, train_file)
    
    with open(validation_json_path, 'w') as validation_file:
        json.dump(validation_data, validation_file)
    
    with open(test_json_path, 'w') as test_file:
        json.dump(test_data, test_file) 

class DatasetWithTransform(Dataset):
  def __init__(self, 
                data,
                img_dir="",
                transform = None):
    self.data = data
    self.img_dir = img_dir
    self.transform = transform
      
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    file_name  = os.path.join(self.img_dir, 
                              self.data[idx]['file_name'])
    image = cv2.imread(file_name)
    if self.transform is not None:
        image = self.transform(image = image)['image']
    
    return image

