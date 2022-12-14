import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import sys
import glob
from xml_to_txt_class import XmlToTxt
from split_class import SplitData
import yaml
from od_augmentation import Augmentation

DATA_FOLDER_PATH = '/home/test/2_Yolo/data'
DATASET_NAME = 'ciga_9'
AUGMENTATION = False

##############################################################################

DATASET_PATH = os.path.join(DATA_FOLDER_PATH, DATASET_NAME)
LABEL_MAP_PATH = os.path.join(DATASET_PATH, 'predefined_classes.txt')

IMAGE_PATH = os.path.join(DATASET_PATH, 'images')
XML_PATH = os.path.join(DATASET_PATH, 'xmls')
TXT_PATH = os.path.join(DATASET_PATH, 'labels')  # txt 저장경로

if AUGMENTATION == True:
    AUG_DATA_PATH = os.path.join(DATASET_PATH, 'auged_data')

    AUG_IMAGE_PATH = os.path.join(AUG_DATA_PATH, 'images')
    AUG_XML_PATH = os.path.join(AUG_DATA_PATH, 'xmls')  
    AUG_TXT_PATH = os.path.join(AUG_DATA_PATH, 'labels')


def make_yaml(train_path, val_path, label_list, epoch, yaml_path, dataset_name,
              img_size = 960, batch = 32,
              model_path = '/home/test/2_Yolo/yolov5/models',
              model_name = 'yolov5m'):

    result = {
        'train' : train_path,
        'val' : val_path,
        'nc' : len(label_list),
        'names' : label_list,
        'epoch' : epoch,
        'name' : dataset_name,
        'cfg' : os.path.join(model_path, f'{model_name}.yaml'),
        'data' : os.path.join(yaml_path, f'{dataset_name}.yaml'),
        'img' : img_size,
        'batch' : batch,
    }

    with open(os.path.join(yaml_path,f'{dataset_name}.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(result, f, allow_unicode=True)

    return result


###########################################################################


if AUGMENTATION == True:
    print('Augmentation Mode....')
    ag = Augmentation(input_images_path = IMAGE_PATH, input_xmls_path = XML_PATH,
                        output_path = AUG_DATA_PATH, num_of_aug = 2)
    ag.run()

    XML_PATH = AUG_XML_PATH
    IMAGE_PATH = AUG_IMAGE_PATH
    TXT_PATH = AUG_TXT_PATH
    DATASET_PATH = AUG_DATA_PATH

### 1. xml to txt  ####
xt = XmlToTxt(xml_path = XML_PATH, label_map_path = LABEL_MAP_PATH, output_path=TXT_PATH)
xt.convert_xml2txt()

### 2. split train, test
sd = SplitData(image_path = IMAGE_PATH, output_path = DATASET_PATH)
sd.run()

### 3. make yaml file
cfg = make_yaml(train_path = sd.train_file_path, 
          val_path = sd.test_file_path, 
        label_list = xt.read_labels(), 
        epoch = 300,
        yaml_path = DATASET_PATH, 
        dataset_name = DATASET_NAME)

#print(f'python3 /home/test/2_Yolo/yolov5/train.py --img {cfg["img"]}')

### train
os.system(f"python3 /home/test/2_Yolo/yolov5/train.py \
            --img {cfg['img']} \
            --batch {cfg['batch']} \
            --epochs {cfg['epoch']} \
            --data {cfg['data']} \
            --cfg {cfg['cfg']} \
            --name {cfg['name']} \
            --single-cls \
            ")



# with open('/home/test/2_Yolo/data/test/test1.yaml') as f:
#     conf = yaml.load(f)

# print(conf['names'])
