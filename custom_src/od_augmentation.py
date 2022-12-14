import random
import cv2
import os
import sys
import glob
import xml.etree.ElementTree as ET
from xml.dom import minidom
import albumentations as A
from tqdm import tqdm
from distutils.dir_util import copy_tree

class Augmentation:

    def __init__(self, input_images_path, input_xmls_path, output_path, num_of_aug):
        self.input_images_path = input_images_path
        self.input_xmls_path = input_xmls_path
        self.output_path = output_path
        self.num_of_aug = num_of_aug


    def make_dir(self):
        print('Augmentation output folder 생성중')
        if os.path.isdir(self.output_path):
            if os.path.isdir(self.output_path + '/images') and os.path.isdir(self.output_path + '/xmls'):
                pass
            else:
                os.makedirs(self.output_path + '/images')
                os.makedirs(self.output_path + '/xmls')
        else:
            os.makedirs(self.output_path)
            os.makedirs(self.output_path + '/images')
            os.makedirs(self.output_path + '/xmls')


    def get_boxes(self, label_path):
        tree = ET.parse(label_path)
        root = tree.getroot()
        obj_xml = root.findall('object')
        
        if obj_xml[0].find('bndbox') != None:

            result = []
            name_list = []
            idx = 0
            category_id = []

            for obj in obj_xml:
                bbox_original = obj.find('bndbox')
                names = obj.find('name')
            
                xmin = int(float(bbox_original.find('xmin').text))
                ymin = int(float(bbox_original.find('ymin').text))
                xmax = int(float(bbox_original.find('xmax').text))
                ymax = int(float(bbox_original.find('ymax').text))

                result.append([xmin, ymin, xmax, ymax])
                name_list.append(names.text)
                category_id.append(idx)
                idx+=1
            return result, name_list, category_id

    def make_categori_id(self, label_list):
        idx = 0
        category_id_to_name = {}

        for label in label_list:
            category_id_to_name.update({int(idx):str(label)})
            idx += 1

        return category_id_to_name

    def modify_coordinate(self, transformed, xml, idx):
        filename = xml.split('/')[-1]
        filename = filename.split('.')[0]
        print('aaaaaaaaaaaaaaaaaaa : ', filename)
        tree = ET.parse(xml)
        root = tree.getroot()
        obj_xml = root.findall('object')
        
        auged_height, auged_width, _ = transformed['image'].shape
        size_xml = root.find('size')
        size_xml.find('width').text = str(int(auged_width))
        size_xml.find('height').text = str(int(auged_height))
        
        bbox_mod = transformed['bboxes']

        for x, obj in enumerate (obj_xml):
            bbox_original = obj.find('bndbox')
            bbox_original.find('xmin').text = str(int(bbox_mod[x][0]))
            bbox_original.find('ymin').text = str(int(bbox_mod[x][1]))
            bbox_original.find('xmax').text = str(int(bbox_mod[x][2]))
            bbox_original.find('ymax').text = str(int(bbox_mod[x][3]))

            # del bbox_mod[0]

        root.find('filename').text = filename + '_' + str(idx) + '.jpg'
        tree.write(self.output_path + '/' + 'xmls/' + 'aug_' + filename + '_' + str(idx) + '.xml')


    def run(self):
        
        self.make_dir()

        print('원본 데이터 옮기는 중.......')
        copy_tree(self.input_images_path, os.path.join(self.output_path, 'images'))
        copy_tree(self.input_xmls_path, os.path.join(self.output_path, 'xmls'))

        print('Augmentation 시작!!!!!!!')
        image_set_path = self.input_images_path + '/*.jpg'
        image_list = sorted(glob.glob(image_set_path))

        xml_set_path = self.input_xmls_path + '/*.xml'
        xml_list = sorted(glob.glob(xml_set_path))

        cnt = 0
    
        for image, xml in tqdm(zip(image_list, xml_list), desc = 'Augmentations : '):        
            image_name = image.split('/')[-1]
            image_name = image_name.split('.')[0]

            xml_name = xml.split('/')[-1]

            image = cv2.imread(image)
            height, width, channels = image.shape

            # try:

            bbox, str_label, category_id = self.get_boxes(xml)
            category_id_to_name = self.make_categori_id(str_label)
            # print(image_name, xml_name, str_label)
            transform = A.Compose([
                A.HorizontalFlip(p=0.6),
                A.VerticalFlip(p=0.6),
                A.RandomBrightnessContrast(p=0.5),
                A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
                ],
                bbox_params = A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
            )

            for num in range(int(self.num_of_aug)):
                transformed = transform(image=image, bboxes=bbox, category_ids=category_id)
                #cv2.imwrite(output_path + '/' + 'images/' + image_name + '_' + str(x) + '.jpg', transformed['image'])
                cv2.imwrite(self.output_path + '/' + 'images/' +'aug_'+ image_name + '_' +str(num) + '.jpg', transformed['image'])
                self.modify_coordinate(transformed = transformed, xml = xml, idx = num)

            # except:
            #     print(cnt, xml_name, ' This file does not contain objects.')
            #     pass
            #     cnt += 1


if __name__ == "__main__":

    input_images_path = '/home/test/2_Yolo/data/ciga_test/images'
    input_xmls_path = '/home/test/2_Yolo/data/ciga_test/xmls'

    output_path = '/home/test/2_Yolo/data/ciga_test/output'
    aug_num = 2

    ag = Augmentation(input_images_path = input_images_path, input_xmls_path = input_xmls_path,
                       output_path = output_path, num_of_aug = 2)
    ag.run()
