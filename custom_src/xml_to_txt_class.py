# -*- coding: utf-8 -*-
import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import sys
import glob
from tqdm import tqdm

# 단일라벨 변경기능
class XmlToTxt:

    def __init__(self, xml_path, label_map_path, output_path):
        self.xml_path = xml_path
        self.label_map_path = label_map_path
        self.output_path = output_path

    def convert_coordinates(self, size, box):
        dw = 1./(size[0])
        dh = 1./(size[1])
        x = (box[0] + box[1])/2.0 - 1
        y = (box[2] + box[3])/2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh

        return (x,y,w,h)

    def read_labels(self):
        df = pd.read_csv(self.label_map_path, sep = '\n', index_col=False, header=None)
        classes = df[0].tolist()
        
        return classes


    def convert_xml2txt(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        label_list = self.read_labels()
        print('class : ', label_list)
        for fname in tqdm(glob.glob(self.xml_path + "/*.xml"), desc = 'convert xml to txt file'):
            print(fname)
            filename = fname.split('/')[-1]
            filename = filename.split('.')[0]

            tree = ET.parse(fname)
            root = tree.getroot()

            obj_xml = root.findall("object")
            size_xml = root.find('size')

            image_width = int(size_xml.find("width").text)
            image_hegiht = int(size_xml.find("height").text)

            if obj_xml[0].find('bndbox') != None:
                with open(self.output_path + '/' + filename + ".txt", 'w') as f:
                    for obj in obj_xml:
                        class_id = obj.find("name").text
                        if class_id in label_list:
                            label_str = str(label_list.index(class_id))
                        else:
                            label_str = "-1"
                            print (f"warning: {filename} label '%s' not in look-up table" %class_id)

                        bboxes = obj.find("bndbox")
                        xmin = float(float(bboxes.find('xmin').text))
                        ymin = float(float(bboxes.find('ymin').text))
                        xmax = float(float(bboxes.find('xmax').text))
                        ymax = float(float(bboxes.find('ymax').text))

                        box = (float(xmin), float(xmax), float(ymin), float(ymax))
                        # print(filename, image_width, image_hegiht, class_id, xmin, ymin, xmax, ymax)
                        result = self.convert_coordinates((image_width, image_hegiht), box)
                        # print(filename,image_width, image_hegiht, result)

                        f.write(label_str + " " + " ".join([("%.6f" % a) for a in result]) + '\n')



if __name__ == '__main__':

    input_path = '/home/test/2_Yolo/data/test/xml'
    label_map = '/home/test/2_Yolo/data/test/predefined_classes_test.txt'
    output_path = '/home/test/2_Yolo/data/test/txt'

    xt = XmlToTxt(xml_path = input_path, label_map_path = label_map, output_path=output_path)
    xt.convert_xml2txt()