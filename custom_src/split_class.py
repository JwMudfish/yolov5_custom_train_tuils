import glob, os
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# parser = argparse.ArgumentParser(description='Dataset split train/test')
# parser.add_argument('--input_images_path', type=str)
# parser.add_argument('--train_file_name', type=str, default='train.txt')
# parser.add_argument('--test_file_name', type=str, default='test.txt')
# args = parser.parse_args()

class SplitData:
    def __init__(self, image_path, output_path, test_size = 0.1):
        self.image_path = image_path
        self.test_size = test_size
        # self.splited_path = image_path.split('/')[:-1]
        self.output_path = output_path
        self.train_file_name = 'train.txt'
        self.test_file_name = 'test.txt'
        
        self.train_file_path = os.path.join(self.output_path, self.train_file_name)
        self.test_file_path = os.path.join(self.output_path, self.test_file_name)
    
    def run(self):
        print('Split Train Test .............')
        dataset = [f for f in os.listdir(self.image_path) if f[-4:] == '.jpg']
        train, test = train_test_split(dataset, test_size = self.test_size)
        print(f'train_size: {len(train)}, test_size: {len(test)}')

        with open(self.train_file_path, 'w') as f:
            for t in train:
                f.write(os.path.join(os.path.abspath(self.image_path), t) + '\n')

        with open(self.test_file_path, 'w') as f:
            for t in test:
                f.write(os.path.join(os.path.abspath(self.image_path), t) + '\n')


if __name__ == '__main__':
    DATA_FOLDER_PATH = '/home/test/2_Yolo/data'
    DATASET_NAME = 'test'

    DATASET_PATH = os.path.join(DATA_FOLDER_PATH, DATASET_NAME)
    IMAGE_PATH = os.path.join(DATASET_PATH, 'images')

    SplitData(image_path = IMAGE_PATH, output_path = DATASET_PATH).run()

