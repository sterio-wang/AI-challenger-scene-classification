import json
from scipy.misc import imread, imresize
import numpy as np
import os

class initialize(object):
    # Get image-label list for train and validation
    def __init__(self, feature_path, label_path):
        self.image_label_dict = {}
        with open(label_path, 'r') as f:
            label_list = json.load(f)
        for image in label_list:
            self.image_label_dict[image['image_id']] = int(image['label_id'])
        self.start = 0
        self.end = 0
        self.length = len(self.image_label_dict) # number of feature images
        self.image_name = list(self.image_label_dict.keys())
        self.feature_path = feature_path
    
    # Read image in feature path, resize and normalize to [-1, 1]
    def get_image(self, image_path, image_size):
        image = imread(image_path)
        image = imresize(image, [image_size, image_size])       
        image = np.array(image).astype(np.float32)
        image = 2 * (image - np.min(image)) / np.ptp(image) - 1
        return image
    
    # Get feature and label batch
    def get_batch(self, batch_size, image_size):
        self.start = self.end
        if self.start >= self.length:
            self.start = 0
        batch_feature = []
        batch_label = []
        index = self.start

        while len(batch_feature) < batch_size:
            i_image_path = os.path.join(self.feature_path, self.image_name[index])
            i_image = self.get_image(i_image_path, image_size)
            i_label = self.image_label_dict[self.image_name[index]]

            batch_feature.append(i_image)
            batch_label.append(i_label)
            index += 1
        self.end = index
        return batch_feature, batch_label