""" Importing Required Libraries """
import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

""" Defining Dataset through class DeepGlobeDataset """
class DeepGlobeDataset(Dataset):

    """ Init method to fetch required constructor properties. This helps form the separate data subsets based on the
    literal 'train','valid' or 'test'. Fetches the path for the dataset, metadata file, subset string and the image
    transformer. """
    def __init__(self, dataset_path, metadata_file_path, subset, image_transform, mask_transform, class_rgb_values):

        """ Fetching and initializing the constructor/init properties which are used throughout the class """
        self.dataset_path = dataset_path
        self.metadata_file_path = metadata_file_path
        self.subset = subset
        self.transform = image_transform
        self.mask_transform = mask_transform
        self.class_rgb_values = class_rgb_values

        """ Defining the arrays required to fetch the required data for executing the model """
        self.training_data = []
        self.training_masks = []
        self.validation_data = []
        self.validation_masks = []
        self.testing_data = []
        self.testing_masks = []

        """ Opening the Metadata file to read the contents of the file and fetch data accordingly. The metadata file 
        contains the split information about the training, validation and testing data. """
        with open(self.metadata_file_path) as file:
            reader = csv.reader(file, delimiter=',')
            line_count = 0
            for row in reader:
                if line_count > 0 and row[1] == 'train':
                    self.training_data.append(self.dataset_path + row[2])
                    self.training_masks.append(self.dataset_path + row[3])
                elif line_count > 0 and row[1] == 'valid':
                    self.validation_data.append(self.dataset_path + row[2])
                    self.validation_masks.append(self.dataset_path + row[3])
                elif line_count > 0 and row[1] == 'test':
                    self.testing_data.append(self.dataset_path + row[2])
                    self.testing_masks.append(self.dataset_path + row[3])
                line_count += 1

    """ Overriding the superclass method to return model specific values. This returns the length based on the string 
    literal 'train', 'valid' or 'test' """
    def __len__(self):
        if self.subset == 'train' or self.subset == 'visualize':
            return len(self.training_data)
        elif self.subset == 'valid':
            return len(self.validation_data)
        elif self.subset == 'test':
            return len(self.testing_data)

    """ This method converts a segmentation image label array to one-hot format by replacing each pixel value with a 
    vector of length num_classes and returns a 2D array with the same width and height as the input, but with a depth 
    size of num_classes"""
    def one_hot_encoding(self, mask):
        encoded_mask = []

        """ Converting the masked image into numpy array for processing """
        local_mask = np.asarray(mask)
        for colour in self.class_rgb_values:

            """" Matching the pixel values for all the channels (RGB) and comparing that with the colour (RGB) for that 
            specific class. This equality comparison, creates a boolean map, which is then converted to a single channel 
            binary map. """
            equality = np.equal(local_mask, colour)
            class_map = np.all(equality, axis=2)
            local_map = class_map.astype('float')
            encoded_mask.append(local_map)

        """ This binary map is then converted to class label index map (One-Hot Encoding Format) using argmax """
        encoded_mask = np.stack(encoded_mask, axis=-1)
        semantic_mask = np.argmax(encoded_mask, axis=-1).astype('int')

        return semantic_mask

    """ Overriding the superclass method to return model specific values. This returns the original image along with 
    its mask as required for training. In case of validation and testing, it returns the image. """
    def __getitem__(self, index):
        image_path = None
        mask_path = None
        if self.subset == 'train' or self.subset == 'visualize':
            image_path = self.training_data[index]
            mask_path = self.training_masks[index]
        elif self.subset == 'valid':
            image_path = self.validation_data[index]
            mask_path = self.validation_masks[index]
            return len(self.validation_data)
        elif self.subset == 'test':
            image_path = self.testing_data[index]
            mask_path = self.testing_masks[index]

        """ This where data transformation and augmentation is performed. This is necessary for both better performance 
        and faster processing. A seed value is used in order to make sure same level of transformation across all the 
        images. """
        seed = np.random.randint(2147483647)
        if self.transform is not None:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)

            if mask_path is not None and self.subset == 'visualize':
                mask = Image.open(mask_path).convert("RGB")
                mask = self.transform(mask)

                """ Commented code can be used to revert back to original mask from the One-Hot Encoding Format """
                # to_be_encoded = Image.open(mask_path).convert("RGB")
                # encoded_mask = self.one_hot_encoding(to_be_encoded)
                # colour_codes = np.array(self.class_rgb_values)
                # colour_mask = colour_codes[encoded_mask]
                # temp_mask = Image.fromarray(colour_mask.astype('uint8'), 'RGB')
                # one_hot_mask = self.mask_transform(temp_mask)

                to_be_encoded = Image.open(mask_path).convert("RGB")
                transformed_mask = self.mask_transform(to_be_encoded)
                encoded_mask = self.one_hot_encoding(transformed_mask)

                # colour_codes = np.array(self.class_rgb_values)
                # colour_mask = colour_codes[encoded_mask]
                # temp_mask = Image.fromarray(colour_mask.astype('uint8'), 'RGB')
                # one_hot_mask = self.transform(temp_mask)

                return image, mask, encoded_mask

            elif mask_path is not None:
                mask = Image.open(mask_path).convert("RGB")
                transformed_mask = self.mask_transform(mask)
                encoded_mask = self.one_hot_encoding(transformed_mask)
                processed_tensor = torch.from_numpy(encoded_mask)

                return image, processed_tensor