""" Importing Required Libraries """

import torch
import torchvision
import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from DeepGlobeDataset import DeepGlobeDataset

""" Through this class, configurations are done for creating the dataset. """
class DataConfiguration:

    """ The constructor of this class defines various paths required to fetch and process data. """
    def __init__(self):
        self.project = "<Project_Directory>"
        self.dataset = self.project + "<Dataset_Directory>"
        self.class_dict_file = self.dataset + "/class_dict.csv"
        self.metadata_file = self.dataset + "/metadata.csv"

        """ Setting the batch size as 16 to process the images as Mini-Batches. """
        self.batch_size = 16

        """ Setting this to True to shuffle the data """
        self.shuffle = True

        """ GPU specific properties for faster computation and training. """
        self.pin_memory = True
        self.num_workers = 0

    """ This method transforms the images for faster training through the model. As SegNet architecture is based on 
    VGG Network and VGG Network works best with image size of 224*224. Hence, the images are resized and converted to 
    tensor. However, this can be changed to a bigger size but will result in longer training time. """
    def create_image_transform(self):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.CenterCrop(10),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                #  transforms.ToPILImage(),
                transforms.Resize((1024, 1024)),
                transforms.CenterCrop(10),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

    """ Opening the Metadata file to read the contents of the file and fetch data accordingly in order to process the 
    One-Hot encoding. """
    def get_class_configurations(self):
        reader = pd.read_csv(self.class_dict_file)
        self.class_names = reader['name'].tolist()
        self.class_rgb_values = reader[['r', 'g', 'b']].values.tolist()
        return self.class_names, self.class_rgb_values

    """ This method call the DeepGlobeDataset class to create the specific datasets for Training, Validation and 
    Testing. """
    def configure_dataset(self):
        self.create_image_transform()
        self.training_set = DeepGlobeDataset(self.dataset, self.metadata_file, 'train', self.image_transform,
                                             self.mask_transform, self.class_rgb_values)
        self.validation_set = DeepGlobeDataset(self.dataset, self.metadata_file, 'valid', self.image_transform,
                                               self.mask_transform, self.class_rgb_values)
        self.test_set = DeepGlobeDataset(self.dataset, self.metadata_file, 'test', self.image_transform,
                                         self.mask_transform, self.class_rgb_values)
        self.visual_set = DeepGlobeDataset(self.dataset, self.metadata_file, 'visualize', self.image_transform,
                                           self.mask_transform, self.class_rgb_values)

    """ This method uses the respective datasets created to form the respective dataloaders and returns these 
    dataloaders to be used later in the code. """
    def create_dataloaders(self):
       self.configure_dataset()
       self.training_loader = DataLoader(dataset=self.training_set, shuffle=self.shuffle, batch_size=self.batch_size,
                                         num_workers=self.num_workers, pin_memory=self.pin_memory)
       self.validation_loader = DataLoader(dataset=self.validation_set, shuffle=self.shuffle,
                                           batch_size=self.batch_size, num_workers=self.num_workers,
                                           pin_memory=self.pin_memory)
       self.testing_loader = DataLoader(dataset=self.test_set, shuffle=self.shuffle, batch_size=self.batch_size,
                                        num_workers=self.num_workers, pin_memory=self.pin_memory)
       self.visual_loader = DataLoader(dataset=self.visual_set, shuffle=self.shuffle, batch_size=self.batch_size,
                                       num_workers=self.num_workers, pin_memory=self.pin_memory)
       return self.training_loader, self.validation_loader, self.testing_loader, self.visual_loader

    def fetch_configuration(self):
       return self.project, self.dataset, self.class_dict_file, self.metadata_file, self.batch_size