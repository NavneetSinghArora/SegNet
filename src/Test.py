""" Importing Required Libraries """

import torch.nn.functional as F
import torch
from Train import Train


""" Entire Testing of the Trained Network is done through this class. """
class Test():

    """ Init method to fetch required constructor properties. This receives all the required data configuration
    properties including the data loaders and the trained model  """
    def __init__(self):
        self.train = Train()
        self.train.load_model("<Best Model Path>", 'cuda')
        self.training_loader, self.validation_loader, self.testing_loader, self.visual_loader = self.train.fetch_data_loaders()
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')

    """ This method is used to test the model using the test data"""
    def test_model(self):
        for batch_idx, (images, masks) in enumerate(self.testing_loader):

            if batch_idx >= 150:
                print(f'   -- Testing Batch - {batch_idx}: In Progress')

                if self.device == 'cuda':
                    images, masks = images.cuda(), masks.cuda()

                # Forward Processing
                output_images = self.train.model(images)
                self.train.training_visualization(images, masks, F.softmax(output_images, dim=-1, dtype=torch.float))