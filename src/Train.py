""" Importing Required Libraries """

import torch
import torchvision
import copy
import numpy as np
import torch.nn as nn
import torch.optim as opti
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from DataConfiguration import DataConfiguration
from SegNet_VGG13 import SegNet
# from SegNet_VGG13_Pretrained import SegNet


""" Entire Training and Validation of the Network is done through this class. """
class Train():

    """ Init method to fetch required constructor properties. This receives all the required data configuration
    properties including the data loaders, class rgb values and the datasets. Also, it initializes various
    hyperparameters required for the training """
    def __init__(self):

        """ Calls the DataConfiguration which in turn calls the method to create the respective dataloaders """
        self.configuration = DataConfiguration()
        self.save_path = "<Project_Directory>"
        self.class_names, self.class_rgb_values = self.configuration.get_class_configurations()
        self.training_loader, self.validation_loader, self.testing_loader, self.visual_loader = \
            self.configuration.create_dataloaders()
        self.project, self.dataset, self.class_dict_file, self.metadata_file, self.batch_size = \
            self.configuration.fetch_configuration()

        """ Defining the initial hyperparameters """
        self.learning_rate = 0.001
        self.number_of_epochs = 100
        self.number_of_classes = 7

        """ Creating the model object as well as defining the Cross Entropy Loss function which uses LogSoftmax and 
        calculates the loss. Along with this, optimizer is created using momentum, regularization and scheduling the 
        learning rate"""
        self.model = SegNet(self.number_of_classes)
        # self.model = SegNet(self.number_of_classes).get_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9,
                                         weight_decay=0.0005)
        self.scheduler = opti.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)

        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(torch.device(self.device))
        self.mask_transform = transforms.Compose([transforms.ToTensor()])

    """ This method loads the model and its state from the previously trained checkpoints """
    def load_model(self, path, location):
        checkpoint = torch.load(path, map_location=torch.device(location))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def fetch_data_loaders(self):
        return self.training_loader, self.validation_loader, self.testing_loader, self.visual_loader

    """ This methods is used to visualize the Raw Images, Raw Masked Images and One-Hot Encoded Masks """
    def visualize_original_data(self):

        count = 0
        iterations = 0
        grid_images = []
        grid_masks = []
        grid_colour = []

        for batch_idx, (images, masks, colour_masks) in enumerate(self.visual_loader):

            grid_images = [np.transpose(image.numpy(), (1, 2, 0)) for image in images]
            grid_masks = [np.transpose(image.numpy(), (1, 2, 0)) for image in masks]
            grid_colour = [image for image in colour_masks]

        for iteration in range(len(grid_images)):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

            ax1.imshow(grid_images[iteration])
            ax2.imshow(grid_masks[iteration])
            ax3.imshow(grid_colour[iteration])
            plt.show()

    """ This method is used to visualize the training done and the output received while training. It is useful to see 
    how the training is progressing."""
    def training_visualization(self, images, masks, outputs):
        images, masks, outputs = images.cpu().detach(), masks.cpu().detach().numpy(), outputs.cpu().detach().numpy()
        colour_codes = np.array(self.class_rgb_values)

        # ground_truth_array = []
        # for i in range(len(masks)):
        #     colour_mask = colour_codes[masks[i]]
        #     temp_mask = Image.fromarray(colour_mask.astype('uint8'), 'RGB')
        #     ground_truth = self.mask_transform(temp_mask)
        #     ground_truth_array.append(ground_truth)
        # ground_truths = torch.stack(ground_truth_array, 0)

        # prediction_array = []
        # for i in range(len(outputs)):
        #     class_mask = np.argmax(outputs[i], axis=0).astype('int')
        #     colour_prediction = colour_codes[class_mask]
        #     temp_prediction = Image.fromarray(colour_prediction.astype('uint8'), 'RGB')
        #     prediction = self.mask_transform(temp_prediction)
        #     prediction_array.append(prediction)
        # predictions = torch.stack(prediction_array, 0)

        # image_grid = torchvision.utils.make_grid(images)
        # ground_truth_grid = torchvision.utils.make_grid(ground_truths)
        # prediction_grid = torchvision.utils.make_grid(predictions)

        # image_grid = np.transpose(image_grid.numpy(), (1, 2, 0))
        # ground_truth_grid = np.transpose(ground_truth_grid.numpy(), (1, 2, 0))
        # prediction_grid = np.transpose(prediction_grid.numpy(), (1, 2, 0))

        grid_images = [np.transpose(image.numpy(), (1, 2, 0)) for image in images]
        grid_masks = [image for image in masks]
        grid_colour = [np.argmax(image, axis=0).astype('int') for image in outputs]

        for iteration in range(len(grid_images)):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

            ax1.imshow(grid_images[iteration])
            ax2.imshow(grid_masks[iteration])
            ax3.imshow(grid_colour[iteration])
            plt.show()

        # fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(25, 5))
        # ax1.imshow(image_grid)
        # ax2.imshow(ground_truth_grid)
        # ax3.imshow(prediction_grid)
        # plt.show()

    """ This methods just displays various parameters """
    def get_model_parameters(self):
        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

        # Print optimizer's state_dict
        print("Optimizer's state_dict:")
        for var_name in self.optimizer.state_dict():
            print(var_name, "\t", self.optimizer.state_dict()[var_name])

    """ This method helps find out the mean IoU Score for a particular batch and help analyse the model's 
    performance """
    def get_iou(self, ground_truths, predictions):
        mean_iou = 0
        for i in range(len(predictions)):
            prediction = predictions[i].view(-1)
            truth = ground_truths[i].view(-1)

            intersection = (prediction * truth).sum()
            total = (prediction + truth).sum()
            union = total - intersection

            iou = intersection / union
            mean_iou += iou

        mean_iou /= len(predictions)
        return mean_iou

    """ Similar to IoU, this method helps find out the mean Class Pixel Accuracy for a particular batch and help analyse the 
    model's performance """
    def get_accuracy(self, ground_truths, predictions):
        accuracy = 0
        for i in range(len(predictions)):
            c, w, h = predictions[i].size()
            total_pixels = w * h
            prediction = predictions[i]
            truth = ground_truths[i]
            output = torch.argmax(prediction, dim=0)
            iteration_accuracy = (torch.sum((output == truth))).to(dtype=torch.float) / total_pixels
            accuracy += iteration_accuracy
        accuracy /= len(predictions)
        return accuracy

    """ Similar to IoU and accuracy, this method helps find out the mean Class Pixel F1 score along with precision and 
        recall for a particular batch and help analyse the model's performance """
    def get_f1_score(self, ground_truths, predictions):
        precision_arr = []
        recall_arr = []
        f1_arr = []
        for i in range(len(predictions)):
            c, w, h = predictions[i].size()
            prediction = predictions[i]
            truth = ground_truths[i]
            output = torch.argmax(prediction, dim=0)

            tp = (truth * output).sum()
            tn = ((1 - truth) * (1 - output)).sum()
            fp = ((1 - truth) * output).sum()
            fn = (truth * (1 - output)).sum()

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

            f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

            precision_arr.append(precision)
            recall_arr.append(recall)
            f1_arr.append(f1_score)

        return np.average(precision_arr), np.average(recall_arr), np.average(f1_arr)


    """ This is a common method used to fetch all the result metrics """
    def get_accuracies(self, ground_truths, predictions):

        iou = self.get_iou(ground_truths, predictions)
        accuracy = self.get_accuracy(ground_truths, predictions)
        precision, recall, f1_score = self.get_f1_score(ground_truths, predictions)

        return iou, accuracy, precision, recall, f1_score

    """ This model helps save the checkpoints, overall network as well as the model with the best performance so far """
    def save_checkpoint(self, model, optimizer, check_point_index):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, self.save_path + "segnet_cp_" + str(check_point_index) + ".pth")

        print("Checkpoint saved at {}".format(self.save_path + "segnet_cp_" + str(check_point_index) + ".pth"))

    """ This is the main method, which is used to train the model."""
    def train_model(self):

        total_training_loss = 0
        total_validation_loss = 0
        best_acc = 0.0

        """ Starting with the Training of the Model """
        for epoch in range(self.number_of_epochs):

            """ Defining the variables to keep track of the validation and training loss and accuracies """
            running_loss_training = 0.0
            running_corrects_training = 0
            running_loss_validation = 0.0
            running_corrects_validation = 0
            total_epoch_loss_training = 0
            total_epoch_loss_validation = 0

            self.model.train()
            for batch_idx, (images, masks) in enumerate(self.training_loader):

                if self.device == 'cuda':
                    images, masks = images.cuda(), masks.cuda()

                """ optimizer.zero_grad() operation- this empties the gradient tensors from previous batch so that the 
                gradients for the new batch are calculated anew """
                self.optimizer.zero_grad()

                """ Forward Processing """
                output_images = self.model(images)
                masked_images = masks
                values, preds = torch.max(output_images, 1)

                """ Calling the loss funtion to get the loss """
                loss = self.criterion(output_images.float(), masked_images.long())
                total_epoch_loss_training += loss.item()
                running_loss_training += loss.item() * images.size(0)
                running_corrects_training += torch.sum(preds == masked_images)
                iou, accuracy, precision, recall, f1_score = self.get_accuracies(masked_images, output_images)
                print(
                   f'Epoch: {epoch}, Batch: {batch_idx}, Batch Loss: {loss.item() / self.batch_size}, '
                   f'Epoch Loss: {running_loss_training}, Batch Corrects: {running_corrects_training}, '
                   f'Batch Accuracy: {accuracy}, Batch Precision: {precision}, Batch Recall: {recall}, '
                   f'Batch F1 Score: {f1_score}')

                loss.backward()
                self.optimizer.step()

                # self.training_visualization(images, masks, F.softmax(output_images, dim=-1, dtype=torch.float))


            self.model.eval()
            for batch_idx, (images, masks) in enumerate(self.validation_loader):

                if self.device == 'cuda':
                    images, masks = images.cuda(), masks.cuda()

                """ optimizer.zero_grad() operation- this empties the gradient tensors from previous batch so that the gradients for the new batch are calculated anew """
                self.optimizer.zero_grad()

                """ Forward Processing """
                output_images = self.model(images)
                masked_images = masks
                values, preds = torch.max(output_images, 1)

               """ Calling the loss funtion to get the loss """
               loss = self.criterion(output_images.float(), masked_images.long())
               total_epoch_loss_validation += loss.item()
               running_loss_validation += loss.item() * images.size(0)
               running_corrects_validation += torch.sum(preds == masked_images)
               iou, accuracy = self.get_accuracies(masked_images, output_images)
               print(
                   f'Epoch: {epoch}, Batch: {batch_idx}, Batch Loss: {loss.item() / self.batch_size}, Epoch Loss: {running_loss_validation}, Batch Corrects: {running_corrects_validation}, Batch Accuracy: {accuracy}')

               # self.training_visualization(images, masks, F.softmax(output_images, dim=-1, dtype=torch.float))

            self.scheduler.step()
            epoch_loss_training = running_loss_training / (150 * 4)
            epoch_acc_training = running_corrects_training.double() / (150 * 4)

            total_training_loss += total_epoch_loss_training / (150 * 4)
            print(f'============ Training Average Loss: {total_training_loss} ============')

            epoch_loss_validation = running_loss_validation / (50 * 4)
            epoch_acc_validation = running_corrects_validation.double() / (50 * 4)

            total_validation_loss += total_epoch_loss_validation / (50 * 4)
            print(f'============ Validation Average Loss: {total_validation_loss} ============')

            if epoch_acc_validation > best_acc:
                best_acc = epoch_acc_validation
                best_model_wts = copy.deepcopy(self.model.state_dict())

                self.save_checkpoint(self.model, self.optimizer, 99999)
            else:
                self.save_checkpoint(self.model, self.optimizer, epoch)

            print("Training complete. Saving checkpoint...")
            self.save_checkpoint(self.model, self.optimizer, 000)