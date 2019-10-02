import torch, torchvision
from torchvision import datasets, models, transforms
from predict import Prediction
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# validate image
def is_valid_image(image_path):
    try:
        Image.open(image_path)
        return True
    except:
        return False
    
class ImageRegression:
    
    def __init__(self, dataset_path, model=None, loss_func=None, optimizer=None, epochs=25):      
        self.dataset = dataset_path
        self.train_directory = os.path.join(self.dataset, 'train')
        self.valid_directory = os.path.join(self.dataset, 'valid')
        self.test_directory = os.path.join(self.dataset, 'test')

        # Batch size
        self.bs = 32

        # Device, use GPU when available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")              

        # Applying Transforms to the Data
        self.image_transforms = { 
            'train': transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ])
        }
               
        # Load pretrained ResNet50 Model
        if model == None: self.model = models.resnet50(pretrained=True)
        else: self.model = model
        
        self.model.to(self.device) 

        # Change the final layer of ResNet50 Model for Transfer Learning
        fc_inputs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

        # Define Optimizer and Loss Function
        if loss_func == None: self.loss_func = nn.MSELoss()
        else: self.loss_func = loss_func
        
        if optimizer == None: self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        else: self.optimizer = optimizer
        
        self.epochs = epochs
                
        # Print sizes
        print('Initialized.')


    def train_and_validate(self):
        '''
        Function to train and validate
        Parameters
    
        Returns
            model: Trained Model with best validation accuracy
            history: (dict object): Having training loss, accuracy and validation loss, accuracy
        '''
        
        # Load Data from folders
        data = {
            'train': datasets.ImageFolder(root=self.train_directory, transform=self.image_transforms['train'], is_valid_file=is_valid_image),
            'valid': datasets.ImageFolder(root=self.valid_directory, transform=self.image_transforms['valid'], is_valid_file=is_valid_image),
            'test': datasets.ImageFolder(root=self.test_directory, transform=self.image_transforms['test'], is_valid_file=is_valid_image)
        }
        
        # Size of Data, to be used for calculating Average Loss and Accuracy
        train_data_size = len(data['train'])
        valid_data_size = len(data['valid'])
        test_data_size = len(data['test'])

        print('Found (train,valid,test):', train_data_size, valid_data_size, test_data_size)

        # Get a mapping of the indices to the class names, in order to see the output classes of the test images.
        # self.idx_to_class = {v: k for k, v in self.data['train'].class_to_idx.items()}
    
        # Create iterators for the Data loaded using DataLoader module
        train_data_loader = DataLoader(self.data['train'], batch_size=self.bs, shuffle=True, num_workers=2)
        valid_data_loader = DataLoader(self.data['valid'], batch_size=self.bs, shuffle=True, num_workers=2)
        test_data_loader = DataLoader(self.data['test'], batch_size=self.bs, shuffle=True, num_workers=2)
        
        history = []

        for epoch in range(self.epochs):
            epoch_start = time.time()
            print("Epoch: {}/{}".format(self.epoch+1, self.epochs))
            
            # Set to training mode
            self.model.train()
            
            # Loss and Accuracy within the epoch
            train_loss = 0.0
            train_acc = 0.0
            
            valid_loss = 0.0
            valid_acc = 0.0
        
            print('Training step...')

            try:
            
                for i, (inputs, labels) in enumerate(train_data_loader):
                    try:
                        print(f'Step {i}')

                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        
                        # Clean existing gradients
                        self.optimizer.zero_grad()
                        
                        # Forward pass - compute outputs on input data using the model
                        outputs = self.model(inputs)
                        
                        # Compute loss
                        loss = self.loss_func(outputs, labels)
                        
                        # Backpropagate the gradients
                        loss.backward()
                        
                        # Update the parameters
                        self.optimizer.step()
                        
                        # Compute the total loss for the batch and add it to train_loss
                        train_loss += loss.item() * inputs.size(0)
                        
                        # Compute the accuracy
                        ret, predictions = torch.max(outputs.data, 1)
                        correct_counts = predictions.eq(labels.data.view_as(predictions))
                        
                        # Convert correct_counts to float and then compute the mean
                        acc = torch.mean(correct_counts.type(torch.FloatTensor))
                        
                        # Compute total accuracy in the whole batch and add to train_acc
                        train_acc += acc.item() * inputs.size(0)
                        
                    except Exception as e:
                        print(f'Error step {i}: {e}')
                        continue 
                
                #print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))
            except Exception as e:
                print(f'Error: {e}')
                pass
            
            print('Validation step...')
                
            # Validation - No gradient tracking needed
            with torch.no_grad():

                # Set to evaluation mode
                self.model.eval()
                
                try:
                
                    # Validation loop
                    for j, (inputs, labels) in enumerate(valid_data_loader):                
                        print(f'Step {j}')
                        
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        # Forward pass - compute outputs on input data using the model
                        outputs = self.model(inputs)

                        # Compute loss
                        loss = self.loss_func(outputs, labels)

                        # Compute the total loss for the batch and add it to valid_loss
                        valid_loss += loss.item() * inputs.size(0)

                        # Calculate validation accuracy
                        ret, predictions = torch.max(outputs.data, 1)
                        correct_counts = predictions.eq(labels.data.view_as(predictions))

                        # Convert correct_counts to float and then compute the mean
                        acc = torch.mean(correct_counts.type(torch.FloatTensor))

                        # Compute total accuracy in the whole batch and add to valid_acc
                        valid_acc += acc.item() * inputs.size(0)

                        #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item())     
                
                except Exception as e:
                    print(f'Error: {e}')
                    pass               
                
            # Find average training loss and training accuracy
            avg_train_loss = train_loss/self.train_data_size 
            avg_train_acc = train_acc/self.train_data_size

            # Find average training loss and training accuracy
            avg_valid_loss = valid_loss/self.valid_data_size 
            avg_valid_acc = valid_acc/self.valid_data_size

            history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
                    
            epoch_end = time.time()
        
            print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
            
            # Save if the model has best accuracy till now
            torch.save(self.model, self.dataset+'_model_'+str(epoch)+'.pt')
                
        return model, history
    
    def train_and_print(self):
        trained_model, history = self.train_and_validate()

        torch.save(history, self.dataset+'_history.pt')

        history = np.array(history)
        plt.plot(history[:,0:2])
        plt.legend(['Tr Loss', 'Val Loss'])
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss')
        plt.ylim(0,1)
        plt.savefig(self.dataset+'_loss_curve.png')
        plt.show()

        plt.plot(history[:,2:4])
        plt.legend(['Tr Accuracy', 'Val Accuracy'])
        plt.xlabel('Epoch Number')
        plt.ylabel('Accuracy')
        plt.ylim(0,1)
        plt.savefig(self.dataset+'_accuracy_curve.png')
        plt.show()
