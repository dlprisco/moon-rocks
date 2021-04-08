# modules
import os
import time
import copy 

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F

from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
from torchvision import datasets, models, transforms

class Classifier(object):

    def __init__(self, data_transforms, model, device, data_dir):
        self.dataTransforms = data_transforms
        self.model = model
        self.dataDir = data_dir
        self.device = device

        # Images data (train, test)
        self.imageDatasets = {x: datasets.ImageFolder(self.dataDir, self.dataTransforms[x]) for x in ['train', 'test']}

        # Determine Length of training images and self.split them   
        self.numTrain, self.indices = len(self.imageDatasets['train']), list(range(len(self.imageDatasets['train'])))
        self.split = int(np.floor(.2 * self.numTrain))
        np.random.shuffle(self.indices)
        
        self.trainIdx, self.testIdx = self.indices[self.split:], self.indices[:self.split]
        self.trainSampler, self.testSampler = SubsetRandomSampler(self.trainIdx), SubsetRandomSampler(self.testIdx)

        self.dataLoaders = {x: torch.utils.data.DataLoader(self.imageDatasets[x], batch_size=16, shuffle=True, num_workers=2) for x in ['train', 'test']}
        self.datasetSizes = {x: len(self.imageDatasets[x]) for x in ['train', 'test']}

        # Split classes such as index 0, 1 for train and test
        self.classNames = [self.imageDatasets['train'].classes, self.imageDatasets['test'].classes]

    # A function to randomly select a set of images.
    def randomSamples(self, testTransforms, numSamples=5):
        """showRandomSamples function for Tensor."""
        data = datasets.ImageFolder(self.dataDir, testTransforms)
        classes = data.classes
        self.indices = list(range(len(data)))
        np.random.shuffle(self.indices)
        idx = self.indices[:numSamples]
        sampler = SubsetRandomSampler(idx)
        loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=numSamples)
        dataiter = iter(loader)
        images, labels = dataiter.next()
        
        return images, labels

    # Define a function to train model
    def trainModel(self, model, criterion, optimizer, scheduler, numEpochs=25):

        """This function take pytorch model and train with data."""
    
        since = time.time()
    
        bestModelWts = copy.deepcopy(model.state_dict())
        bestAcc = 0.0
    
        for epoch in range(numEpochs):
            print('Epoch {}/{}'.format(epoch, numEpochs-1))
            print('=' * 100)
    
            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':    
                    model.train()   # training mode
                else:
                    model.eval()    # evaluate mode    
    
                runningLoss = 0.0
                runningCorrects = 0
                
                # Iterate over data
                for inputs, labels in self.dataLoaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
    
                    # Zero the parameter gradients
                    optimizer.zero_grad()
    
                    # Track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
    
                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
    
                    # Statistics
                    runningLoss += loss.item() * inputs.size(0)
                    runningCorrects += torch.sum(preds == labels.data)
                
                if phase == 'train':
                    scheduler.step()
    
                epochLoss = runningLoss / self.datasetSizes[phase]
                epochAcc = runningCorrects.double() / self.datasetSizes[phase]
    
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epochLoss, epochAcc))
    
                # deep copy the model
                if phase == 'test' and epochAcc > bestAcc:
                    bestAcc = epochAcc
                    bestModelWts = copy.deepcopy(model.state_dict())    
            
            print()
        
        timeElapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            timeElapsed // 60, timeElapsed % 60))
        
        print('Best val Acc: {:4f}'.format(bestAcc))
    
        # Load best model weights
        model.load_state_dict(bestModelWts)
        
        # Return model
        return model

    # Visualize model prediction
    def visualizeModelPreview(self, model, numImages=8):
        """Function to display predictions."""
        wasTraining = model.training
        model.eval()
        imagesSoFar = 0
        fig = plt.figure()
    
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataLoaders['test']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
    
                ouputs = model(inputs)
                _, preds = torch.max(ouputs, 1)
    
                fig = plt.figure(figsize = (12.5, 25))
    
                for j in range(inputs.size()[0]):
                    imagesSoFar += 1
                    ax = plt.subplot(numImages//2, 2, imagesSoFar)
                    ax.set_title('predicted: {}'.format(self.classNames[0][preds[j]]))
    
                    imshowElement = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                    #mean = np.array([0.485, 0.456, 0.406])
                    #std = np.array([0.288, 0.292, 0.284])
    
                    #imshowElement = std * imshowElement + mean
                    imshowElement = np.clip(imshowElement, 0, 1)
    
                    plt.imshow(imshowElement)
                    if imagesSoFar == numImages:
                        model.train(mode=wasTraining)
                        return
    
            model.train(mode=wasTraining)
