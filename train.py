import matplotlib
import matplotlib.pyplot as plt

import torch

from torchvision import models, transforms
from torch import nn, optim
from torch.optim import lr_scheduler

from classifier import Classifier

# Create transforms 
dataTransforms = {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor()
    ]), 
    'test' : transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
}

# Determine folder where data is contained 
dataDir = './Data'

# Determine whether you're using a CPU or a GPU to build the deep learning network.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Finetuning the convnet
# Load a pretrained model and reset final fully connected layer.
model = models.resnet50(pretrained=True)

# Set Tensor flag's requires_grad to False.
# This allows us a grained exclusion of subgraphs from gradient computations
# and increase the model training efficiency. 
for param in model.parameters():
  param.requires_grad = False

# Feature extraction
numFtrs = model.fc.in_features
model = model.to(device)

# Model analysis
ml_classifier = Classifier(dataTransforms, model, device, dataDir)

# Here the size of each output sample is set to lenght of classNames list.
# Alternatively, it can be generalized to nn.Linear(numFtrs, len(ml_classifier.classNames))
model.fc = nn.Linear(numFtrs, len(ml_classifier.classNames))
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized.
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs.
expLrScheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train and evaluate
model = ml_classifier.trainModel(model, criterion, optimizer, expLrScheduler, numEpochs=6)

# Display testing
ml_classifier.visualizeModelPreview(model)

# Transform the new image into numbers and resize it.
testTransforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.Resize(224),
                                      transforms.ToTensor(),
                                    ])

# How many images do you want to see? It's set to 5, but you can change the number.
images, labels = ml_classifier.randomSamples(testTransforms, numSamples=5)
toPil = transforms.ToPILImage()
fig = plt.figure(figsize=(20,20))
classes = ml_classifier.classNames

for ii in range(len(images)):
    image = toPil(images[ii])
    sub = fig.add_subplot(1, len(images), ii+1)
    plt.imshow(image)
    plt.pause(0.01)

plt.title = classes
plt.show()

# Get a batch of training data
inputs, classes = next(iter(ml_classifier.dataLoaders['train']))