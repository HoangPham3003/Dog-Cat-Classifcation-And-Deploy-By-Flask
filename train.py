import os
import torch
from torch import nn
import pickle
import torchvision
from tqdm import tqdm
import torch.nn.functional as F

from vgg16 import VGG16
from datasets import DogCatDataLoader


dataset_folder = "Data/train/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 2
num_epochs = 10
batch_size = 16
learning_rate = 0.005
weight_decay = 0.005

# model = VGG16(num_classes).to(device)

model = torchvision.models.vgg16(pretrained=True)
model.classifier[-1] = nn.Linear(in_features=4096, out_features=2)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# dataset loader
dataset_file = "../input/cat-dog-recognition/train.csv"
data_loader = DogCatDataLoader(dataset_file=dataset_file, batch_size=batch_size)
train_loader, valid_loader = data_loader.create_data()
# train_loader = data_loader.create_data()

# Train the model
iterations = len(train_loader)

print("Start training...")
loss_opt = 1e9
for epoch in range(num_epochs):
    loss = None
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      
    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, iterations, loss.item()))
    if loss < loss_opt:
        loss_opt = loss
        torch.save(model.state_dict(), 'best.pth')
            
    # Validation
    with torch.no_grad(): 
        # turn off model.train() -> turn on model.eval() -> turn off model.eval() -> and then auto turn on model.train() again
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)
#             print(outputs)
            _, predicted = torch.max(outputs.data, 1)
#             print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
        print('Accuracy of the network on the validation images: {} %'.format(100 * correct / total))  