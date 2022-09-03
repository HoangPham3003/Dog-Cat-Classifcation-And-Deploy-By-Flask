import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
from PIL import Image


class DogCatPredicter:
    def __init__(self, image_path=None, check_point='./best.pth'):
        self.image_path = image_path
        self.check_point = check_point
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.get_model()
        
    

    def get_model(self):
        model = torchvision.models.vgg16()
        model.classifier[-1] = nn.Linear(in_features=4096, out_features=2)
        model.load_state_dict(torch.load(self.check_point, map_location=torch.device('cpu')))
        model = model.to(self.device)
        model.eval()
        return model

    def predict(self):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010],)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)

        output = self.model(image)
        output = F.softmax(output, dim=1)
        value_hat, label_hat = torch.max(output.data, 1)
        label_hat = np.array(label_hat)[0]
        return label_hat

