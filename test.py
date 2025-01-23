
import torch
import torch.utils
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn 
from torch.nn import Conv2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Arcanum(nn.Module):
    def __init__(self):
        super(Arcanum, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.pool(nn.ReLU()(self.bn2(self.conv2(x))))
        x = self.pool(nn.ReLU()(self.bn3(self.conv3(x))))
        x = self.pool(nn.ReLU()(self.bn4(self.conv4(x))))
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  
        
        x = self.dropout(nn.ReLU()(self.fc1(x)))
        x = self.dropout(nn.ReLU()(self.fc2(x)))
        x = self.dropout(nn.ReLU()(self.fc3(x)))
        x = self.fc4(x)
        
        return x

# init the model
model = Arcanum().to(device)

# upload
model.load_state_dict(torch.load("cifar10_cnn.pth", map_location=device))

# setting as a evaluation mode
model.eval()

print("model is uploaded！")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# imread the dataset
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

# choose one image
dataiter = iter(test_loader)
image, label = next(dataiter)

# imshow the original image
def imshow(img):
    img = img / 2 + 0.5  
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

imshow(image[0])

# prediction
image = image.to(device)
output = model(image)
_, predicted = torch.max(output, 1)

# CIFAR-10 classification
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

print(f"real_class: {classes[label[0]]}")
print(f"model_prediction: {classes[predicted[0]]}")

