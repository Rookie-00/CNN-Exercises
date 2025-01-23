
import torch
import torch.utils
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn 
from torch.nn import Conv2d

batch_size = 64
learning_rate = 0.001
epochs = 40

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


dataset_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_set = torchvision.datasets.CIFAR10(root="./dataset",train = True,transform=dataset_transforms,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",train = False,transform=dataset_transforms,download=True)

train_loader = DataLoader(dataset = train_set,batch_size = batch_size,shuffle = True)
test_loader = DataLoader(dataset = test_set,batch_size = batch_size,shuffle = False)


# class Arcanum(nn.Module):
#     def __init__(self):
#         super(Arcanum, self).__init__()
#         self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
#         self.conv2 = Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

#         self.fc1 = nn.Linear(16 * 6 * 6, 120)  
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)  

#     def forward(self, x):
#         x = self.pool(nn.ReLU()(self.conv1(x)))  
#         x = self.pool(nn.ReLU()(self.conv2(x)))  
#         x = x.view(-1, 16 * 6 * 6)  
#         x = nn.ReLU()(self.fc1(x))
#         x = nn.ReLU()(self.fc2(x))
#         x = self.fc3(x)
#         return x


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

    
model = Arcanum().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

writer = SummaryWriter("./")
# for data in train_loader:
#     imgs, targets = data
#     output = model(imgs)
#     print(imgs.shape)
#     print(output.shape)
print("begain to train the model...")


for epoch in range(epochs):
    running_loss = 0.0
    for batch_idx, (imgs,labels) in enumerate(train_loader):

        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        output = model(imgs)

        loss =  criterion(output,labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print (f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    writer.add_scalar("Loss/train", running_loss / len(train_loader), epoch)
    print(f"Epoch {epoch+1}, average Loss: {running_loss / len(train_loader):.4f}")

print("trainning finish！")

# test model
print("begain to test model...")
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1) 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"accuracy of training dataset {accuracy:.2f}%")

# save
torch.save(model.state_dict(), "cifar10_cnn.pth")
print("model saved as cifar10_cnn.pth")

# close TensorBoard record
writer.close()


