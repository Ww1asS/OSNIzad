import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class ClassificatorCNN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64,3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64 * 7 * 7,128)
        self.fc2 = nn.Linear(128,num_classes)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

num_classes = 10

model = ClassificatorCNN(num_classes).to(device)

crit = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01)

num_epoch = 10

train_losses = []
test_losses = []

for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        vihod = model(images)
        loss = crit(vihod, labels)

        _, predicted_labels = torch.max(vihod,1)
        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    acc = correct / total
    train_losses.append(epoch_loss)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            vihod = model(images)
            loss = crit(vihod, labels)
            test_loss += loss.item() * images.size(0)

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print(f'Epoch [{epoch + 1}/{num_epoch}], Loss: {epoch_loss:.4f}, Accuracy: {acc:.4f}')

model.eval()
all_preds = []
all_labels = []
with torch.inference_mode():
    for images, labels in test_loader:  # ИЗМЕНЕНО: батчи
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)
print('\n', f'Accuracy: {accuracy:.4f}')

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Ошибка (обучение)", linewidth=2, color='blue')
plt.plot(test_losses, label="Ошибка (проверка)", linewidth=2, color='red')
plt.xlabel("Эпоха")
plt.ylabel("Значение ошибки")
plt.title("Изменение функции ошибки")
plt.legend()
plt.grid(True)
plt.show()


