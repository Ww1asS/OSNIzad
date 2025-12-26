import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import pandas as pd
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.int64).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.int64).to(device)

class SimpleClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layers_1 = nn.Linear(in_features, 120)
        self.layers_2 = nn.Linear(120, 10)
        self.layers_3 = nn.Linear(10, out_features)

    def forward(self, x):
        x = self.layers_3(self.layers_2(self.layers_1(x)))
        return x

in_features = X_train.shape[1]
num_classes = len(set(y))

model = SimpleClassifier(in_features, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epoch = 1000

train_losses = []
test_losses = []

for epoch in range(num_epoch):
    model.train()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    _, predicted_labels = torch.max(outputs, 1)
    correct_predictions = (predicted_labels == y_train_tensor).sum().item()
    total_samples = len(y_train_tensor)
    acc = correct_predictions / total_samples

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # Добавлено: сбор test_loss
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        test_losses.append(test_loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}, Accuracy: {acc:.4f}')

model.eval()
with torch.inference_mode():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = accuracy_score(y_test, predicted.cpu().numpy())
    print('\n', f'Accuracy: {accuracy:.4f}')

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Ошибка (обучение)", linewidth=2, color = 'blue')
plt.plot(test_losses, label="Ошибка (проверка)", linewidth=2, color = 'red')
plt.xlabel("Эпоха")
plt.ylabel("Значение ошибки")
plt.title("Изменение функции ошибки")
plt.legend()
plt.grid(True)
plt.show()
