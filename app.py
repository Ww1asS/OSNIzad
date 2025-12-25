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

iris = load_iris() # Готовый датасет
X = iris.data # Конкретные данные
y = iris.target # Наша цель

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train,dtype=torch.float).to(device)
y_train_tensor = torch.tensor(y_train,dtype=torch.int64).to(device)
X_test_tensor = torch.tensor(X_test,dtype=torch.float).to(device)
y_test_tensor = torch.tensor(y_test,dtype=torch.int64).to(device)

class SimpleClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layers_1 = nn.Linear(in_features, 120)
        self.layers_2 = nn.Linear(120, 10)
        self.layers_3 = nn.Linear(10, out_features)

    def forward(self,x):
        x = self.layers_3(self.layers_2(self.layers_3(x)))
        return x

in_features = X_train.shape[1]
num_classes =  len(set(y))

model = SimpleClassifier(in_features, num_classes).to(device)




