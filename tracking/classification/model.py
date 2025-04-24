import torch
import torch.nn as nn
import torch.nn.functional as F

class BlobCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(BlobCNN, self).__init__()

        # === Bloque 1: convolución inicial ===
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # normalización por batch para 16 canales
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # reduce tamaño a la mitad

        # === Bloque 2 ===
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # === Bloque 3 ===
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # === Capa fully-connected ===
        # Tamaño tras 3 bloques con pooling: 256 -> 128 -> 64 -> 32
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # capa oculta intermedia
        self.dropout = nn.Dropout(p=0.3) # Dropout para evitar overfitting
        self.fc2 = nn.Linear(128, num_classes)  # capa de salida con 3 clases

    def forward(self, x):
        # Bloque 1: convolución + batchnorm + ReLU + pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Bloque 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Bloque 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten para pasar a capas lineales
        x = x.view(x.size(0), -1)

        # Fully connected + ReLU
        x = F.relu(self.fc1(x))

        # Dropout
        x = self.dropout(x)

        # Capa final (sin softmax , ya que la funcion de crossentropyloss la aplica internamente)
        x = self.fc2(x)

        return x
