# Подключение необходимых библиотек
import os
import random
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Указание устройства (CPU или GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# Настройка путей к данным
images_dir = 'D://blohina/Task08_HepaticVessel/imagesTr'
labels_dir = 'D://blohina/Task08_HepaticVessel/labelsTr'

# Проверка наличия данных
try:
    images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)
                     if f.endswith('.nii.gz') and not f.startswith('._') and not f.startswith('.')])
    labels = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir)
                     if f.endswith('.nii.gz') and not f.startswith('._') and not f.startswith('.')])

    if len(images) == 0 or len(labels) == 0:
        raise ValueError("Не найдено изображений или меток для обучения.")
    print(f"Обнаружено {len(images)} изображений и {len(labels)} меток.")
except Exception as e:
    print(f"Ошибка при проверке данных: {e}")
    exit()

# Подготовка данных
class MedicalSegmentationDataset(Dataset):
    def __init__(self, images, labels, target_size=(128, 128, 32)):
        self.images = images
        self.labels = labels
        self.target_size = target_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = nib.load(self.images[idx]).get_fdata().astype(np.float32)
            label = nib.load(self.labels[idx]).get_fdata().astype(np.int64)
            # Приведение данных к формату [1, depth, height, width]
            image = np.expand_dims(image, axis=0)
            image = torch.tensor(image, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.long)
            # Масштабирование данных до target_size
            image = F.interpolate(image.unsqueeze(0), size=self.target_size, mode='trilinear', align_corners=False).squeeze(0)
            label = F.interpolate(label.unsqueeze(0).unsqueeze(0).float(), size=self.target_size, mode='nearest').squeeze(0).long()
            # Приведение меток к диапазону [0, 1]
            label = torch.clamp(label, 0, 1)
            return image, label
        except Exception as e:
            print(f"Ошибка загрузки данных (индекс {idx}): {e}")
            raise

# Создание DataLoader
dataset = MedicalSegmentationDataset(images, labels)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Определение архитектуры NCA-CNN
class SegmentationNCA(nn.Module):
    def __init__(self):
        super(SegmentationNCA, self).__init__()
        self.initial_encoder = nn.Conv3d(1, 96, kernel_size=3, padding=1)  # Первый слой для 1 канала
        self.encoder = nn.Sequential(
            nn.Conv3d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(96, 96, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Conv3d(96, 2, kernel_size=1)

    def forward(self, x, steps=10):
        x = self.initial_encoder(x)
        for _ in range(steps):
            updated = self.encoder(x)
            x = updated + x  # Остаточное соединение
        x = self.decoder(x)
        return x

# Инициализация модели, функции потерь и оптимизатора
model = SegmentationNCA().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
model.train()
num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Проверка уникальных значений меток
        unique_labels = torch.unique(labels)
        print(f"Уникальные значения меток: {unique_labels}")
        optimizer.zero_grad()
        outputs = model(inputs, steps=10)

        outputs = outputs.view(-1, 2)
        labels = labels.view(-1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(data_loader):.4f}')

print("Обучение завершено.")

# Сегментация нового изображения
model.eval()

segmentation_idx = random.choice(range(len(images)))
image_np = nib.load(images[segmentation_idx]).get_fdata().astype(np.float32)
image_tensor = torch.tensor(np.expand_dims(image_np, axis=0), dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_tensor, steps=10)
    prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

slice_index = image_np.shape[2] // 2

cmap = ListedColormap(['black', 'green', 'red'])

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_np[:, :, slice_index], cmap='gray')
plt.title('Оригинальное изображение')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(prediction[:, :, slice_index], cmap=cmap, alpha=0.5)
plt.title('Предсказанные метки')
plt.axis('off')

plt.show()
print("Визуализация завершена.")
