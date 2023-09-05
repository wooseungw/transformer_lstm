import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torchvision.models import vit_b_16

# 데이터 준비
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 사전 학습된 ViT 모델 불러오기
model = vit_b_16(pretrained=True)

# 모델의 마지막 분류 레이어 수정
model.heads = nn.Linear(model.heads.in_features, 10)  # CIFAR10은 10개의 클래스를 가지므로

# 손실 함수 및 최적화기 설정
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습
for epoch in range(10):  # 10 에포크 동안 학습
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
