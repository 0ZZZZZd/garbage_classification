import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B4_Weights
import torchvision.transforms.autoaugment as autoaugment
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# ========== 超参数 ==========
batch_size = 128
num_epochs = 100
learning_rate = 0.0005
num_classes = 4  # 分类类别数（harmful, kitchen, other, recyclable）

# 检测设备（优先使用GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 数据预处理 ==========改进
transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),  # 50% 概率水平翻转
    transforms.RandomRotation(15),  # 随机旋转 ±15 度
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 颜色扰动
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # 随机平移
    autoaugment.TrivialAugmentWide(),  # 使用AutoAugment策略增强数据
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

transform_val = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ========== 数据加载 ==========
train_dir = 'dataset/train'
val_dir = 'dataset/val'

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

# ========== 使用更强的预训练模型 EfficientNet-B4 ==========改进
print("使用预训练的 EfficientNet-B4")

# ========= 加载预训练模型 ================#
model = models.efficientnet_b4(pretrained=False)  # 禁用自动下载
model.load_state_dict(torch.load("efficientnet_b4.pth"))  # 加载本地文件

# 修改全连接层（适配垃圾分类任务）
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)
model = model.to(device)

# ========== 损失函数和优化器 ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# ========== 训练函数 ==========
# ========== 训练函数 ==========
def train():
    print("开始训练模型")
    best_acc = 0.0  # 记录最佳验证集准确率
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # 计算验证集 loss 和准确率
        val_loss, val_acc = validate()

        # 在每一轮打印训练和验证集的情况
        print(f'Epoch [{epoch + 1}/{num_epochs}] '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%} | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_efficientnet_b4_3.pth')
            print("==> 发现更好的模型，已保存！")


# ========== 验证函数 ==========
def validate():
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = correct / total
    return val_loss, val_acc


# 运行训练
if __name__ == '__main__':
    train()