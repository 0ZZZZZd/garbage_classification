import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import os
import json
import numpy as np

# ========== 超参数 ========== #
num_classes = 4  # 分类类别数（harmful, kitchen, other, recyclable）
device = torch.device("cpu")  # 使用 GPU 或 CPU

# 加载预训练的 EfficientNet-B4 模型
def load_model(model_path):
    # 1. 创建基础模型结构
    model = models.efficientnet_b4(weights=None)  # 使用新版本torchvision的参数名称

    # 2. 修改分类头（必须与训练时的结构完全一致）
    in_features = model.classifier[1].in_features  # 获取原始分类头的输入维度
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    # 3. 加载训练好的权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到：{model_path}")

    # 添加 weights_only=True 参数解决安全警告
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    model = model.to(device)
    model.eval()
    return model

# 图片预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # 添加批次维度并移动到设备
    return image

# 分类并返回结果和概率
def classify_image(model, image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        probabilities = probabilities.squeeze().cpu().numpy()  # 转换为 NumPy 数组

        # 保留小数点后7位并乘以100
        probabilities = [round(prob * 100, 5) for prob in probabilities]

    return predicted_class, probabilities

# 使用训练好的模型进行分类
def main(image_path, model_path="/root/Project/tmp/best_efficientnet_b4_3.pth"):
    model = load_model(model_path)

    # 进行图像分类
    predicted_class, probabilities = classify_image(model, image_path)

    # 分类标签
    class_labels = ['harmful', 'kitchen', 'other', 'recyclable']

    # 将概率值转换为 Python float 类型
    probabilities = [float(prob) for prob in probabilities]

    # 返回分类结果
    result = {
        'predicted_class': predicted_class,
        'probabilities': {class_labels[i]: probabilities[i] for i in range(len(class_labels))}
    }

    # 输出为 JSON 格式
    print(json.dumps(result))

if __name__ == "__main__":
    image_path = "/root/Project/tmp/uploaded.jpg"  # 替换为你的图片路径
    main(image_path)
