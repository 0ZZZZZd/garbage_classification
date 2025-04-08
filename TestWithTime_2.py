import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_B4_Weights
from PIL import Image
import matplotlib.pyplot as plt
import time  # 导入时间模块

# 定义分类类别（必须与训练时的顺序一致）
class_names = ['harmful', 'kitchen', 'other', 'recyclable']  # 根据实际类别修改

# **加载模型**
def load_model(model_path):
    start_time = time.time()  # 记录加载模型的开始时间

    # 加载预训练的EfficientNet-B4模型
    model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

    # 修改分类层，使其匹配训练时的结构
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, len(class_names))
    )

    # 加载训练好的权重
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # 切换为推理模式

    load_time = time.time() - start_time  # 计算加载模型的时间
    print(f"模型加载时间: {load_time:.4f} 秒")
    return model

# **图像预处理（与训练时一致）**
def preprocess_image(image_path):
    start_time = time.time()  # 记录图像预处理的开始时间

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image = Image.open(image_path).convert('RGB')  # 确保图像是RGB格式
    processed_image = transform(image).unsqueeze(0)  # 增加 batch 维度

    preprocess_time = time.time() - start_time  # 计算图像预处理的时间
    print(f"图像预处理时间: {preprocess_time:.4f} 秒")
    return processed_image

# **测试主函数**
if __name__ == "__main__":
    total_start = time.time()  # 记录程序总运行开始时间

    MODEL_PATH = "best_efficientnet_b4_3.pth"  # 训练保存的模型路径
    IMAGE_PATH = "test.jpg"  # 待测试的图片路径

    device = torch.device("cpu")

    print("步骤1: 加载模型")
    model = load_model(MODEL_PATH).to(device)

    print("步骤2: 图像预处理")
    input_tensor = preprocess_image(IMAGE_PATH).to(device)

    print("步骤3: 模型推理")
    inference_start = time.time()  # 记录推理开始时间
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)  # 计算类别概率
    inference_time = time.time() - inference_start  # 计算推理时间
    print(f"推理时间: {inference_time:.4f} 秒")

    print("\n各类别概率：")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {probabilities[i].item():.4f}")

    # 获取预测类别
    predicted_class = torch.argmax(probabilities).item()
    print(f"\n预测结果: {class_names[predicted_class]} (置信度: {probabilities[predicted_class]:.4f})")

    total_time = time.time() - total_start  # 计算总运行时间
    print(f"\n总运行时间: {total_time:.4f} 秒")

    # **可视化结果（可选）**
    # image = Image.open(IMAGE_PATH)
    # plt.imshow(image)
    # plt.title(f"Predicted: {class_names[predicted_class]}")
    # plt.axis('off')
    # plt.show()
