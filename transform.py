import torch
import torchvision.models as models

# 设备设置（使用 CPU 进行转换，以兼容 Android）
device = torch.device("cpu")

# 加载模型（必须与训练时结构一致）
model = models.efficientnet_b4(weights=None)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(model.classifier[1].in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, 4)  # 4 类垃圾分类
)

# 加载训练好的权重
model.load_state_dict(torch.load("best_efficientnet_b4_1.pth", map_location=device))
model.to(device)
model.eval()

# 随机生成一个输入张量（与训练时图像尺寸一致）
example_input = torch.rand(1, 3, 128, 128).to(device)

# 使用 torch.jit.trace 进行转换（更稳定）
traced_script_module = torch.jit.trace(model, example_input)

# 保存 TorchScript 模型
traced_script_module.save("model_android.pt")

print("TorchScript 模型已保存为 model_android.pt，可用于 Android 部署！")
