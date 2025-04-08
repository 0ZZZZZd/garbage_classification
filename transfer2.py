import os
import shutil

# 获取当前目录
current_dir = os.getcwd()

# 创建目标文件夹
kitchen_dir = os.path.join(current_dir, "kitchen")
recyclable_dir = os.path.join(current_dir, "recyclable")
other_dir = os.path.join(current_dir, "other")
harmful_dir = os.path.join(current_dir, "harmful")

# 创建目标文件夹（如果不存在）
os.makedirs(kitchen_dir, exist_ok=True)
os.makedirs(recyclable_dir, exist_ok=True)
os.makedirs(other_dir, exist_ok=True)
os.makedirs(harmful_dir, exist_ok=True)

# 定义文件夹范围
kitchen_range = range(0, 52)  # 0到51
recyclable_range = range(52, 201)  # 52到200
other_range = range(201, 251)  # 201到250
harmful_range = range(251, 265)  # 251到264

# 遍历0到264的文件夹
for folder_num in range(265):
    folder_path = os.path.join(current_dir, str(folder_num))

    if os.path.isdir(folder_path):
        # 获取当前文件夹中的所有图片文件
        for filename in os.listdir(folder_path):
            # 只选择JPG格式的图片文件
            if filename.lower().endswith('.jpg'):
                source_path = os.path.join(folder_path, filename)

                # 根据文件夹编号决定移动到哪个文件夹
                if folder_num in kitchen_range:
                    destination_path = os.path.join(kitchen_dir, filename)
                elif folder_num in recyclable_range:
                    destination_path = os.path.join(recyclable_dir, filename)
                elif folder_num in other_range:
                    destination_path = os.path.join(other_dir, filename)
                elif folder_num in harmful_range:
                    destination_path = os.path.join(harmful_dir, filename)

                # 移动文件
                shutil.move(source_path, destination_path)
                print(f"Moved {filename} to {destination_path}")
