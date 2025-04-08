import os
from PIL import Image
import pandas as pd

def check_dataset_structure(dataset_root):
    """检测数据集结构并统计各分类的图片数量"""
    stats = {"类别": [], "训练集数量": [], "验证集数量": []}
    categories = ["harmful", "other", "kitchen", "recyclable"]

    # 遍历所有分类
    for category in categories:
        train_dir = os.path.join(dataset_root, "train", category)
        val_dir = os.path.join(dataset_root, "val", category)

        # 统计训练集
        train_count = len([f for f in os.listdir(train_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(train_dir) else 0
        # 统计验证集
        val_count = len([f for f in os.listdir(val_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(val_dir) else 0

        stats["类别"].append(category)
        stats["训练集数量"].append(train_count)
        stats["验证集数量"].append(val_count)

    # 生成统计表格
    df = pd.DataFrame(stats)
    df.loc["总计"] = ["-", df["训练集数量"].sum(), df["验证集数量"].sum()]
    print("\n数据集结构统计:")
    print(df.to_markdown(index=False))

def check_and_clean_dataset(dataset_root):
    """检查并清理损坏文件，按类别统计"""
    categories = ["harmful", "other", "kitchen", "recyclable"]
    corrupted_files = []
    stats = []

    # 遍历训练集和验证集
    for split in ["train", "val"]:
        for category in categories:
            folder = os.path.join(dataset_root, split, category)
            if not os.path.exists(folder):
                continue

            total = 0
            valid = 0
            # 检查每个文件
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                total += 1
                try:
                    with Image.open(filepath) as img:
                        img.verify()  # 验证图片完整性
                        img.close()
                    valid += 1
                except Exception as e:
                    corrupted_files.append(filepath)
                    print(f"损坏文件: {filepath} | 错误: {str(e)}")

            # 记录统计信息
            stats.append({
                "数据集": split,
                "类别": category,
                "总图片": total,
                "有效图片": valid,
                "损坏图片": total - valid
            })

    # 打印统计报告
    print("\n损坏检测报告:")
    report_df = pd.DataFrame(stats)
    print(report_df.to_markdown(index=False))

    # 处理损坏文件
    if corrupted_files:
        print("\n发现损坏文件列表:")
        for file in corrupted_files:
            print(f" - {file}")

        choice = input("\n是否要删除所有损坏文件？(y/n): ").lower()
        if choice == "y":
            for file in corrupted_files:
                os.remove(file)
                print(f"已删除: {file}")
            print(f"已清理 {len(corrupted_files)} 个损坏文件")
        else:
            print("未执行删除操作")
    else:jav
        print("\n未发现损坏文件")

if __name__ == "__main__":
    dataset_root = "dataset"  # 修改为你的数据集根目录路径
    check_dataset_structure(dataset_root)
    check_and_clean_dataset(dataset_root)