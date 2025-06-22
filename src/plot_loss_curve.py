import re
import matplotlib.pyplot as plt
import os

# 修改字体配置为Windows系统常用中文字体
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 训练输出文件路径
file_path = r"c:\Users\User\PycharmProjects\deepLearning\output\训练输出.txt"

# 提取训练损失和验证损失的正则表达式
pattern = r'\[Epoch结果\] 训练损失: (\d+\.\d+) \| 验证损失: (\d+\.\d+)'

epochs = []
train_losses = []
val_losses = []

# 读取并解析文件
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()
    matches = re.findall(pattern, content)
    
    # 提取epoch数据
    for i, (train_loss, val_loss) in enumerate(matches, 1):
        epochs.append(i)
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

# 创建输出目录
output_dir = os.path.dirname(file_path)

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, 'b-', linewidth=2, label='训练损失')
plt.plot(epochs, val_losses, 'r--', linewidth=2, label='验证损失')
plt.title('训练过程中的损失变化', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('损失值', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# 保存图像
save_path = os.path.join(output_dir, 'loss_curve.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"训练曲线已保存至: {save_path}")