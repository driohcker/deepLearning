import os
import glob
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torchvision.models import ResNet18_Weights, ResNet34_Weights

# 添加设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[设备信息] 使用{'GPU' if torch.cuda.is_available() else 'CPU'}进行训练")

# 新增CUDA优化配置
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True  # 加速卷积运算
    print(f"[CUDA信息] 设备: {torch.cuda.get_device_name(0)}, 内存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()

# 设置随机数种子，保证结果可复现
torch.manual_seed(0)
if use_cuda:
    torch.cuda.manual_seed(0)


def parse_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # 原始SVHN中类别10为数字0
        lbl = np.array(self.img_label[index], dtype=int)
        # 将标签填充到固定长度5，不足的用10(代表无字符)填充
        pad_width = max(0, 5 - len(lbl))
        lbl = np.pad(lbl, (0, pad_width), mode='constant', constant_values=10)[:5]
        lbl = list(lbl)
        return img, torch.from_numpy(np.array(lbl))

    def __len__(self):
        return len(self.img_path)


# 读取图像路径
train_path = glob.glob('../input/train/*.png')
train_path.sort()
train_json = parse_json('../input/train.json')
train_label = [train_json[x]['label'] for x in train_json]
print(f"[数据加载] 训练集图像数量: {len(train_path)}, 标签数量: {len(train_label)}")

# 读取验证集图像路径
val_path = glob.glob('../input/val/*.png')
val_path.sort()
val_json = parse_json('../input/val.json')
val_label = [val_json[x]['label'] for x in val_json]
print(f"[数据加载] 验证集图像数量: {len(val_path)}, 标签数量: {len(val_label)}")

# 定义数据转换
train_transform = transforms.Compose([
    transforms.Resize((68, 128)),
    transforms.RandomCrop((64, 128)),
    transforms.ColorJitter(0.3, 0.3, 0.2),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((64, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 创建数据加载器
train_dataset = SVHNDataset(train_path, train_label, train_transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=40, shuffle=True, num_workers=0,
    pin_memory=True if use_cuda else False  # 新增GPU内存固定
)

val_dataset = SVHNDataset(val_path, val_label, val_transform)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=40, shuffle=False, num_workers=0,
    pin_memory=True if use_cuda else False  # 新增GPU内存固定
)


class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()
        model_conv = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv

        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
        self.fc5 = nn.Linear(512, 11)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        return c1, c2, c3, c4, c5


class SVHN_Model2(nn.Module):
    def __init__(self):
        super(SVHN_Model2, self).__init__()
        model_conv = models.resnet34(pretrained=True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv

        self.hd_fc1 = nn.Linear(512, 128)
        self.hd_fc2 = nn.Linear(512, 128)
        self.hd_fc3 = nn.Linear(512, 128)
        self.hd_fc4 = nn.Linear(512, 128)
        self.hd_fc5 = nn.Linear(512, 128)

        self.dropout_1 = nn.Dropout(0.25)
        self.dropout_2 = nn.Dropout(0.25)
        self.dropout_3 = nn.Dropout(0.25)
        self.dropout_4 = nn.Dropout(0.25)
        self.dropout_5 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128, 11)
        self.fc2 = nn.Linear(128, 11)
        self.fc3 = nn.Linear(128, 11)
        self.fc4 = nn.Linear(128, 11)
        self.fc5 = nn.Linear(128, 11)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)

        feat1 = self.hd_fc1(feat)
        feat2 = self.hd_fc2(feat)
        feat3 = self.hd_fc3(feat)
        feat4 = self.hd_fc4(feat)
        feat5 = self.hd_fc5(feat)
        feat1 = self.dropout_1(feat1)
        feat2 = self.dropout_2(feat2)
        feat3 = self.dropout_3(feat3)
        feat4 = self.dropout_4(feat4)
        feat5 = self.dropout_5(feat5)

        c1 = self.fc1(feat1)
        c2 = self.fc2(feat2)
        c3 = self.fc3(feat3)
        c4 = self.fc4(feat4)
        c5 = self.fc5(feat5)

        return c1, c2, c3, c4, c5


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    train_loss = []
    total_batches = len(train_loader)

    train_pbar = tqdm(enumerate(train_loader), total=total_batches, desc=f"Epoch {epoch + 1}/20 [Train]")
    for i, (input, target) in train_pbar:
        # 使用统一的device变量
        input = input.to(device)
        target = target.to(device)

        c0, c1, c2, c3, c4 = model(input)
        loss = criterion(c0, target[:, 0].long()) + criterion(c1, target[:, 1].long()) + \
               criterion(c2, target[:, 2].long()) + criterion(c3, target[:, 3].long()) + \
               criterion(c4, target[:, 4].long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        # 在进度条上显示当前批次损失
        train_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    return np.mean(train_loss)


def validate(val_loader, model, criterion, epoch):  # 添加epoch参数
    model.eval()
    val_loss = []
    total_batches = len(val_loader)

    # 添加验证进度条
    val_pbar = tqdm(enumerate(val_loader), total=total_batches, desc=f"Epoch {epoch + 1}/20 [Val]")
    with torch.no_grad():
        for i, (input, target) in val_pbar:
            if use_cuda:
                input = input.cuda()
                target = target.cuda()

            c0, c1, c2, c3, c4 = model(input)
            loss = criterion(c0, target[:, 0].long()) + \
                   criterion(c1, target[:, 1].long()) + \
                   criterion(c2, target[:, 2].long()) + \
                   criterion(c3, target[:, 3].long()) + \
                   criterion(c4, target[:, 4].long())
            val_loss.append(loss.item())
            # 在进度条上显示当前批次损失
            val_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    return np.mean(val_loss)


def predict(test_loader, model, tta=10):
    model.eval()
    test_pred_tta = None
    total_batches = len(test_loader)
    total_samples = len(test_loader.dataset)

    # 仅保留TTA总进度条，移除批次级进度条
    tta_pbar = tqdm(range(tta), desc="TTA增强轮次", total=tta)
    for tta_round in tta_pbar:
        test_pred = []
        processed = 0

        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                if use_cuda:
                    input = input.cuda()

                c0, c1, c2, c3, c4 = model(input)
                if use_cuda:
                    output = np.concatenate([
                        c0.data.cpu().numpy(),
                        c1.data.cpu().numpy(),
                        c2.data.cpu().numpy(),
                        c3.data.cpu().numpy(),
                        c4.data.cpu().numpy()], axis=1)
                else:
                    output = np.concatenate([
                        c0.data.numpy(),
                        c1.data.numpy(),
                        c2.data.numpy(),
                        c3.data.numpy(),
                        c4.data.numpy()], axis=1)

                test_pred.append(output)
                processed += input.size(0)
                # 仅在TTA总进度条显示处理进度
                tta_pbar.set_postfix({
                    f"TTA-{tta_round+1}": f"{processed}/{total_samples}样本",
                    "进度": f"{i+1}/{total_batches}批次"
                })

        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta


model = SVHN_Model1().to(device)  # 直接在初始化时移动到设备
criterion = nn.CrossEntropyLoss().to(device)  # 损失函数也移动到GPU
best_loss = 1000.0

if use_cuda:
    model = model.cuda()

for epoch in range(20):
    print(f"===== 开始第 {epoch + 1} / 20 个Epoch ===== ")
    if epoch < 10:
        optimizer = torch.optim.Adam(model.parameters(), 0.001)
        print("[优化器] 使用学习率: 0.001")
    else:
        optimizer = torch.optim.Adam(model.parameters(), 0.0001)
        print("[优化器] 使用学习率: 0.0001")

    train_loss = train(train_loader, model, criterion, optimizer, epoch)
    val_loss = validate(val_loader, model, criterion, epoch)  # 传入epoch参数

    val_label = [''.join(map(str, x)) for x in val_loader.dataset.img_label]
    val_predict_label = predict(val_loader, model, 1)
    val_predict_label = np.vstack([
        val_predict_label[:, :11].argmax(1),
        val_predict_label[:, 11:22].argmax(1),
        val_predict_label[:, 22:33].argmax(1),
        val_predict_label[:, 33:44].argmax(1),
        val_predict_label[:, 44:55].argmax(1),
    ]).T
    val_label_pred = []
    for x in val_predict_label:
        val_label_pred.append(''.join(map(str, x[x != 10])))

    val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))

    print(f'[Epoch结果] 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f} | 验证集字符准确率: {val_char_acc:.4f}')
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), './model.pt')
        print(f"[模型保存] √ 保存最佳模型，当前最佳验证损失: {best_loss:.4f}")

# 预测
test_path = glob.glob('../input/test_a/*.png')
test_path.sort()
test_label = [[1]] * len(test_path)
print(f"[预测准备] 测试集图像数量: {len(test_path)}, 标签数量: {len(test_label)}")

test_loader = torch.utils.data.DataLoader(
    SVHNDataset(test_path, test_label,
                transforms.Compose([
                    transforms.Resize((68, 128)),
                    transforms.RandomCrop((64, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
    batch_size=40,
    shuffle=False,
    num_workers=0
)

# 加载保存的最优模型
model.load_state_dict(torch.load('model.pt', map_location=device))  # 新增map_location确保设备兼容
model = model.to(device)

test_predict_label = predict(test_loader, model, 1)
print(test_predict_label.shape)

test_label = [''.join(map(str, x)) for x in test_loader.dataset.img_label]
test_predict_label = np.vstack([
    test_predict_label[:, :11].argmax(1),
    test_predict_label[:, 11:22].argmax(1),
    test_predict_label[:, 22:33].argmax(1),
    test_predict_label[:, 33:44].argmax(1),
    test_predict_label[:, 44:55].argmax(1),
]).T

test_label_pred = []
for x in test_predict_label:
    test_label_pred.append(''.join(map(str, x[x != 10])))

import pandas as pd

df_submit = pd.read_csv('../input/test_A_sample_submit_A.csv')
df_submit['file_code'] = test_label_pred
df_submit.to_csv('submit.csv', index=None)