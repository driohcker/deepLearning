# 【AI入门系列】城市探险家：街景字符识别学习赛代码实现

## 介绍
这是一个基于CRNN的街景门牌号数字自动识别项目。它能够从图像中提取文字信息，适用于如街景地图、图像检索等场景。

选题来自：https://tianchi.aliyun.com/competition/entrance/531795/introduction

## 依赖安装
使用以下命令安装依赖：
```bash
pip install -r requirements.txt
```

## 模型结构 
特征提取：ResNet18(预训练) + 分类头：5个独立FC层(输出11类字符)

## 开发环境：
1.	Windows 11
2.	PyTorch 1.10+
3.	CUDA 11.8
4.	Python 3.13

## 数据集准备
下载使用SVHN数据集，解压并放入input文件夹

数据集	SVHN： (Street View House Numbers) 街景门牌号数据集，测试集40k，训练集40k
```
file	size	link
mchar_train.zip	345.91MB	http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_train.zip
mchar_train.json	3.16MB	http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_train.json
mchar_val.zip	200.16MB	http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_val.zip
mchar_val.json	1.03MB	http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_val.json
mchar_test_a.zip	370.6MB	http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_test_a.zip
mchar_sample_submit_A.csv	507.83KB	http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_sample_submit_A.csv
```

数据集应分为训练集和测试集，并按照以下目录结构组织：
```
dataset/
    train/
        000000.png
        000001.png
        ...
    test/
        000000.png
        000001.png
        ...
```

## 使用说明
最终效果文件为 character recognition_GPU_v3.py

如果不使用gpu加速请使用 character recognition.py

## 训练成果：58.3%
