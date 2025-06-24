基于CRNN的街景门牌号数字自动识别项目

开发环境：
1.	Windows 11
2.	PyTorch 1.10+
3.	CUDA 11.8
4.	Python 3.13

数据集	SVHN： (Street View House Numbers) 街景门牌号数据集，测试集40k，训练集40k

模型结构	特征提取：ResNet18(预训练) + 分类头：5个独立FC层(输出11类字符)

训练成果：58.3%
