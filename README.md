# CV homework

> SAST SUMMER 2023, CV HOMEWORK

## 1. 环境配置
使用 `conda` 创建本项目对应的虚拟环境，然后运行如下命令：
```bash
pip install -r requirements.txt
```

本项目使用**AFHQ 数据集**。此数据集在论文 StarGAN v2 中首次使用，包含约 15000 张 512*512 的高质量图片，主要是猫猫狗狗等可爱动物头像。

在运行程序之前，请点击 [此清华云盘链接](https://cloud.tsinghua.edu.cn/d/a747c0d1110d451099f9/files/?p=%2Fafhq.zip&dl=1) 下载数据并解压到与 `main.py` 或 `LatentDiffusion-colab.ipynb` 同路径下。如果你使用 Google Colab 笔记本，则不必事先下载文件。

## 2. 项目运行与输出结果
运行本项目：
- 可在命令行中运行 `main.py` （在运行之前请确认激活了正确的虚拟环境）；如果在Linux服务器上**后台运行**，可以使用如下命令：
```bash
nohup python main.py &
```
- 在Google Colab上运行`LatentDiffusion-colab.ipynb`，逐行代码执行即可


代码的所有 `TODO` 部分均已完成。项目在 Zeus 服务器运行，获得的 checkpoints 已上传到 [此清华云盘链接](https://cloud.tsinghua.edu.cn/f/2a786c8e95db4ac6acf3/)。受限于硬件条件，目前尚未完成训练。

此外，本项目在执行过程中添加了简单的 loss 可视化代码，中间输出结果已上传到 [此清华云盘链接](https://cloud.tsinghua.edu.cn/f/d14bd33bd85c4d1abcd2/) 。



