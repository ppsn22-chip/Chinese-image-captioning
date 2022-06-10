# 基于中文的视障辅助系统
本项目是南京大学人工智能学院项目，作者侍昀琦，黄博。

# 文件结构
文件分为前后端两个主要部分，前端集成了Android Studio开发出的应用本体，后端是一个中文图像描述的模型，包含一个预训练好的模型，可以直接使用

# 前端程序
前端使用Android Studio集成开发，主要程序功能在MainActivity中实现，包括申请手机权限，语音转化，与后端进行交互等。

# 后端模型
后端包含模型的实现代码和一个预训练好的模型。具体信息如下所示：

## Environment
Python==3.5  
Tensorflow==1.5.0  
Keras==2.2.2  


## 数据集
vizwiz数据集
https://vizwiz.org/tasks-and-datasets/image-captioning/

AI Challenger数据集
数据来自[2017 AI Challenger](https://challenger.ai/competition/caption)  
数据集对给定的每一张图片有五句话的中文描述。数据集包含30万张图片，150万句中文描述。  
训练集：210,000 张   
验证集：30,000 张   
测试集 A：30,000 张   
测试集 B：30,000 张  
[数据集下载](https://challenger.ai/dataset/caption)，放在data目录  

## 模型结构
<div align=center><img width="600" src="https://github.com/HughChi/Image-Caption/raw/master/images/net.png"></div>

## 使用方式
### Demo
下载 [预训练模型](https://github.com/HughChi/Image-Caption/releases/download/v1.0/model.04-1.3820.hdf5) 放在models目录

```bash
$ python app.py
```
| Image | Caption |
| --- | --- |
| ![image](https://github.com/HughChi/Image-Caption/raw/master/images/0_bs_image.jpg) | Beam Search, k=1: 一个穿着潜水服的人在蔚蓝的海里潜水<br>Beam Search, k=3: 海面上有一个穿着潜水服的人在潜水<br>Beam Search, k=5: 海面上有一个穿着潜水服的人在潜水<br>Beam Search, k=7: 波涛汹涌的大海里有一个穿着潜水服的人在冲浪 |
| ![image](https://github.com/HughChi/Image-Caption/raw/master/images/1_bs_image.jpg) | Beam Search, k=1: 大厅里一群人旁边有一个穿着黑色衣服的女人在下国际象棋<br>Beam Search, k=3: 大厅里一群人的旁边有一个左手托着下巴的女人在下国际象棋<br>Beam Search, k=5: 大厅里一群人旁有一个戴着眼镜的女人在下国际象棋<br>Beam Search, k=7: 大厅里一群人旁边有一个戴着眼镜的女人在下国际象棋 |

### 数据预处理
```bash
$ python generated.py
```

### 训练
```bash
$ python backward.py
```
### 可视化训练过程
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

## 后端使用方法
将代码放置于后端var/www/html路径下，在命令行执行主程序即可
```bash
$ python main.py
```

# 程序安装及使用方法
前端已经提供了一个封装好的程序，目前只能使用南京大学校园网访问使用。
如果想要使用自己的服务器，将后端模型置于服务器根目录下var/www/html文件夹下，然后前端在MainActivity.java文件中将服务器地址改为自己的，封装为apk文件即可。
