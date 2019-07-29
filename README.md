# 开源QG系统(Question Generation,问题生成)


## 系统基本介绍
使用Python版本 : Python3.6(其中模型评估部分脚本基本Python2.7)  
使用深度学习框架 : Pytorch0.4.1  
使用模型结构 : Transformer(详情参考论文《Attention Is All You Need》)  
使用数据集 : SQuAD(可扩展到其它QG数据集)  


## 代码运行方式：
`sh pipeline.sh [指定运行模式] [指定GPU]`  

### 指定运行模式包括：
0: 安装依赖包  
1: 加载数据集  
2: 数据预处理  
3: 模型训练  
4: 模型测试  
5: 模型评估  
6: 测试demo  

可以同时指定多个运行模式,例如希望顺序执行"训练"和"测试"阶段,则命令为:  
`sh pipeline.sh 23 [指定GPU]`  

### 指定GPU：
使用第几块GPU，若当前无可用GPU会自动转换为CPU模式  


## 数据下载(SQuAD):
#### 百度网盘:
链接: https://pan.baidu.com/s/1JKaHR5mde8GGxYRXk33rSw  
提取码: b95w  
其中包含了所有原始数据和经过步骤1/2后得到的处理过后的数据,将数据放在新建的data文件夹中,即data/squad/  
(其中原始数据由Song et al., 2018提供)  


## 文件结构介绍：
#### 文件夹:
/data : 存储原始数据  
/checkpoint : 存储训练好的模型参数(需要时会自动创建)  
/output : 存储预测文件(需要时会自动创建)  
/evaluate : 评价结果的脚本,包含BLEU/METEOR/ROUGH-L(由Du et al., 2017提供)  
/image : 存储README.md中所用到的图片  
#### python文件:
load_dataset.py : 读取原始数据构造成文本形式(只针对SQuAD数据集)  
preprocess.py : 数据预处理  
train.py : 模型训练+验证  
test.py : 模型测试  
demo.py : demo测试  
vocab.py : Vocab类(构造任务所使用词表)  
dataset.py : Dataset类(将数据构造成batch形式)  
model.py : Model类(Transformer模型)  
optimizer.py : Optimizer类(训练模型所使用优化器)  
beam.py : Beam类(模型测试时所使用beam search)  
#### shell脚本:
pipeline.sh 运行脚本  
#### 其它文件:
requirements.txt : 需要安装的依赖包及对应版本  


## 实验结果:
BLEU-1: 27.28  
BLEU-2: 11.35  
BLEU-3: 6.01  
BLEU-4: 3.53  
METEOR: 8.36  
ROUGH-L: 26.84  


## 测试demo:
<img src="https://raw.githubusercontent.com/qjzhzw/Open_QG/master/image/demo.png" width=100)  
