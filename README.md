# 开源QG系统(Question Generation,问题生成)
### 系统基于Pytorch编写,使用结构为Transformer
### 正在开发，敬请期待

## 代码运行方式：
`sh pipeline.sh [指定运行模式] [指定GPU]`

### 指定运行模式包括：
0: 加载数据集（基本完成）  
1: 数据预处理（基本完成）  
2: 模型训练（正在进行）  
3: 模型测试（尚未完成）  
4: 模型评估（基本完成）  

可以同时指定多个运行模式,例如希望顺序执行"训练"和"测试"阶段,则命令为:  
`sh pipeline.sh 23 [指定GPU]`  

### 指定GPU：
使用第几块GPU，若当前无可用GPU会自动转换为CPU模式

## 文件结构介绍：
#### 文件夹:
/data : 存储原始数据(数据较大无法上传)  
/checkpoint : 存储训练好的模型参数(需要时会自动创建)  
/output : 存储预测文件(需要时会自动创建)  
/evaluate : 评价结果的脚本,包含BLEU/METEOR/ROUGH-L(由Du et al. 2017提供)  
#### python文件:
load_dataset.py : 读取原始数据构造成文本形式  
preprocess.py : 数据预处理  
train.py : 模型训练+验证  
vocab.py : Vocab类(构造任务所使用词表)  
dataset.py : Dataset类(将数据构造成batch形式)  
model.py : Model类(Transformer模型)  
#### shell脚本:
pipeline.sh 运行脚本  
