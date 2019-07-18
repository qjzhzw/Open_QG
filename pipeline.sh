load_dataset(){
    python load_dataset.py
}

preprocess(){
    python preprocess.py
}

train(){
    python train.py --num_epochs 1
}

evaluate(){
    python2.7 evaluate/eval.py
}

if [ ! -n "$1" ] || [ ! -n "$2" ]; then
    echo "
    代码运行方式：
    sh pipeline.sh [指定运行模式] [指定GPU]

    指定运行模式包括：
    0: 加载数据集（基本完成）  
    1: 数据预处理（基本完成）  
    2: 模型训练（正在进行）  
    3: 模型测试（尚未完成）  
    4: 模型评估（基本完成）  
    可以同时指定多个运行模式,例如希望顺序执行'训练'和'测试'阶段,则命令为:  
    sh pipeline.sh 23 [指定GPU]  

    指定GPU：
    使用第几块GPU，若当前无可用GPU会自动转换为CPU模式
    "
else
    export CUDA_VISIBLE_DEVICES=$2
    if [ $(echo $1 | grep "0") != "" ]; then
        echo "指定运行模式为'加载数据集'"
        echo "指定的GPU为$2"
        load_dataset
    fi
    if [ $(echo $1 | grep "1") != "" ]; then
        echo "指定运行模式为'数据预处理'"
        echo "指定的GPU为$2"
        preprocess
    fi
    if [ $(echo $1 | grep "2") != "" ]; then
        echo "指定运行模式为'模型训练'"
        echo "指定的GPU为$2"
        train
    fi
    if [ $(echo $1 | grep "4") != "" ]; then
        echo "指定运行模式为'模型评估'"
        echo "指定的GPU为$2"
        evaluate
    fi
fi