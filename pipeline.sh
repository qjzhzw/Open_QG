load_dataset(){
    python load_dataset.py
}

preprocess(){
    python preprocess.py
}

train(){
    python train.py
}

if [ ! -n "$1" ] || [ ! -n "$2" ]; then
    echo "
    请输入指定的参数 1.运行模式 2.指定GPU

    指定运行模式包括：
    0: 加载数据集（基本完成）
    1: 数据预处理（基本完成）
    2: 模型训练（正在进行）
    3: 模型测试（尚未完成）
    4: 模型评估（尚未完成）
    
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
fi