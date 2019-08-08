#!/bin/bash

install_requirements(){
    pip install -r requirements.txt
}

load_dataset(){
    python src/load_dataset.py
}

preprocess(){
    python src/preprocess.py
}

train(){
    python src/train.py
}

test(){
    python src/test.py
}

evaluate(){
    python2.7 evaluate/eval.py
}

demo(){
    python src/demo.py
}

if [ ! -n "$1" ] || [ ! -n "$2" ]; then
    echo "
    代码运行方式:
    sh pipeline.sh [指定运行模式] [指定GPU]

    指定运行模式包括:
    0: 安装依赖包
    1: 加载数据集
    2: 数据预处理
    3: 模型训练
    4: 模型测试
    5: 模型评估
    6: 测试demo
    可以同时指定多个运行模式,例如希望顺序执行 '3: 模型训练' 和 '4: 模型测试' 阶段,则命令为:  
    `./pipeline.sh 34 [指定GPU]`  

    指定GPU:
    使用第几块GPU，若当前无可用GPU会自动转换为CPU模式
    "
else
    export CUDA_VISIBLE_DEVICES=$2
    if [[ $(echo $1 | grep "0") != "" ]]; then
        echo ""
        echo "指定运行模式为'安装依赖包'"
        echo "指定的GPU为$2"
        install_requirements
        echo ""
    fi
    if [[ $(echo $1 | grep "1") != "" ]]; then
        echo ""
        echo "指定运行模式为'加载数据集'"
        echo "指定的GPU为$2"
        load_dataset
        echo ""
    fi
    if [[ $(echo $1 | grep "2") != "" ]]; then
        echo ""
        echo "指定运行模式为'数据预处理'"
        echo "指定的GPU为$2"
        preprocess
        echo ""
    fi
    if [[ $(echo $1 | grep "3") != "" ]]; then
        echo ""
        echo "指定运行模式为'模型训练'"
        echo "指定的GPU为$2"
        train
        echo ""
    fi
    if [[ $(echo $1 | grep "4") != "" ]]; then
        echo ""
        echo "指定运行模式为'模型测试'"
        echo "指定的GPU为$2"
        test
        echo ""
    fi
    if [[ $(echo $1 | grep "5") != "" ]]; then
        echo ""
        echo "指定运行模式为'模型评估'"
        echo "指定的GPU为$2"
        evaluate
        echo ""
    fi
    if [[ $(echo $1 | grep "6") != "" ]]; then
        echo ""
        echo "指定运行模式为'测试demo'"
        echo "指定的GPU为$2"
        demo
        echo ""
    fi
    if [[ $(echo $1 | grep "7") != "" ]]; then
        echo ""
        echo "指定运行模式为'交替测试'"
        echo "指定的GPU为$2"
        for i in {1..5}
        do
            train
            test
            evaluate
        done
        echo ""
    fi
fi
