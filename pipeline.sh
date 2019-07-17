load_dataset(){
    python load_dataset.py \
    --main_data_dir data \
    --data_dir squad
}

preprocess(){
    python preprocess.py \
    --main_data_dir data \
    --data_dir squad \
    --train_input train/sentence.txt \
    --train_output train/question.txt \
    --train_answer_start train/answer_start.txt \
    --train_answer_end train/answer_end.txt \
    --dev_input dev/sentence.txt \
    --dev_output dev/question.txt \
    --dev_answer_start dev/answer_start.txt \
    --dev_answer_end dev/answer_end.txt \
    --vocab_dir vocab.txt \
    --output_dir output.pt
}

train(){
    python train.py \
    --main_data_dir data \
    --data_dir squad \
    --input_dir output.pt \
    --cuda True \
    --num_workers 0 \
    --batch_size 32 \
    --learning_rate 0.1 \
    --num_epochs 10 \
    --embedding_size 512
}

if [ ! -n "$1" ] || [ ! -n "$2" ]; then
    echo "请输入指定的参数 1.运行模式 2.指定GPU"
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