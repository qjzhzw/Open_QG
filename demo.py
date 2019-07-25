### 测试demo

import argparse
import logging
import os
import torch

from vocab import Vocab
from model import Model

logger = logging.getLogger()


def test_demo(vocab, params, input_sentence):
    '''
    vocab: 从pt文件中读取的该任务所使用的vocab
    params: 参数集合
    input_sentence: 输入句子
    '''

    # 定义模型
    model = Model(params, vocab)

    # 如果参数中设置了使用cuda且当前cuda可用,则将模型放到cuda上
    flag_cuda = False
    if params.cuda and torch.cuda.is_available():
        model.cuda()
        flag_cuda = True

    # 如果参数中设置了打印模型结构,则打印模型结构
    if params.print_model:
        logger.info(model)

    # 断言: 模型参数文件存在
    assert os.path.exists(params.checkpoint_file)
    # 加载模型参数
    model_params = torch.load(params.checkpoint_file)
    model.load_state_dict(model_params)
    logger.info('正在从{}中读取已经训练好的模型参数'.format(params.checkpoint_file))


if __name__ == '__main__':

    # logger的一些设置
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s:  %(message)s ', '%Y/%m/%d %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # 加载参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_data_dir', type=str, default='data', help='数据主目录')
    parser.add_argument('--main_checkpoint_dir', type=str, default='checkpoint', help='输出的模型参数目录')
    parser.add_argument('--dataset_dir', type=str, default='squad', help='任务所使用的数据集所在的子目录')
    parser.add_argument('--temp_pt_file', type=str, default='data.pt', help='输入的pt文件位置')
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint.pt', help='输出的模型参数位置')
    parser.add_argument('--cuda', type=bool, default=True, help='是否使用cuda')
    parser.add_argument('--print_model', type=bool, default=False, help='是否打印出模型结构')
    parser.add_argument('--embedding_size', type=int, default=512, help='transformer模型超参数:embedding_size(词向量维度,在transformer模型中和d_model一致)')
    parser.add_argument('--num_layers', type=int, default=6, help='transformer模型超参数:num_layers')
    parser.add_argument('--num_heads', type=int, default=8, help='transformer模型超参数:num_heads')
    parser.add_argument('--d_model', type=int, default=512, help='transformer模型超参数:d_model')
    parser.add_argument('--d_k', type=int, default=64, help='transformer模型超参数:d_k')
    parser.add_argument('--d_v', type=int, default=64, help='transformer模型超参数:d_v')
    parser.add_argument('--d_ff', type=int, default=2048, help='transformer模型超参数:d_ff')
    parser.add_argument('--dropout', type=float, default=0.1, help='transformer模型超参数:dropout')
    params = parser.parse_args()

    params.temp_pt_file = os.path.join(params.main_data_dir, params.dataset_dir, params.temp_pt_file)
    params.checkpoint_file = os.path.join(params.main_checkpoint_dir, params.dataset_dir, params.checkpoint_file)

    # 打印参数列表
    logger.info('参数列表:{}'.format(params))

    # 从已保存的pt文件中读取vocab
    data = torch.load(params.temp_pt_file)
    vocab = data['vocab']

    # 测试demo的输入句子
    input_sentence = '<cls> he is playing on the background . <sep> on the background <sep>'
    logger.info('输入句子的文本形式为 : {}'.format(input_sentence))

    # 将输入句子转化为索引形式
    input_sentence = vocab.convert_sentence2index(input_sentence)
    logger.info('输入句子的索引形式为 : {}'.format(input_sentence))

    # 测试demo
    test_demo(vocab, params, input_sentence)
