### 将SQuAD原始数据转换为transformer模型可用数据

import argparse
import logging
import json
import os

logger = logging.getLogger()


def load_dataset(origin_file, sentence_file, question_file, answer_start_file, answer_end_file):
    '''
    origin_file: 输入的原始文件
    sentence_file: 输出的句子文件
    question_file: 输出的问题文件
    answer_start_file: 输出的答案开始位置文件
    answer_end_file: 输出的答案结束位置文件
    '''

    logger.info('正在从{}中加载数据'.format(origin_file))

    # 将原始数据加载进来
    instances = json.loads(open(origin_file, 'r').read())

    # total和num用于判断有多少数据被成功处理
    total = len(instances)
    num = 0

    # 所有输出文件
    f_sentence = open(sentence_file, 'w')
    f_question = open(question_file, 'w')
    f_answer_start = open(answer_start_file, 'w')
    f_answer_end = open(answer_end_file, 'w')

    # 依次处理所有数据
    for instance in instances:
        # 加载sentence/question/answer
        sentence = instance['annotation1']['toks'].strip().lower()
        question = instance['annotation2']['toks'].strip().lower()
        answer = instance['annotation3']['toks'].strip().lower()

        # 将str切分为list
        sentence = sentence.split()
        question = question.split()
        answer = answer.split()

        # 找到答案起止位置
        answer_start = None
        answer_end = None
        for index in range(len(sentence)):
            if answer == sentence[index : index + len(answer)]:
                answer_start = index
                answer_end = index + len(answer)
                break

        # 将sentence和answer连接起来(类似于bert中的做法)
        # <cls> sentence <sep> answer <sep>
        sentence.insert(0, '<cls>')
        sentence.append('<sep>')
        sentence.extend(answer)
        sentence.append('<sep>')

        # 将处理好的数据存入本地文件
        if answer_start != None and answer_end != None:
            f_sentence.write(' '.join(sentence) + '\n')
            f_question.write(' '.join(question) + '\n')
            f_answer_start.write(str(answer_start) + '\n')
            f_answer_end.write(str(answer_end) + '\n')
            num += 1

    # 关闭所有文件
    f_sentence.close()
    f_question.close()
    f_answer_start.close()
    f_answer_end.close()

    logger.info('原始数据{}, 成功加载数据{}'.format(total, num))


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
    parser.add_argument('--dataset_dir', type=str, default='squad', help='任务所使用的数据集所在的子目录')
    params = parser.parse_args()

    # 判断子目录train/dev/test是否存在，若不存在则创建
    params.train_dir = os.path.join(params.main_data_dir, params.dataset_dir, 'train')
    params.dev_dir = os.path.join(params.main_data_dir, params.dataset_dir, 'dev')
    params.test_dir = os.path.join(params.main_data_dir, params.dataset_dir, 'test')
    if not os.path.exists(params.train_dir):
        os.makedirs(params.train_dir)
    if not os.path.exists(params.dev_dir):
        os.makedirs(params.dev_dir)
    if not os.path.exists(params.test_dir):
        os.makedirs(params.test_dir)

    # 打印参数列表
    logger.info('参数列表:{}'.format(params))

    # 加载数据集
    load_dataset(os.path.join(params.main_data_dir, params.dataset_dir, 'origin/train_sent_pre.json'),
                os.path.join(params.main_data_dir, params.dataset_dir, 'train/sentence.txt'),
                os.path.join(params.main_data_dir, params.dataset_dir, 'train/question.txt'),
                os.path.join(params.main_data_dir, params.dataset_dir, 'train/answer_start.txt'),
                os.path.join(params.main_data_dir, params.dataset_dir, 'train/answer_end.txt'))
    load_dataset(os.path.join(params.main_data_dir, params.dataset_dir, 'origin/dev_sent_pre.json'),
                os.path.join(params.main_data_dir, params.dataset_dir, 'dev/sentence.txt'),
                os.path.join(params.main_data_dir, params.dataset_dir, 'dev/question.txt'),
                os.path.join(params.main_data_dir, params.dataset_dir, 'dev/answer_start.txt'),
                os.path.join(params.main_data_dir, params.dataset_dir, 'dev/answer_end.txt'))
    load_dataset(os.path.join(params.main_data_dir, params.dataset_dir, 'origin/test_sent_pre.json'),
                os.path.join(params.main_data_dir, params.dataset_dir, 'test/sentence.txt'),
                os.path.join(params.main_data_dir, params.dataset_dir, 'test/question.txt'),
                os.path.join(params.main_data_dir, params.dataset_dir, 'test/answer_start.txt'),
                os.path.join(params.main_data_dir, params.dataset_dir, 'test/answer_end.txt'))
