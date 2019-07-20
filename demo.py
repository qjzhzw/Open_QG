### 测试demo

import argparse
import logging

logger = logging.getLogger()


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
    params = parser.parse_args()

    input_sentence = '<cls> he is playing on the background . <sep> on the background <sep>'
    logger.info(input_sentence)
