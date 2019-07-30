#!/usr/bin/env python
# encoding: utf-8
'''
logger方法:
(1)用来输出日志信息,每个模块都会用到
'''
__author__ = 'qjzhzw'

import logging


def logger():
    '''
    return logger: 日志信息的输出器
                   用法: logger.info(XXX)
    '''

    # logger的一些设置
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s:  %(message)s ', '%Y/%m/%d %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    return logger
