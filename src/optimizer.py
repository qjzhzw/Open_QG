#!/usr/bin/env python
# encoding: utf-8
'''
Optimizer类:
(1)模型所使用的优化器,其中学习率是动态变化的
'''
__author__ = 'qjzhzw'

import numpy as np
import torch


class Optimizer():
    def __init__(self, params, model):
        '''
        Optimizer类:
        模型所使用的优化器,其中学习率是动态变化的

        输入参数:
        params: 参数集合
        model: 当前使用模型
        '''

        # 定义Adam优化器
        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=params.learning_rate,
                                          betas=(0.9, 0.98),
                                          eps=1e-09)
        self.params = params
        self.step_num = 0

    def zero_grad(self):
        '''
        作用:
        重写优化器的zero_grad方法(和原版一样)
        '''
        
        self.optimizer.zero_grad()

    def step(self):
        '''
        作用:
        重写优化器的step方法(在原版step的基础上,增加了更新学习率的功能)
        '''

        self.update_learning_rate()
        self.optimizer.step()

    def update_learning_rate(self):
        '''
        作用:
        更新学习率
        '''

        # 更新学习率的公式,和transformer原论文保持一致
        self.step_num += 1
        learning_rate = np.power(self.params.d_model, -0.5) * \
                             min(np.power(self.step_num, -0.5), self.step_num * np.power(self.params.warmup_steps, -1.5))
        
        # 修改优化器中的学习率
        for param_groups in self.optimizer.param_groups:
            param_groups['lr'] = learning_rate
