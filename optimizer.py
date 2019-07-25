### 优化器

import torch
import numpy as np

class Optimizer():
    def __init__(self, model, params):
        '''
        model: 当前使用模型
        params: 参数集合
        '''

        # 定义Adam优化器
        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=params.learning_rate,
                                          betas=(0.9, 0.98),
                                          eps=1e-09)
        self.params = params
        self.step_num = 0

    # 重写优化器的zero_grad方法(和原版一样)
    def zero_grad(self):
        self.optimizer.zero_grad()

    # 重写优化器的step方法(在原版step的基础上,增加了更新学习率的功能)
    def step(self):
        self.update_learning_rate()
        self.optimizer.step()

    # 更新学习率
    def update_learning_rate(self):
        # 更新学习率的公式,和transformer原论文保持一致
        self.step_num += 1
        learning_rate = np.power(self.params.d_model, -0.5) * \
                             min(np.power(self.step_num, -0.5), self.step_num * np.power(self.params.warmup_steps, -1.5))
        
        # 修改优化器中的学习率
        for param_groups in self.optimizer.param_groups:
            param_groups['lr'] = learning_rate
