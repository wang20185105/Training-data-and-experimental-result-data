# -*- coding: utf-8 -*-
"""
@ project: S2AC
@ author: wzy
@ file: params.py
@ time: 2023/10/10 15:16
"""

class param():
    def __init__(self):
        self.lr = 1e-3
        self.weight_clay = 5e-4
        self.momentum = 0.9
        self.train_batch = 8
        self.test_batch = 8
        self.epochs = 160
