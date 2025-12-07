import argparse
import os

image_path = '../../../mn/mnist_test/0/mnist_test_3.jpg'
EPS = 255


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', default=image_path,
                                 help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--max_epsilon', default=EPS, type=float)
        self.parser.add_argument('--iter', default=10000, type=int)
        self.parser.add_argument('--confidence', default=0, type=float)
        self.parser.add_argument('--phase', type=str, default='test')
        self.parser.add_argument('--name', type=str, default='greedyfool')
        self.parser.add_argument('--init_lambda1', type=float, default=1e-3,
                                 help='set the init value for lambda1 at the start of binary search')
        self.parser.add_argument('--lr_e', type=float, default=0.5,
                                 help='initial learning rate for noise')
        self.parser.add_argument('--lr_g', type=float, default=0.1,
                                 help='initial learning rate for mask G')
        self.parser.add_argument('--momentum', default=0.9, type=float)

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()
        self.opt = opt
        return self.opt
