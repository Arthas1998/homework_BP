# _*_ coding: utf-8 _*_
# @File:    loader
# @Time:    2025/12/1 17:29
# @Author:  ArthasMenethil/wuweihang
# @Contact: wuweihang1998@gmail.com
# @Version: V 0.1
import numpy as np

class DatasetLoader:

    def __init__(self, npz_path):
        self.path = npz_path
        self.data = np.load(npz_path)

    def train(self):
        return self.data["X_train"], self.data["Y_train"]

    def test(self):
        return self.data["X_test"], self.data["Y_test"]

    def all(self):
        return self.train() + self.test()

    def summary(self):
        print("数据集路径:", self.path)
        for k in self.data.files:
            print(f"{k}: shape = {self.data[k].shape}")
