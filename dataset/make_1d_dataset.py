# _*_ coding: utf-8 _*_
# @File:    make_1d_dataset
# @Time:    2025/12/1 17:27
# @Author:  ArthasMenethil/wuweihang
# @Contact: wuweihang1998@gmail.com
# @Version: V 0.1
import os
import numpy as np


def split_train_test(X, Y, train_ratio=0.9, shuffle=True):
    N = X.shape[0]
    idx = np.arange(N)

    if shuffle:
        np.random.shuffle(idx)

    n_train = int(N * train_ratio)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    return X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]


def make_1d_dataset(f, x_min, x_max, num_samples=10000):
    x = np.linspace(x_min, x_max, num_samples).reshape(-1, 1)
    y = f(x)

    return split_train_test(x, y)


if __name__ == "__main__":

    def f(x):
        return np.sin(0.5 * np.pi * x) + np.sin(np.pi * x)

    X_MIN = -4
    X_MAX = 4
    NUM_SAMPLES = 40000

    SAVE_DIR = "./data"
    SAVE_NAME = "dataset_1d.npz"

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, SAVE_NAME)

    X_train, Y_train, X_test, Y_test = make_1d_dataset(
        f,
        X_MIN,
        X_MAX,
        NUM_SAMPLES
    )

    np.savez(save_path,
             X_train=X_train,
             Y_train=Y_train,
             X_test=X_test,
             Y_test=Y_test)

    print(f"一维函数数据集已保存到：{save_path}")

