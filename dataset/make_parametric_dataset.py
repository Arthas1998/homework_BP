# _*_ coding: utf-8 _*_
# @File:    make_parametric_dataset
# @Time:    2025/12/1 17:28
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

def make_parametric_dataset(fx, fy, fz, t_min, t_max, num_samples=10000):
    t = np.linspace(t_min, t_max, num_samples).reshape(-1, 1)

    x = fx(t)
    y = fy(t)
    z = fz(t)

    X = np.hstack([x, y])
    Y = z

    return split_train_test(X, Y)

if __name__ == "__main__":

    def fx(t):
        return (t + 0.5 * np.pi) * np.sin(t + 0.5 * np.pi)

    def fy(t):
        return (t + 0.5 * np.pi) * np.cos(t + 0.5 * np.pi)

    def fz(t):
        return 1.5 * t

    T_MIN = 0
    T_MAX = 10 * np.pi
    NUM_SAMPLES = 40000

    SAVE_DIR = "./data"
    SAVE_NAME = "dataset_param.npz"

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, SAVE_NAME)

    X_train, Y_train, X_test, Y_test = make_parametric_dataset(
        fx,
        fy,
        fz,
        T_MIN,
        T_MAX,
        NUM_SAMPLES
    )

    np.savez(save_path,
             X_train=X_train,
             Y_train=Y_train,
             X_test=X_test,
             Y_test=Y_test)

    print(f"参数方程数据集已保存到：{save_path}")

