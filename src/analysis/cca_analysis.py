# src/analysis/cca_analysis.py

"""
基于CCA的SSVEP分析模块
包括： CCA, 滤波器组CCA (FB-CCA), 和 TRCA 算法
"""
import os
import sys
import numpy as np
from scipy.stats import pearsonr

sys.path.append(os.path.abspath('..'))
from src.preprocess.preprocessing import cheby_bandpass_filter

def get_reference_signals(n_samples, target_freq, fs, num_harmonics, target_phase=0.0):
    """
    生成用于 SSVEP 的正弦-余弦参考信号

    参数:
    - n_samples: 信号样本数量
    - target_freq: 目标频率 (Hz)
    - fs: 采样频率 (Hz)
    - num_harmonics: 谐波数量

    返回：形状为 (2 * num_harmonics, len_time) 的数组
    """
    reference_signals = []
    t = np.arange(n_samples) / fs
    
    for i in range(1, num_harmonics + 1):
        reference_signals.append(np.sin(2 * np.pi * i * target_freq * t + (i * target_phase)))
        reference_signals.append(np.cos(2 * np.pi * i * target_freq * t + (i * target_phase)))
        
    return np.array(reference_signals)

def calculate_itr(n_classes, acc, time_sec):
    """
    计算信息传输率 (ITR) 以 bits/min 为单位

    参数:
    - n_classes: 类别数量
    - acc: 准确率 (0 到 1 之间)
    - time_sec: 每次选择所需的时间（秒）

    返回: ITR (bits/min)
    """
    # 处理极端数据，避免除以零或对数的零
    if acc <= 1.0 / n_classes:
        return 0.0
    
    # 限制准确率以避免数值问题
    acc = min(acc, 0.9999)
    
    # 使用标准公式计算 ITR
    itr_bits = (np.log2(n_classes) + 
                acc * np.log2(acc) + 
                (1 - acc) * np.log2((1 - acc) / (n_classes - 1)))
    
    # 转换为每分钟的比特数
    itr_per_min = (itr_bits / time_sec) * 60
    
    return max(0.0, itr_per_min)

class SSVEPClassifier:

    """
    基于CCA的SSVEP分类器，支持三种方法：
    1. 标准CCA
    2. 滤波器组CCA (FB-CCA) 
    3. 任务相关成分分析 (TRCA)
    """

    def __init__(self, fs=250):
        self.fs = fs

    # ==========================
    # CCA 核心求解器
    # ==========================
    def solve_cca(self, X, Y):

        """
        使用 QR 分解计算典型相关系数 (rho)
        X: EEG 数据 (n_channels, n_time)
        Y: 参考信号 (n_refs, n_time)
        """

        # 1. 中心化数据
        X = X - np.mean(X, axis=1, keepdims=True)
        Y = Y - np.mean(Y, axis=1, keepdims=True)
        
        # 2. QR 分解
        Q_x, R_x = np.linalg.qr(X.T)
        Q_y, R_y = np.linalg.qr(Y.T)
        
        # 3. 内相关矩阵的 SVD
        S = np.linalg.svd(np.dot(Q_x.T, Q_y), compute_uv=False)
        
        # 返回最大的典型相关系数
        return S[0]

    # 1. Standard CCA
    def predict_cca(self, test_data, reference_signals):

        """
        使用标准 CCA 进行分类

        参数：
        - test_data: 测试数据 (n_channels, n_time)
        - reference_signals: 参考信号列表，每个元素形状为 (2 * num_harmonics, n_time)

        返回: 预测的目标索引
        """

        corrs = [self.solve_cca(test_data, ref) for ref in reference_signals]
        return np.argmax(corrs)

    # 2. Filter Bank CCA (FB-CCA)
    def predict_fbcca(self, test_data, reference_signals, num_subbands=3):

        """
        使用包含多个子带的 FB-CCA 进行分类

        参数：
        - test_data: 测试数据 (n_channels, n_time)
        - reference_signals: 参考信号列表，每个元素形状为 (2 * num_harmonics, n_time)
        - num_subbands: 使用的子带数量

        返回: 预测的目标索引
        """

        # FB-CCA 的标准权重: n^-a + b (a=1.25, b=0.25)
        a, b = 1.25, 0.25
        weights = [(n+1)**(-a) + b for n in range(num_subbands)]
        
        rho_integrated = np.zeros(len(reference_signals))

        # 定义子带频率范围
        subband_ranges = [
            (6, 88),   # 基频 + 所有谐波
            (14, 88),  # 从第2谐波开始
            (22, 88)   # 更高谐波
        ]
        
        for k in range(min(num_subbands, len(subband_ranges))):
            low_cut, high_cut = subband_ranges[k]
            
            # 带通滤波
            X_sb = cheby_bandpass_filter(data=test_data, lowcut=low_cut, highcut=high_cut, fs=self.fs, order=4)
            
            # 计算加权相关系数
            for i, ref in enumerate(reference_signals):
                rho = self.solve_cca(X_sb, ref)
                rho_integrated[i] += weights[k] * (rho ** 2)
        
        return np.argmax(rho_integrated)

    # 3. TRCA (Task-Related Component Analysis)
    def train_trca(self, train_data):

        """
        训练 TRCA 模型，计算空间滤波器 (W) 和模板 (平均训练数据)

        参数：
        - train_data: 训练数据，形状为 (n_targets, n_blocks, n_channels, n_time)

        返回: W (空间滤波器), Templates (平均训练数据)
        """

        n_targets, n_blocks, n_ch, n_time = train_data.shape
        W = np.zeros((n_targets, n_ch))
        
        # 计算模板 (平均训练数据)
        templates = np.mean(train_data, axis=1) 
        
        for target_idx in range(n_targets):
            # 将所有块沿时间轴拼接
            X = train_data[target_idx] # (Blocks, Ch, Time)
            
            # 1. 计算 S (总协方差)
            # S = Σᵢ XᵢXᵢᵀ
            X_concat = np.transpose(X, (1, 0, 2)).reshape(n_ch, -1) # (n_ch, n_blocks*n_time)
            X_concat -= np.mean(X_concat, axis=1, keepdims=True)
            S = np.dot(X_concat, X_concat.T)
            
            # 2. 计算 Q (跨块协方差)
            # Q = Σᵢ Σⱼ≠ᵢ XᵢXⱼᵀ
            Q = np.zeros((n_ch, n_ch))
            for i in range(n_blocks):
                Xi = X[i] - np.mean(X[i], axis=1, keepdims=True)  # (n_ch, n_time)
                for j in range(n_blocks):
                    if i != j:
                        Xj = X[j] - np.mean(X[j], axis=1, keepdims=True)
                        Q += Xi @ Xj.T
            
            # 自适应正则化防止过拟合
            # 对不同通道数/数据尺度更鲁棒 
            reg_param = 0.1 * np.trace(S) / n_ch
            S_reg = S + reg_param * np.eye(n_ch)
            
            # 3. 求解广义特征值问题: Q*w = lambda*S*w
            try:
                eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S_reg) @ Q)
                eigvals = np.real(eigvals)
                eigvecs = np.real(eigvecs)
                W[target_idx] = eigvecs[:, np.argmax(eigvals)]
            except np.linalg.LinAlgError:
                # 备用方案：使用第一个主成分
                U, _, _ = np.linalg.svd(S_reg)
                W[target_idx] = U[:, 0]
            
        return W, templates

    def predict_trca(self, test_data, W, templates):

        """
        使用 TRCA 和学习到的的空间滤波器进行分类

        参数：
        - test_data: 测试数据 (n_channels, n_time)
        - W: 空间滤波器权重，形状为 (n_targets, n_channels)
        - templates: 模板信号，形状为 (n_targets, n_channels, n_time)

        返回: 预测的目标索引

        """

        # 中心化测试数据
        test_data = test_data - np.mean(test_data, axis=1, keepdims=True)

        # 计算与每个模板的相关系数
        corrs = []

        for i in range(len(templates)):

            # 使用空间滤波器投影测试数据和模板
            feat_test = np.dot(W[i].T, test_data) 
            feat_template = np.dot(W[i].T, templates[i])
            
            # 计算 Pearson 相关系数
            if np.std(feat_test) > 1e-10 and np.std(feat_template) > 1e-10:
                r, _ = pearsonr(feat_test, feat_template)
                corrs.append(r)
            else:
                corrs.append(0.0)
            
        return np.argmax(corrs)
    