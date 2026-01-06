# src/analysis/ssvep_ml_analysis.py

import numpy as np
from numpy import fft
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut

from ..preprocess.preprocessing import calculate_snr_spectrum

def extract_features(trial_data, target_freqs, freq_res):
    """提取单个试次的幅值+SNR特征"""
    fft_complex = fft.rfft(trial_data, axis=1)
    fft_amps = np.abs(fft_complex)
    fft_snr = calculate_snr_spectrum(fft_amps)
    
    features = []
    for ch_idx in range(fft_amps.shape[0]):
        ch_amp = fft_amps[ch_idx]
        ch_snr = fft_snr[ch_idx]
        ch_feats = []
        for f in target_freqs:
            idx1 = int(round(f / freq_res))
            idx2 = int(round(2 * f / freq_res))
            idx3 = int(round(3 * f / freq_res))
            size = ch_amp.size
            v1 = ch_amp[idx1] if idx1 < size else 0
            v2 = ch_amp[idx2] if idx2 < size else 0
            v3 = ch_amp[idx3] if idx3 < size else 0
            s1 = ch_snr[idx1] if idx1 < size else 0
            s2 = ch_snr[idx2] if idx2 < size else 0
            s3 = ch_snr[idx3] if idx3 < size else 0
            ch_feats.extend([v1, v2, v3, s1, s2, s3])
        features.extend(ch_feats)
    return np.array(features)

def build_dataset(x_filtered, selected_channels, target_freqs, fs, sample_points):
    """构建特征矩阵 X_all、标签 y_all、分组 groups_all"""
    FREQ_RES = fs / sample_points
    n_channels, n_samples, n_targets, n_blocks = x_filtered.shape

    X_all, y_all, groups_all = [], [], []

    for block in range(n_blocks):
        for target in range(n_targets):
            raw_trial = x_filtered[selected_channels, :, target, block]
            feat_vec = extract_features(raw_trial, target_freqs, FREQ_RES)
            X_all.append(feat_vec)
            y_all.append(target)
            groups_all.append(block)

    return np.array(X_all), np.array(y_all), np.array(groups_all)

def run_svm_cv(X, y, groups, n_targets, C=1.0, pca_var_ratio=0.95, print_result=True):
    logo = LeaveOneGroupOut()
    clf = SVC(kernel='linear', C=C)
    test_accuracies = []
    train_accuracies = []
    conf_matrices = []
    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=pca_var_ratio)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        clf.fit(X_train_pca, y_train)
        y_pred = clf.predict(X_test_pca)

        acc_test = accuracy_score(y_test, y_pred)
        acc_train = clf.score(X_train_pca, y_train)

        test_accuracies.append(acc_test)
        train_accuracies.append(acc_train)
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        cm = confusion_matrix(y_test, y_pred, labels=np.arange(n_targets))
        conf_matrices.append(cm)

        if print_result:
            fold = len(test_accuracies)
            print(f"Fold {fold}: Train={acc_train:.4f}, Test={acc_test:.4f} → Gap={acc_train - acc_test:.4f}")

    mean_test = np.mean(test_accuracies)
    mean_train = np.mean(train_accuracies)
    std_test = np.std(test_accuracies)

    if print_result:
        print(f"\n>>> 最终结果 <<<")
        print(f"平均训练准确率: {mean_train:.4f}")
        print(f"平均测试准确率: {mean_test:.4f} ± {std_test:.4f}")
        print(f"平均性能差距: {mean_train - mean_test:.4f}")

    return {
        'test_accuracies': test_accuracies,
        'train_accuracies': train_accuracies,
        'conf_matrices': np.array(conf_matrices),
        'y_true_all': y_true_all,
        'y_pred_all': y_pred_all,
        'mean_test_acc': mean_test,
        'mean_train_acc': mean_train,
        'std_test_acc': std_test
    }


def run_svm_for_channels(selected_channels, data, desc_name, target_freqs, fs, sample_points,
                         C=1.0, pca_var_ratio=0.95):
    """
    对给定的通道子集提取特征，并执行6折留一block交叉验证
    
    参数:
        selected_channels: 要使用的通道索引列表（如 [60,61,62]）
        data: 预处理后的四维EEG数据，形状为 (n_channels, n_samples, n_targets, n_blocks)
        desc_name: 当前实验的描述名称（用于打印日志）
        target_freqs: 目标刺激频率列表
        fs: 采样频率
        sample_points: 每个试次的采样点数
    
    返回:
        mean_train_acc: 训练集上内部交叉验证的平均准确率（%）
        mean_test_acc: 测试集（留出block）上的平均准确率（%）

    """
    print(f"\n--- 正在处理: {desc_name} （共 {len(selected_channels)} 个通道） ---")

    # 1. 提取特征
    X, y, groups = build_dataset(data, selected_channels, target_freqs, fs, sample_points)
    n_targets = len(target_freqs)

    # 2. 运行 SVM 交叉验证
    result = run_svm_cv(
        X, y, groups, n_targets,
        C=C,
        pca_var_ratio=pca_var_ratio,
        print_result=False
    )

    return result['mean_train_acc'], result['mean_test_acc']