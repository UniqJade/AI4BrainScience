# src/evaluation/cross_validation.py
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from ..analysis.ssvep_ml_analysis import extract_features  
from ..analysis.cca_analysis import SSVEPClassifier, calculate_itr               


def run_ssvep_cv(
    x_cleaned,
    reference_signals,
    target_freqs,
    fs,
    n_targets,
    n_blocks,
    selected_channels=None,
    methods=('CCA', 'FBCCA', 'TRCA', 'SVM'),
    time_per_trial=5.64,
    num_subbands=3,
    svm_param_grid=None,
    print_per_block=True
):
    """
    统一执行多种 SSVEP 分类方法的 Leave-One-Block-Out 交叉验证。
    
    支持: CCA, FBCCA, TRCA, SVM
    
    Returns:
        dict: 包含 train/test acc & ITR 的结构化结果
    """
    if selected_channels is None:
        selected_channels = list(range(x_cleaned.shape[0]))
    
    n_chans, n_samples, _, _ = x_cleaned.shape
    FREQ_RES = fs / n_samples

    # 初始化结果容器
    test_accs = {m: [] for m in methods}
    train_accs = {m: [] for m in methods if m in ['TRCA', 'SVM']}  # 只有 TRCA/SVM 有训练 acc
    conf_matrices = {m: [] for m in methods}
    all_labels_list = []
    all_preds_dict = {m: [] for m in methods}

    classifier = SSVEPClassifier(fs=fs)

    # 预构建 SVM 特征
    if 'SVM' in methods:
        X_svm, y_svm, groups_svm = [], [], []
        for block in range(n_blocks):
            for target in range(n_targets):
                trial = x_cleaned[:, :, target, block]
                feat = extract_features(trial[selected_channels], target_freqs, FREQ_RES)
                X_svm.append(feat)
                y_svm.append(target)
                groups_svm.append(block)
        X_svm = np.array(X_svm)
        y_svm = np.array(y_svm)
        groups_svm = np.array(groups_svm)

    # === LOBO 主循环 ===
    for test_block in tqdm(range(n_blocks), desc="Cross-Validation Block", unit="block"):

        # --- 公共标签（按目标索引 0 ~ n_targets-1）---
        labels = list(np.arange(n_targets))  # shape: (n_targets,)
        all_labels_list.extend(labels)

        # --- 准备原始 EEG 测试数据 ---
        test_data_raw = x_cleaned[:, :, :, test_block].transpose(2, 0, 1)  # (Targets, Chans, Time)

        # --- 准备 TRCA 训练数据 ---
        train_indices = [b for b in range(n_blocks) if b != test_block]
        train_data_trca = x_cleaned[:, :, :, train_indices].transpose(2, 3, 0, 1)  # (Targets, Blocks, Chans, Time)

        # ========================
        # 1. CCA / FBCCA / TRCA
        # ========================
        if 'TRCA' in methods or 'CCA' in methods or 'FBCCA' in methods:
            if 'TRCA' in methods:
                W_trca, templates_trca = classifier.train_trca(train_data_trca)

                # 计算 TRCA 训练准确率
                trca_train_preds = []
                trca_train_labels = []
                for i in range(n_targets):
                    for b_idx in train_indices:
                        trial = x_cleaned[:, :, i, b_idx]
                        pred = classifier.predict_trca(trial, W_trca, templates_trca)
                        trca_train_preds.append(pred)
                        trca_train_labels.append(i)
                trca_train_acc = accuracy_score(trca_train_labels, trca_train_preds)
                train_accs['TRCA'].append(trca_train_acc)

            # 测试预测
            cca_pred, fbcca_pred, trca_pred = [], [], []
            for i in range(n_targets):
                trial = test_data_raw[i]
                if 'CCA' in methods:
                    cca_pred.append(classifier.predict_cca(trial, reference_signals))
                if 'FBCCA' in methods:
                    fbcca_pred.append(classifier.predict_fbcca(trial, reference_signals, num_subbands))
                if 'TRCA' in methods:
                    trca_pred.append(classifier.predict_trca(trial, W_trca, templates_trca))

            # 计算混淆矩阵并保存
            if 'CCA' in methods:
                acc = accuracy_score(labels, cca_pred)
                cm = confusion_matrix(labels, cca_pred, labels=np.arange(n_targets))
                test_accs['CCA'].append(acc)
                conf_matrices['CCA'].append(cm)
                all_preds_dict['CCA'].extend(cca_pred)
                if print_per_block:
                    print(f"    CCA - 测试: {acc*100:.2f}%")

            if 'FBCCA' in methods:
                acc = accuracy_score(labels, fbcca_pred)
                cm = confusion_matrix(labels, fbcca_pred, labels=np.arange(n_targets))
                test_accs['FBCCA'].append(acc)
                conf_matrices['FBCCA'].append(cm)
                all_preds_dict['FBCCA'].extend(fbcca_pred)
                if print_per_block:
                    print(f"    FBCCA - 测试: {acc*100:.2f}%")

            if 'TRCA' in methods:
                acc = accuracy_score(labels, trca_pred)
                cm = confusion_matrix(labels, trca_pred, labels=np.arange(n_targets))
                test_accs['TRCA'].append(acc)
                conf_matrices['TRCA'].append(cm)
                all_preds_dict['TRCA'].extend(trca_pred)
                if print_per_block:
                    print(f"    TRCA - 训练: {trca_train_acc*100:.2f}%, 测试: {acc*100:.2f}%")

        # ========================
        # 2. SVM
        # ========================
        if 'SVM' in methods:
            train_mask = (groups_svm != test_block)
            test_mask = (groups_svm == test_block)

            X_train, y_train = X_svm[train_mask], y_svm[train_mask]
            X_test, y_test = X_svm[test_mask], y_svm[test_mask]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 默认参数网格
            if svm_param_grid is None:
                svm_param_grid = {
                    'C': [1, 10, 100],
                    'gamma': ['scale', 0.0001, 0.001],
                    'kernel': ['rbf']
                }

            grid = GridSearchCV(SVC(), svm_param_grid, cv=3, n_jobs=-1)
            grid.fit(X_train_scaled, y_train)

            y_pred_svm = grid.best_estimator_.predict(X_test_scaled)
            svm_test_acc = accuracy_score(y_test, y_pred_svm)
            svm_train_acc = grid.best_score_
            svm_cm = confusion_matrix(y_test, y_pred_svm, labels=np.arange(n_targets))

            test_accs['SVM'].append(svm_test_acc)
            train_accs['SVM'].append(svm_train_acc)
            conf_matrices['SVM'].append(svm_cm)
            all_preds_dict['SVM'].extend(y_pred_svm.tolist()) # 转为列表以便扩展

            if print_per_block:
                print(f"    SVM - Train: {svm_train_acc*100:.2f}%, Test: {svm_test_acc*100:.2f}%")
                print(f"Best params = {grid.best_params_}")
                real_train_acc = accuracy_score(y_train, grid.best_estimator_.predict(X_train_scaled))
                print(f"  Real Train Acc (full): {real_train_acc*100:.2f}%")

    # === 计算平均值和 ITR ===
    mean_test_acc = {m: np.mean(test_accs[m]) for m in methods}
    mean_train_acc = {}
    for m in ['TRCA', 'SVM']:
        if m in methods:
            mean_train_acc[m] = np.mean(train_accs[m])

    mean_itr_test = {m: calculate_itr(n_targets, mean_test_acc[m], time_per_trial) for m in methods}
    mean_itr_train = {}
    for m in ['TRCA', 'SVM']:
        if m in methods:
            mean_itr_train[m] = calculate_itr(n_targets, mean_train_acc[m], time_per_trial)

    return {
        'test': {
            'acc': mean_test_acc,
            'itr': mean_itr_test,
            'acc_list': test_accs,
            'conf_matrices': conf_matrices,
            'all_labels': all_labels_list,
            'all_preds': all_preds_dict
        },
        'train': {
            'acc': mean_train_acc,
            'itr': mean_itr_train,
            'acc_list': train_accs
        }
    }