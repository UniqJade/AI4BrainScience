# src/analysis/ssvep_parameter_sweep.py

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from .cca_analysis import SSVEPClassifier, get_reference_signals, calculate_itr
from ..preprocess.preprocessing import get_freq_phase, preprocess_trial


class SSVEPParameterSweeper:
    """
    对 SSVEP 分类器（CCA、FBCCA、TRCA）进行时长与谐波数的参数扫描，
    结合留一区块交叉验证与信息传输率（ITR）评估。
    """

    def __init__(self,
                 fs=250,
                 n_targets=40,
                 n_blocks=6,
                 selected_channels=None,
                 gaze_shift=0.50,
                 window_start=0.14):
        """
        使用实验配置初始化扫描器。

        参数:
            fs (int): 采样频率（Hz）。
            n_targets (int): SSVEP 目标数量。
            n_blocks (int): 每个目标的试验区块数。
            selected_channels (list): 要使用的通道索引列表。
            gaze_shift (float): 从提示出现到预期注视转移的时间（秒）。
            window_start (float): 从刺激开始到分析窗口起始的时间（秒）。
        """
        self.fs = fs
        self.n_targets = n_targets
        self.n_blocks = n_blocks
        self.selected_channels = selected_channels or [47, 55, 61, 60, 62, 54, 56, 53, 57]
        self.gaze_shift = gaze_shift
        self.window_start = window_start
        self.total_delay = gaze_shift + window_start


    def load_and_preprocess_data(self, data_path):
        """
        加载原始 SSVEP 数据，并对每个试次进行带通滤波。

        参数:
            data_path (str): .npy 文件路径，数据形状为 (通道, 时间, 目标, 区块)。

        返回:
            np.ndarray: 滤波后的数据，仅包含选定通道，形状同输入。
        """
        x = np.load(data_path)
        print(f"  原始数据形状: {x.shape}")

        x_roi = x[self.selected_channels, :, :, :]
        x_cleaned = np.zeros_like(x_roi)

        for target in tqdm(range(self.n_targets), desc="  预处理目标"):
            for block in range(self.n_blocks):
                x_cleaned[:, :, target, block] = preprocess_trial(
                    x_roi[:, :, target, block],
                    fs=self.fs,
                    low=4,
                    high=60
                )

        print(f"  滤波后数据形状: {x_cleaned.shape}")
        return x_cleaned

    def run_sweep(self, data, durations=None, harmonics=None, methods=None, fbcca_subbands=3):
        """
        对信号时长与时谐波数进行网格搜索，采用留一区块交叉验证。

        参数:
            data (np.ndarray): 预处理后的 EEG 数据，形状为 (通道, 时间, 目标, 区块)
            durations (array-like): 信号时长列表（秒），步长 0.2
            harmonics (list): 谐波数量列表，默认为 [1, 2, 3, 4, 5]
            methods (list): 要评估的分类器，默认为 ['CCA', 'FBCCA', 'TRCA']
            fbcca_subbands (int): FBCCA 使用的子频带数量

        返回:
            dict: 每种方法的准确率与 ITR 网格结果（含标准差）。
            dict: 每种方法在最佳准确率和最佳 ITR 下的参数。
        """
        max_duration = 5.0 - self.window_start  # = 5.0 - 0.14 = 4.86
        if durations is None:
            durations = np.arange(0.2, max_duration, 0.2)
        if harmonics is None:
            harmonics = [1, 2, 3, 4, 5]
        if methods is None:
            methods = ['CCA', 'FBCCA', 'TRCA']

        # 初始化结果网格
        results_grid = {}
        for method in methods:
            results_grid[method] = {
                'acc': np.full((len(durations), len(harmonics)), np.nan),
                'acc_std': np.full((len(durations), len(harmonics)), np.nan),
                'itr': np.full((len(durations), len(harmonics)), np.nan)
            }

        # 初始化分类器
        classifier = SSVEPClassifier(fs=self.fs)

        total_steps = len(durations) * len(harmonics)
        pbar = tqdm(total=total_steps, desc="  参数扫描")

        for d_idx, duration in enumerate(durations):
            start_sample = int(self.window_start * self.fs)
            n_samples = int(duration * self.fs)
            end_sample = start_sample + n_samples

            if end_sample > data.shape[1]:
                print(f"\n  警告: 时长 {duration:.1f}s 超出数据长度，跳过。")
                pbar.update(len(harmonics))
                continue

            current_data = data[:, start_sample:end_sample, :, :]
            selection_time = duration + self.gaze_shift

            # --------------------------
            # TRCA：不依赖谐波数 → 每个时长只计算一次
            # --------------------------
            fold_accs_trca = []

            for test_block in range(self.n_blocks):
                # 提取测试集
                test_set = current_data[:, :, :, test_block].transpose(2, 0, 1)  # (目标, 通道, 时间)
                train_indices = [b for b in range(self.n_blocks) if b != test_block]
                train_set = current_data[:, :, :, train_indices].transpose(2, 3, 0, 1)  # (目标, 区块, 通道, 时间)

                # 训练 TRCA 模型
                W, templates = classifier.train_trca(train_set)
                labels = list(range(self.n_targets))
                preds_trca = [
                    classifier.predict_trca(test_set[i], W, templates)
                    for i in range(self.n_targets)
                ]
                fold_accs_trca.append(accuracy_score(labels, preds_trca))

            # 计算 TRCA 的平均准确率与 ITR
            avg_acc_trca = np.mean(fold_accs_trca)
            itr_trca = calculate_itr(self.n_targets, avg_acc_trca, selection_time)

            # 广播到所有谐波列（TRCA 与谐波无关）
            results_grid['TRCA']['acc'][d_idx, :] = avg_acc_trca
            results_grid['TRCA']['itr'][d_idx, :] = itr_trca
            results_grid['TRCA']['acc_std'][d_idx, :] = np.std(fold_accs_trca, ddof=1)

            # --------------------------
            # CCA / FBCCA：依赖谐波数 → 在谐波循环中计算
            # --------------------------
            methods_to_run_in_harmonic_loop = [m for m in methods if m in ['CCA', 'FBCCA']]
            if not methods_to_run_in_harmonic_loop:
                pbar.update(len(harmonics))
                continue

            for h_idx, n_harmonics in enumerate(harmonics):
                # 为每个目标生成参考信号（基于当前谐波数）
                refs = []
                for target_idx in range(self.n_targets):
                    freq, phase = get_freq_phase(target_idx)
                    ref = get_reference_signals(
                        n_samples=n_samples,
                        target_freq=freq,
                        fs=self.fs,
                        num_harmonics=n_harmonics,
                        target_phase=phase
                    )
                    refs.append(ref)

                fold_accs = {m: [] for m in methods_to_run_in_harmonic_loop}

                # 留一区块交叉验证
                for test_block in range(self.n_blocks):
                    test_set = current_data[:, :, :, test_block].transpose(2, 0, 1)  # (目标, 通道, 时间)
                    train_indices = [b for b in range(self.n_blocks) if b != test_block]
                    train_set = current_data[:, :, :, train_indices].transpose(2, 3, 0, 1)  # (目标, 区块, 通道, 时间)

                    labels = list(range(self.n_targets))
                    preds_cca, preds_fbcca = [], []

                    for i in range(self.n_targets):
                        trial = test_set[i]
                        if 'CCA' in methods_to_run_in_harmonic_loop:
                            preds_cca.append(classifier.predict_cca(trial, refs))
                        if 'FBCCA' in methods_to_run_in_harmonic_loop:
                            preds_fbcca.append(classifier.predict_fbcca(trial, refs, num_subbands=fbcca_subbands))

                    if 'CCA' in methods_to_run_in_harmonic_loop:
                        fold_accs['CCA'].append(accuracy_score(labels, preds_cca))
                    if 'FBCCA' in methods_to_run_in_harmonic_loop:
                        fold_accs['FBCCA'].append(accuracy_score(labels, preds_fbcca))

                # 计算并存储指标
                for method in methods_to_run_in_harmonic_loop:
                    acc_array = np.array(fold_accs[method])
                    avg_acc = np.mean(acc_array)
                    std_acc = np.std(acc_array, ddof=1)
                    itr = calculate_itr(self.n_targets, avg_acc, selection_time)

                    results_grid[method]['acc'][d_idx, h_idx] = avg_acc
                    results_grid[method]['acc_std'][d_idx, h_idx] = std_acc
                    results_grid[method]['itr'][d_idx, h_idx] = itr

                pbar.update(1)

        pbar.close()

        optimal_params = self._find_optimal_parameters(results_grid, durations, harmonics, methods)
        return results_grid, optimal_params

    def _find_optimal_parameters(self, results_grid, durations, harmonics, methods):
        """提取每种方法的最佳准确率与最佳 ITR 对应的参数。"""
        optimal = {}
        for method in methods:
            acc_grid = results_grid[method]['acc']
            std_grid = results_grid[method]['acc_std']
            itr_grid = results_grid[method]['itr']

            # 最佳准确率
            best_acc_idx = np.unravel_index(np.nanargmax(acc_grid), acc_grid.shape)
            best_acc = acc_grid[best_acc_idx]
            best_acc_std = std_grid[best_acc_idx]
            best_acc_dur = durations[best_acc_idx[0]]
            best_acc_harm = harmonics[best_acc_idx[1]]

            # 最佳 ITR
            best_itr_idx = np.unravel_index(np.nanargmax(itr_grid), itr_grid.shape)
            best_itr = itr_grid[best_itr_idx]
            best_itr_dur = durations[best_itr_idx[0]]
            best_itr_harm = harmonics[best_itr_idx[1]]

            optimal[method] = {
                'best_acc': float(best_acc),
                'best_acc_std': float(best_acc_std),
                'best_acc_duration': float(best_acc_dur),
                'best_acc_harmonics': int(best_acc_harm),
                'best_itr': float(best_itr),
                'best_itr_duration': float(best_itr_dur),
                'best_itr_harmonics': int(best_itr_harm)
            }
        return optimal