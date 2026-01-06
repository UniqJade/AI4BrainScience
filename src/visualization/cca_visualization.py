"""
Visualization and analysis utilities for SSVEP classification results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def visualize_ssvep_fixed_params(mean_acc, mean_itr, acc_results, all_labels, all_preds, 
                                target_freqs, methods=['CCA', 'FBCCA', 'TRCA'],
                                save_path=None):
    """
    Create comprehensive visualization of SSVEP classification results with multiple CCA-based algorithms(with fixed parameters)
    
    Parameters:
    -----------
    mean_acc : dict
        Mean accuracy for each method
    mean_itr : dict
        Mean ITR for each method
    acc_results : dict
        Per-block accuracy results
    all_labels : list
        Ground truth labels (all trials)
    all_preds : dict
        Predictions for each method (all trials)
    target_freqs : ndarray
        Array of target frequencies
    methods : list
        List of method names to plot
    save_path : str
        Path to save figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Generated figure object
    """
    
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3,
                          left=0.07, right=0.96, top=0.94, bottom=0.06)
    
    colors = ['#4A90E2', '#E27D60', '#50C878']
    n_blocks = len(acc_results[methods[0]])
    
    # ========================================
    # Plot 1: Accuracy Bar Chart
    # ========================================
    ax1 = fig.add_subplot(gs[0, 0])
    accuracies = [mean_acc[m] * 100 for m in methods]
    bars = ax1.bar(methods, accuracies, color=colors, alpha=0.85, 
                   edgecolor='black', linewidth=1.5)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 1.5,
                f'{acc:.1f}%', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Classification Accuracy', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylim([0, 105])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # ========================================
    # Plot 2: ITR Bar Chart
    # ========================================
    ax2 = fig.add_subplot(gs[0, 1])
    itrs = [mean_itr[m] for m in methods]
    bars2 = ax2.bar(methods, itrs, color=colors, alpha=0.85, 
                    edgecolor='black', linewidth=1.5)
    
    for bar, itr_val in zip(bars2, itrs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 1,
                f'{itr_val:.1f}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('ITR (bits/min)', fontsize=13, fontweight='bold')
    ax2.set_title('Information Transfer Rate', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylim([0, max(itrs) * 1.15])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # ========================================
    # Plot 3: Per-Block Performance
    # ========================================
    ax3 = fig.add_subplot(gs[0, 2])
    x_blocks = np.arange(1, n_blocks + 1)
    
    for method, color in zip(methods, colors):
        block_accs = [acc_results[method][i] * 100 for i in range(n_blocks)]
        ax3.plot(x_blocks, block_accs, 'o-', label=method, 
                color=color, linewidth=2.5, markersize=9, alpha=0.85)
    
    ax3.set_xlabel('Block Number', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax3.set_title('Per-Block Performance', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xticks(x_blocks)
    ax3.set_ylim([0, 105])
    ax3.legend(fontsize=11, frameon=True, shadow=True, loc='lower right')
    ax3.grid(alpha=0.3, linestyle='--')
    
    # ========================================
    # Plots 4-6: Confusion Matrices
    # ========================================
    # Sort frequencies for better visualization
    sorted_indices = np.argsort(target_freqs)
    sorted_freqs = target_freqs[sorted_indices]
    
    for idx, method in enumerate(methods):
        ax = fig.add_subplot(gs[1, idx])
        
        # Get confusion matrix
        n_targets = len(np.unique(all_labels))
        cm = confusion_matrix(all_labels, all_preds[method], labels=range(n_targets))
        
        # Reorder by frequency
        cm_sorted = cm[sorted_indices, :][:, sorted_indices]
        cm_norm = cm_sorted.astype('float') / cm_sorted.sum(axis=1)[:, np.newaxis]
        
        # Plot heatmap
        im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Accuracy', rotation=270, labelpad=20, 
                       fontsize=11, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)
        
        # Show every 5th frequency
        tick_positions = np.arange(0, n_targets, 5)
        tick_labels = [f'{sorted_freqs[i]:.1f}' for i in tick_positions]
        
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_yticklabels(tick_labels, fontsize=9)
        
        ax.set_xlabel('Predicted (Hz)', fontsize=11, fontweight='bold')
        ax.set_ylabel('True (Hz)', fontsize=11, fontweight='bold')
        
        diag_acc = np.trace(cm_sorted) / np.sum(cm_sorted) * 100
        ax.set_title(f'{method} | Accuracy: {diag_acc:.2f}%', 
                    fontsize=13, fontweight='bold', pad=10)
        
        # Grid lines
        ax.set_xticks(np.arange(n_targets) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_targets) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"图片已保存到： {save_path}")
    
    plt.show()
    return fig


def print_fixed_params_results(mean_acc_test, mean_itr_test, acc_results_test,
                               mean_acc_train=None, mean_itr_train=None, acc_results_train=None,
                               methods=['CCA', 'FBCCA', 'TRCA']):
    """
    打印格式化结果摘要表，支持训练集和测试集。
    
    参数:
    - mean_acc_test / mean_itr_test: 测试集平均准确率和ITR
    - acc_results_test: 测试集每折准确率
    - mean_acc_train / mean_itr_train: （可选）训练集平均准确率和ITR（仅TRCA有意义）
    - acc_results_train: （可选）训练集每折准确率
    """
    print("\n" + "="*90)
    print("SSVEP 分类结果汇总".center(90))
    print("-"*90)
    if mean_acc_train is not None:
        print(f"{'方法':<10} {'训练集准确率':<20} {'训练集 ITR':<15} {'测试集准确率':<20} {'测试集 ITR':<15}")
        print('-'*90)

        for method in methods:
            # 测试集
            test_mean = mean_acc_test[method] * 100
            test_std = np.std(acc_results_test[method]) * 100
            test_itr = mean_itr_test[method]
            test_str = f"{test_mean:.2f}% ± {test_std:.2f}%"

            # 训练集
            if method == 'TRCA' and mean_acc_train is not None:
                train_mean = mean_acc_train[method] * 100
                train_std = np.std(acc_results_train[method]) * 100
                train_itr = mean_itr_train[method]
                train_str = f"{train_mean:.2f}% ± {train_std:.2f}%"
                print(f"{method:<10} {train_str:<20} {train_itr:<15.2f} {test_str:<20} {test_itr:<15.2f}")
            else:
                print(f"{method:<10} {'N/A':<20} {'N/A':<15} {test_str:<20} {test_itr:<15.2f}")

    else:
        print(f"{'方法':<10} {'测试集准确率':<20} {'测试集 ITR (bits/min)':<20}")
        print("-"*90)
        for method in methods:
            mean = mean_acc_test[method] * 100
            std = np.std(acc_results_test[method]) * 100
            itr = mean_itr_test[method]
            test_acc_str = f"{mean:.2f}% ± {std:.2f}%"
            print(f"{method:<10} {test_acc_str:<20} {itr:<20.2f}")
    
    print("="*90)
    
    print("="*90)
    
    # 性能排名（按测试集准确率）
    print("\n【性能排名】（按测试集准确率）")
    sorted_methods = sorted(methods, key=lambda m: mean_acc_test[m], reverse=True)
    for rank, method in enumerate(sorted_methods, 1):
        print(f"  {rank}. {method:<8} - {mean_acc_test[method]*100:.2f}%")



def visualize_duration_harmonics_sweep(results_grid, durations, harmonics, save_path=None):
    """
    可视化 SSVEP 分类器参数扫描结果（信号时长 vs 谐波数）。
    
    生成一个包含三行子图的图像：
    1. 每种谐波数下，准确率随信号时长的变化曲线；
    2. 每种谐波数下，信息传输率（ITR）随信号时长的变化曲线；
    3. 每种方法的准确率热力图（横轴：时长，纵轴：谐波数）。

    参数:
        results_grid (dict): 嵌套字典，包含各方法的评估结果。
                             结构: {'方法名': {'acc': np.array, 'itr': np.array}}
                             数组形状应为 (时长数量, 谐波数量)
        durations (array-like): 使用的信号时长列表（单位：秒），用于横轴。
        harmonics (array-like): 使用的谐波数量列表。
        save_path (str, 可选): 图像保存路径。默认为 None（不保存）。
    """
    print("\n正在生成可视化图像...")

    sns.set_style("whitegrid")
    methods = ['CCA', 'FBCCA', 'TRCA']
    colors = ['#4A90E2', '#E27D60', '#50C878']

    n_harmonics = len(harmonics)
    n_methods = len(methods)

    # 总列数 = 谐波数 × 方法数，用于灵活布局子图
    total_cols = n_harmonics * n_methods 

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, total_cols, hspace=0.4, wspace=1.0)

    # 定义子图跨列宽度
    top_span = n_methods       # 上两行：每个谐波占 n_methods 列
    bottom_span = n_harmonics  # 底部热力图：每个方法占 n_harmonics 列

    # ---------------------------------------------------------
    # 第一行：准确率 vs 信号时长（按谐波数分组）
    # ---------------------------------------------------------
    for i, n_harm in enumerate(harmonics):
        col_start = i * top_span
        col_end = col_start + top_span
        ax = fig.add_subplot(gs[0, col_start:col_end])
        
        harm_idx = i
        for method, color in zip(methods, colors):
            # 数组形状为 (时长, 谐波)，因此取 [:, harm_idx]
            acc_line = results_grid[method]['acc'][:, harm_idx] * 100
            ax.plot(durations, acc_line, 'o-', label=method, color=color, linewidth=2, markersize=5)
        
        ax.set_xlabel('Duration (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Accuracy vs Duration\n(Harmonics H={n_harm})', fontsize=12, fontweight='bold')
        
        if i == 0: 
            ax.legend(fontsize=9, loc='lower right')
        
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 105])

    # ---------------------------------------------------------
    # 第二行：ITR vs 信号时长（按谐波数分组）
    # ---------------------------------------------------------
    for i, n_harm in enumerate(harmonics):
        col_start = i * top_span
        col_end = col_start + top_span
        ax = fig.add_subplot(gs[1, col_start:col_end])
        
        harm_idx = i
        for method, color in zip(methods, colors):
            itr_line = results_grid[method]['itr'][:, harm_idx]
            ax.plot(durations, itr_line, 's--', label=method, color=color, linewidth=2, markersize=5)
            
        ax.set_xlabel('Duration (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Information Transfer Rate (bits/min)', fontsize=11, fontweight='bold')
        ax.set_title(f'ITR vs Duration\n(Harmonics H={n_harm})', fontsize=12, fontweight='bold')
        
        if i == 0:
            ax.legend(fontsize=9)
        
        ax.grid(alpha=0.3)

    # ---------------------------------------------------------
    # 第三行：准确率热力图（按方法分组）
    # ---------------------------------------------------------
    for idx, method in enumerate(methods):
        col_start = idx * bottom_span
        col_end = col_start + bottom_span
        ax = fig.add_subplot(gs[2, col_start:col_end])
        
        # 转置数据以便 imshow 显示：(谐波数, 时长数)
        # origin='lower' 使谐波数从下往上递增
        acc_data = results_grid[method]['acc'].T * 100
        
        im = ax.imshow(acc_data, cmap='YlGnBu', aspect='auto', 
                       vmin=0, vmax=100, origin='lower')
        
        # 设置横轴刻度（时长）
        x_ticks = np.arange(0, len(durations), max(1, len(durations)//6))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{durations[i]:.1f}' for i in x_ticks], fontsize=8, rotation=45, ha='right')
        
        # 设置纵轴刻度（谐波数）
        ax.set_yticks(range(len(harmonics)))
        ax.set_yticklabels(harmonics, fontsize=9)
        
        ax.set_xlabel('Duration (s)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Harmonics', fontsize=10, fontweight='bold')
        ax.set_title(f'{method} Accuracy Heatmap', fontsize=11, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Accuracy (%)', rotation=0, labelpad=15, fontsize=9)
        
        # 标记最高准确率点
        opt_idx = np.unravel_index(np.argmax(acc_data), acc_data.shape)
        # 注意：由于已转置，opt_idx = (谐波索引, 时长索引)
        ax.plot(opt_idx[1], opt_idx[0], 'r*', markersize=15, 
                markeredgecolor='white', markeredgewidth=1.5)

    # 手动调整边距，确保图像充分利用画布
    plt.subplots_adjust(
        top=0.95,
        bottom=0.08,
        left=0.05,
        right=0.98,
        hspace=0.4,
        wspace=1.5 
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"图像已保存至：{save_path}")
        
    return fig


def visualize_accuracy_itr_vs_window_start(
    results_acc: dict,
    results_itr: dict,
    valid_window_starts: np.ndarray,
    fixed_duration: float,
    methods: list = ['CCA', 'FBCCA', 'TRCA'],
    colors: dict = None,
    markers: dict = None,
    figsize: tuple = (12, 5),
    save_path: str = None
):
    """
    绘制多种 SSVEP 分类方法的准确率（Accuracy）和信息传输率（ITR）随窗口起始时间的变化曲线。

    参数
    ----------
    results_acc : dict
        以方法名称为键的字典，值为准确率列表或数组（未乘以100，即 0~1 范围）。
    results_itr : dict
        以方法名称为键的字典，值为 ITR 列表或数组（单位：bits/min）。
    valid_window_starts : array-like
        对应结果的窗口起始时间数组（单位：秒）。
    fixed_duration : float
        固定的分析窗口长度（秒），用于图表标题说明。
    methods : list, 可选
        要绘制的方法名称列表。默认为 ['CCA', 'FBCCA', 'TRCA']。
    colors : dict, 可选
        每种方法对应的颜色映射。若为 None，则使用内置默认颜色。
    markers : dict, 可选
        每种方法对应的标记样式。若为 None，则使用内置默认标记。
    figsize : tuple, 可选
        图像尺寸（宽, 高）。默认为 (12, 5)。
    save_path : str, 可选
        若提供路径（如 'results.png'），则将图像保存到该位置。
    """
    # 设置绘图风格
    sns.set_style("whitegrid")
    
    # 设置默认颜色和标记
    if colors is None:
        colors = {'CCA': 'blue', 'FBCCA': 'orange', 'TRCA': 'green'}
    if markers is None:
        markers = {'CCA': 'o', 'FBCCA': 's', 'TRCA': 'D'}

    # 创建子图
    plt.figure(figsize=figsize)

    # --- 子图1：准确率（Accuracy）---
    plt.subplot(1, 2, 1)
    for method in methods:
        if method not in results_acc:
            continue  # 如果该方法无准确率数据，则跳过
        accs = np.array(results_acc[method]) * 100  # 转换为百分比
        plt.plot(
            valid_window_starts, accs,
            label=method,
            color=colors.get(method, 'black'),
            marker=markers.get(method, 'o'),
            markersize=5,
            linewidth=2
        )
        # 标出最大准确率点
        best_idx = np.argmax(accs)
        plt.plot(
            valid_window_starts[best_idx], accs[best_idx],
            '*', color='red', markersize=12
        )

    plt.title(f"Accuracy vs. Window Start (Duration={fixed_duration:.1f}s)")
    plt.xlabel("Window Start Time (s)")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    # --- 子图2：信息传输率（ITR）---
    plt.subplot(1, 2, 2)
    for method in methods:
        if method not in results_itr:
            continue  # 如果该方法无 ITR 数据，则跳过
        itrs = np.array(results_itr[method])
        plt.plot(
            valid_window_starts, itrs,
            label=method,
            color=colors.get(method, 'black'),
            marker=markers.get(method, 'o'),
            markersize=5,
            linewidth=2
        )
        # 标出最大 ITR 点
        best_idx = np.argmax(itrs)
        plt.plot(
            valid_window_starts[best_idx], itrs[best_idx],
            '*', color='red', markersize=12
        )

    plt.title(f"ITR vs. Window Start (Duration={fixed_duration:.1f}s)")
    plt.xlabel("Window Start Time (s)")
    plt.ylabel("ITR (bits/min)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # 如果指定了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")

    plt.show()


def visualize_stages_sweep_results(s0, s1, s2, s3, config, save_path=None):
    """
    可视化SSVEP优化实验的结果。

    参数:
        s0 (dict): Stage 0 的结果
        s1 (dict): Stage 1 的结果
        s2 (dict): Stage 2 的结果
        s3 (dict): Stage 3 的结果
        config (object): 配置对象，包含参数如 N_SB_RANGE, A_RANGE, B_RANGE 等。
    """
    fig = plt.figure(figsize=(18, 12))
    plt.suptitle("SSVEP Parameters Optimization", fontsize=16, weight='bold')
    
    # 1. CCA Duration
    ax1 = plt.subplot(2, 3, 1)
    for i, nh in enumerate(s0['harmonics']):
        ax1.plot(
            s0['durations'],
            s0['acc_grid'][:, i] * 100,
            marker='o',
            label=f'Nh={nh}'
        )

    ax1.set_xlabel("Duration (s)")
    ax1.set_ylabel("Acc (%)")
    ax1.set_title("Standard CCA Accuracy")
    ax1.legend(ncol=2, fontsize=9)

    # 2. FBCCA Heatmap
    ax2 = plt.subplot(2, 3, 2)
    opt_N_fb_idx = config.N_SB_RANGE.index(s2['optimal_N'])
    sns.heatmap(s2['results_matrix'][opt_N_fb_idx]*100, ax=ax2, cmap='magma', annot=True, fmt='.1f',
                xticklabels=config.B_RANGE, yticklabels=config.A_RANGE)
    ax2.set_xlabel("b"); ax2.set_ylabel("a"); ax2.invert_yaxis(); ax2.set_title(f"FBCCA (N={s2['optimal_N']})")
    
    # 3. TRCA Heatmap
    ax3 = plt.subplot(2, 3, 3)
    opt_N_tr_idx = config.N_SB_RANGE.index(s3['optimal_N'])
    sns.heatmap(s3['results_stats'][opt_N_tr_idx, :, :, 0]*100, ax=ax3, cmap='viridis', annot=True, fmt='.1f',
                xticklabels=config.B_RANGE, yticklabels=config.A_RANGE)
    ax3.set_xlabel("b"); ax3.set_ylabel("a"); ax3.invert_yaxis(); ax3.set_title(f"TRCA (N={s3['optimal_N']})")
    
    # 4. Bar Chart
    ax4 = plt.subplot(2, 3, 4)
    methods = ['CCA', 'FBCCA', 'TRCA']
    vals = [s1['best_accuracy']*100, s2['best_accuracy']*100, s3['best_accuracy']*100]
    bars = ax4.bar(methods, vals, color=['tab:blue', 'tab:orange', 'tab:green'])
    ax4.set_ylim(0, 105); ax4.set_ylabel("Max Acc (%)"); ax4.set_title("Method Comparison")
    ax4.bar_label(bars, fmt='%.1f')
    
    # 5. Sub-band Sensitivity
    ax5 = plt.subplot(2, 3, 5)
    fb_a = np.where(config.A_RANGE == s2['optimal_a'])[0][0]
    fb_b = np.where(config.B_RANGE == s2['optimal_b'])[0][0]
    tr_a = s3['best_idx'][1]
    tr_b = s3['best_idx'][2]
    
    ax5.plot(config.N_SB_RANGE, s2['results_matrix'][:, fb_a, fb_b]*100, 'o-', label='FBCCA')
    ax5.plot(config.N_SB_RANGE, s3['results_stats'][:, tr_a, tr_b, 0]*100, 's-', label='TRCA')
    ax5.set_xlabel("N Sub-bands"); ax5.set_ylabel("Acc (%)"); ax5.legend(); ax5.set_title("Sub-band Sensitivity")
    
    # 6. ITR
    ax6 = plt.subplot(2, 3, 6)
    itrs = [s1['itrs'][np.argmax(s1['accuracies'])], s2['best_itr'], s3['best_itr']]
    bars2 = ax6.bar(methods, itrs, color=['tab:blue', 'tab:orange', 'tab:green'], alpha=0.6)
    ax6.set_ylabel("ITR (bits/min)"); ax6.set_title("Max ITR")
    ax6.bar_label(bars2, fmt='%.0f')
    
    plt.tight_layout()
    

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"图像已保存至: {save_path}")

    plt.show()