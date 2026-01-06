# src/visualization/ml_visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_accuracy_and_confusion(accuracies, conf_matrices, mean_acc, title_prefix=""):
    """绘制准确率柱状图和聚合混淆矩阵"""
    plt.figure(figsize=(16, 6))

    # Accuracy per block
    plt.subplot(1, 2, 1)
    plt.bar(range(len(accuracies)), [a * 100 for a in accuracies], color="#3B68D9", edgecolor='black')
    plt.axhline(mean_acc * 100, color='r', linestyle='--', 
                label=f'Mean: {mean_acc*100:.1f}%')
    plt.xlabel('Test Block')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{title_prefix}Accuracy with Bandpass & Spatial Filtering')
    plt.ylim(0, 105)
    plt.legend()

    # Aggregated Confusion Matrix
    plt.subplot(1, 2, 2)
    total_conf_matrix = np.sum(conf_matrices, axis=0)
    sns.heatmap(total_conf_matrix, annot=False, cmap='Blues', cbar=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{title_prefix}Aggregated Confusion Matrix')

    plt.tight_layout()
    plt.show()

def plot_sensitivity_analysis(names, trains, tests):
    """
    绘制不同通道方案下的训练/测试准确率对比柱状图
    """
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 绘制两组柱状图：训练 vs 测试
    rects1 = ax.bar(x - width/2, trains, width, label='Train (Internal CV)', color='lightgray', hatch='//')
    rects2 = ax.bar(x + width/2, tests, width, label='Test (Generalization)', color='#1f77b4')

    # 图表样式设置
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Channel Sensitivity Analysis: The Impact of Dimensionality')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 115) 
    ax.grid(axis='y', alpha=0.3)

    # 在柱子顶部显示具体数值
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.show()