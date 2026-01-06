# src/config.py
import numpy as np

# ===============
# SSVEP 数据集配置
# ===============

class Config:
    # 实验参数                   
    FS = 250                      # 采样频率（Hz）
    CHANNELS = [47, 53, 54, 55, 56, 57, 60, 61, 62]       # 优化后的通道选择
    N_TARGETS = 40                # SSVEP 目标刺激数量
    N_BLOCKS = 6                  # 每个目标的实验块数（block 数）                      
    GAZE_SHIFT = 0.50             # 眼动移位
    WINDOW_START = 0.14           # 起始时间（秒）
    TRIAL_DURATION = 5.0          # 每个 trial 的持续时间（秒）
    SAMPLE_POINTS = int(FS * TRIAL_DURATION)  # 每个 trial 的采样点数
    
    # 滤波参数
    BANDPASS_LOW = 7.0            # 带通滤波低频截止（Hz）
    BANDPASS_HIGH = 90.0          # 带通滤波高频截止（Hz）
    NOTCH_FREQ = 50.0             # 工频陷波频率（Hz）
    NOTCH_Q = 30.0                # 陷波器品质因数
    
    # 分析参数
    MAX_DURATION = TRIAL_DURATION - WINDOW_START  # 最大可用持续时间（秒）
    DURATION_MIN = 0.2                            # 最小持续时间（秒）
    DURATION_STEP = 0.2                           # 持续时间步长（秒）
    DURATION_RANGE = None                         # 初始化后赋值
    
    FIXED_DURATION = 1.25         # 根据 Chen et al.(2015)  的研究优化得出
    
    NH_RANGE = list(range(1, 11))                 # 谐波数量范围
    A_RANGE = np.arange(0, 2.25, 0.25)            # 权重参数 a 范围
    B_RANGE = np.arange(0, 1.25, 0.25)            # 权重参数 b 范围
    N_SB_RANGE = list(range(1, 11))               # 子带数量范围
    
    @classmethod
    def initialize_duration_range(cls):
        cls.DURATION_RANGE = np.arange(cls.DURATION_MIN, 
                                       cls.MAX_DURATION - 1.0, 
                                       cls.DURATION_STEP)