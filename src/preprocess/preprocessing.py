# src/preprocess/preprocessing.py 

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy import fft
from scipy.stats import pearsonr

# 从目标索引获取频率和相位
def get_freqs(target_idx=None):

    """
    根据目标索引获取 SSVEP 刺激频率列表

    参数:
    - target_idx: 目标索引 (0 到 39 之间的整数)，可选

    返回: 频率数组
    """

    freqs = []
    for offset in np.arange(0.0, 1.0, 0.2):
        for base in np.arange(8.0, 16.0, 1.0):
            freqs.append(round(base + offset, 1))
    if target_idx is not None:
        if not isinstance(target_idx, int) or target_idx < 0 or target_idx >= 40:
            raise ValueError("target_idx 必须是 0 到 39 之间的整数")
        return freqs[target_idx]

    return np.array(freqs)

def get_freq_phase(target_idx=None):
    """
    根据目标索引获取 SSVEP 刺激频率和相位

    参数:
    - target_idx: 目标索引 (0 到 39 之间的整数)

    返回: (频率, 相位) 元组
    """

    # 定义频率和相位的二维数组（5x8）
    frequencies = np.array([
        [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        [8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2],
        [8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4],
        [8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6],
        [8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8]
    ])  # 单位：Hz

    phases = np.array([
        [0.0, 0.5*np.pi, np.pi, 1.5*np.pi, 0.0, 0.5*np.pi, np.pi, 1.5*np.pi],
        [0.5*np.pi, np.pi, 1.5*np.pi, 0.0, 0.5*np.pi, np.pi, 1.5*np.pi, 0.0],
        [np.pi, 1.5*np.pi, 0.0, 0.5*np.pi, np.pi, 1.5*np.pi, 0.0, 0.5*np.pi],
        [1.5*np.pi, 0.0, 0.5*np.pi, np.pi, 1.5*np.pi, 0.0, 0.5*np.pi, np.pi],
        [0.0, 0.5*np.pi, np.pi, 1.5*np.pi, 0.0, 0.5*np.pi, np.pi, 1.5*np.pi]
    ])  # 单位：rad

    if target_idx is None:
        # 返回展平后的全部频率和相位
        return frequencies.flatten(), phases.flatten()
    else:
        if not isinstance(target_idx, int) or target_idx < 0 or target_idx >= 40:
            raise ValueError("target_idx 必须是 0 到 39 之间的整数")
        row = target_idx // 8
        col = target_idx % 8

    freq = frequencies[row, col]
    phase = phases[row, col]

    return freq, phase

def detect_bad_channels(data, threshold=1.5):
    """
    使用 IQR 方法基于通道标准差检测坏通道
    
    参数:
        data: 形状为 (通道数, 时间点数, ...) 的数组
        threshold: IQR 的倍数阈值, 默认 1.5
    返回:
        good_indices: 正常通道的索引列表
        bad_indices: 坏通道的索引列表
    """
    # 1. 将数据展平，把所有试次和轮次拼接成一个长序列（每通道一条时间序列）
    # 新形状: (64, 1250 * 40 * 6) = (64, 300000)
    n_channel = data.shape[0]
    flat_data = data.reshape(n_channel, -1)
    
    # 2. 计算每个通道的标准差
    # 标准差反映该通道信号的“能量”或“活跃程度”
    channel_stds = np.std(flat_data, axis=1)
    
    # 3. 计算四分位数统计量
    Q1 = np.percentile(channel_stds, 25)  # 第一四分位数（25%）
    Q3 = np.percentile(channel_stds, 75)  # 第三四分位数（75%）
    IQR = Q3 - Q1                         # 四分位距（Interquartile Range）
    
    # 4. 定义上下阈值
    # 上界：用于检测高幅值噪声（如肌电、电源干扰）
    upper_bound = Q3 + (threshold * IQR) # 基于数据分布动态计算“高能量异常值”的统计上界
    # 下界：用于检测“死通道”（信号几乎为零，如电极脱落）
    lower_bound = Q1 - (threshold * IQR)  # 基于数据分布动态计算“低能量异常值”的统计下界
    lower_bound = max(lower_bound, 1e-5)  # 防止下界为负数
    
    # 5. 找出坏通道和好通道的索引
    bad_indices = np.where((channel_stds > upper_bound) | (channel_stds < lower_bound))[0]
    good_indices = np.where((channel_stds <= upper_bound) & (channel_stds >= lower_bound))[0]
    
    return good_indices, bad_indices


def apply_spatial_filter_car(data, good_indices=None):
    """
    应用修正后的共平均参考（CAR）：
    仅使用好通道计算平均参考信号，避免坏通道污染。
    
    参数:
        data: 原始数据，形状 (n_channel, ...)
        good_indices: 好通道的索引列表
    返回:
        经过 CAR 处理后的数据，形状与输入相同
    """
    # 1. 仅选取好通道用于计算平均参考
    if good_indices is not None:
        subset_data = data[good_indices, ...] 
    else:
        subset_data = data
    
    # 2. 在正常通道上沿通道维度求平均（保留维度以便广播）
    common_average = np.mean(subset_data, axis=0, keepdims=True)
    
    # 3. 从所有通道（包括坏通道）中减去这个“干净”的平均参考信号
    return data - common_average

def cheby_bandpass_filter(data, lowcut, highcut, fs, order=4, ripple=1):
    """
    应用 Chebyshev 带通滤波器

    参数:
        data: 输入信号，形状为 (n_channels, n_time_points)（多通道）
              或 (n_time_points,)（单通道）
        lowcut: 低截止频率（单位：Hz），即通带下限
        highcut: 高截止频率（单位：Hz），即通带上限
        fs: 采样频率（单位：Hz）
        order: 滤波器阶数，默认为 4 阶；阶数越高，过渡带越陡峭
        ripple: 通带最大波纹（单位：dB），默认为 1 dB  

    返回:
        y: 经零相位 Chebyshev 带通滤波后的信号，形状与输入 data 完全相同
    """

    # 计算奈奎斯特频率（Nyquist frequency），即采样频率的一半
    nyq = 0.5 * fs

    # 将低截止频率归一化到 [0, 1] 区间，其中 1 对应奈奎斯特频率
    low = lowcut / nyq

    # 将高截止频率归一化到 [0, 1] 区间
    high = highcut / nyq

    # 设计一个指定阶数的数字 Chebyshev 带通滤波器
    # 使用 output='sos' 以增强稳定性
    sos = signal.cheby1(order, ripple, [low, high], btype='band', output='sos')
    
    
    # 判断输入信号维度以决定滤波轴：
    # 若为一维数组（单通道信号），直接沿整个数组滤波
    if data.ndim == 1:
        # 使用 filtfilt 实现零相位滤波：先正向滤波，再反向滤波，消除相位延迟
        y = signal.sosfiltfilt(sos, data)
    else:
        # 若为二维数组（如多通道 EEG 数据，shape: [通道数, 时间点数]），
        # 沿时间轴（axis=1）对每个通道独立进行滤波
        y = signal.sosfiltfilt(sos, data, axis=1)
    
    # 返回滤波后的信号，保持原始数据形状不变
    return y


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    应用 Butterworth 带通滤波器

    参数:
        data: 输入信号，形状为 (n_channels, n_time_points)（多通道）
              或 (n_time_points,)（单通道）
        lowcut: 低截止频率（单位：Hz），即通带下限
        highcut: 高截止频率（单位：Hz），即通带上限
        fs: 采样频率（单位：Hz）
        order: 滤波器阶数，默认为 4 阶；阶数越高，过渡带越陡峭

    返回:
        y: 经零相位 Butterworth 带通滤波后的信号，形状与输入 data 完全相同
    """
    # 计算奈奎斯特频率（Nyquist frequency），即采样频率的一半
    # 根据采样定理，可表示的最高频率为 fs/2
    nyq = 0.5 * fs

    # 将低截止频率归一化到 [0, 1] 区间，其中 1 对应奈奎斯特频率
    low = lowcut / nyq

    # 将高截止频率归一化到 [0, 1] 区间
    high = highcut / nyq

    # 设计一个指定阶数的数字 Butterworth 带通滤波器
    # 返回传递函数形式的分子系数 b 和分母系数 a
    # btype='band' 表示带通滤波器；[low, high] 为归一化后的通带边界
    b, a = signal.butter(order, [low, high], btype='band')
    
    # 判断输入信号维度以决定滤波轴：
    # 若为一维数组（单通道信号），直接沿整个数组滤波
    if data.ndim == 1:
        # 使用 filtfilt 实现零相位滤波：先正向滤波，再反向滤波，消除相位延迟
        y = signal.filtfilt(b, a, data)
    else:
        # 若为二维数组（如多通道 EEG 数据，shape: [通道数, 时间点数]），
        # 沿时间轴（axis=1）对每个通道独立进行滤波
        y = signal.filtfilt(b, a, data, axis=1)
    
    # 返回滤波后的信号，保持原始数据形状不变
    return y


def apply_notch_filter(data, freq, fs, quality_factor=30):
    """
    应用陷波滤波器（Notch Filter）去除特定频率的工频干扰（如 50Hz 或 60Hz）。
    
    参数:
        data: 输入数据，形状 (n_channels, n_time_points) 或 (n_time_points,)
        freq: 需要去除的中心频率 (Hz)，例如 50.0
        fs: 采样频率 (Hz)
        quality_factor (Q): 质量因子，决定了滤波器的带宽。
                            Q = 中心频率 / 带宽。
                            Q=30 在 50Hz 时意味着带宽约为 1.7Hz (49.15-50.85Hz)，
                            既能有效去噪又能尽可能少地影响周围频率。
    返回:
        滤波后的数据
    """
    # 设计陷波滤波器
    # iirnotch 返回分子 (b) 和分母 (a) 多项式系数
    b, a = signal.iirnotch(w0=freq, Q=quality_factor, fs=fs)
    
    # 应用滤波器
    if data.ndim == 1:
        y = signal.filtfilt(b, a, data)
    else:
        # 沿时间轴 (axis=-1) 滤波
        y = signal.filtfilt(b, a, data, axis=-1)
        
    return y

# 预处理（Notch + bandpass + CAR）
def preprocess_trial(trial_data, fs, low=7.0, high=90.0, notch_freq=50.0, good_indices=None):
    """
    Notch Filter -> Bandpass Filter -> CAR
    
    参数:
        trial_data: 单个试次的 EEG 数据 (n_channels, n_time_points)
        fs: 采样频率
        low: 带通下限 (Hz)
        high: 带通上限 (Hz)
        notch_freq: 工频干扰频率 (Hz)，默认 50.0 (中国标准)
                    如果设为 None 或 0，则跳过陷波滤波
        good_indices: CAR 使用的好通道索引
        
    返回:
        clean_data: 预处理后的数据
    """

    # 1. 陷波滤波 (Notch) - 针对性去除 50Hz 工频干扰
    if notch_freq is not None and notch_freq > 0:
        notch_data = apply_notch_filter(trial_data, freq=notch_freq, fs=fs)
    else: 
        notch_data = trial_data

    # 2. 带通滤波 (Bandpass) - 保留 SSVEP 频段 (如 8-90Hz)
    filtered_data = cheby_bandpass_filter(notch_data, low, high, fs)

    # 3. 空间滤波 (CAR) - 去除全脑共模噪声
    clean_data = apply_spatial_filter_car(filtered_data, good_indices)
    
    return clean_data

def preprocess_data(data, fs, low=7.0, high=90.0, notch_freq=50.0, good_indices=None):
    
    """
    对整个数据集进行预处理（每个试次应用 preprocess_trial）
    参数:
        data: 原始数据，形状 (n_channels, n_samples, n_targets, n_blocks)
        fs: 采样频率
        low: 带通下限 (Hz)
        high: 带通上限 (Hz)
        notch_freq: 工频干扰频率 (Hz)，默认 50.0
        good_indices: CAR 使用的好通道索引
    
    返回: 
        preprocessed_data: 预处理后的数据，形状与输入 data 相同

    """
    n_channels, n_samples, n_targets, n_blocks = data.shape

    # 建立一个空数组用于存储预处理后的数据
    preprocessed_data = np.zeros_like(data)

    # 遍历每个目标和每个块，逐试次预处理
    for target_idx in range(n_targets):
        for block_idx in range(n_blocks):
            trial_data = data[:, :, target_idx, block_idx]
            clean_data = preprocess_trial(
                trial_data=trial_data,
                fs=fs,
                low=low,
                high=high,
                notch_freq=notch_freq,
                good_indices=good_indices
            )
            # 存储预处理后的数据
            preprocessed_data[:, :, target_idx, block_idx] = clean_data
    return preprocessed_data


def get_spectrum(data, fs):
    """
    使用 FFT 计算频率和振幅

    参数:
        data: 输入信号，形状为 (n_time_points,)
        fs: 采样频率（单位：Hz）

    返回:
        freqs: 频率数组，形状为 (n_freq_points,)
        amps: 对应频率的振幅数组，形状为 (n_freq_points,)
    """
    n = len(data)
    freqs = fft.rfftfreq(n, 1/fs)
    amps = np.abs(fft.rfft(data)) / n * 2 # 归一化振幅
    return freqs, amps

def visualize_ssvep_filtering(raw_data, filtered_data, channel_idx, target_idx, fs=250, duration=5):
    """
    可视化 SSVEP 信号预处理前后的时域和频域对比（2x2 图像）

    参数:
        raw_data: 原始信号，形状 (n_time_points,)
        filtered_data: 预处理后信号，形状 (n_time_points,)
        channel_idx: 通道索引（用于标题显示）
        target_idx: 目标索引（用于获取基频）
        fs: 采样频率（Hz）
        duration: 信号持续时间（秒）
    输出：
        显示 2x2 图像
    """
    # 根据目标索引获取基频
    base_freq = get_freq_phase(target_idx)[0]
    
    points = int(fs * duration)
    time_axis = np.linspace(0, duration, points)

    # 提取单个通道用于绘图
    sig_raw = raw_data[channel_idx, :]
    sig_filt = filtered_data[channel_idx, :]

    # 计算频谱
    freqs, amps_raw = get_spectrum(sig_raw, fs)
    _, amps_filt = get_spectrum(sig_filt, fs)
    mask_freq = (freqs <= 60) # Limit view to 60Hz

    # 创建 2x2 子图布局
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(hspace=0.3)

    # 1. 时域图像 - 原始
    axes[0, 0].plot(time_axis, sig_raw, color='gray', alpha=0.8, label='Raw')
    axes[0, 0].set_title(f'Time Domain: Raw Signal (Ch {channel_idx}, Target {target_idx})')
    axes[0, 0].set_ylabel('Amplitude (uV)')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 时域图像 - 预处理后
    axes[0, 1].plot(time_axis, sig_filt, color='#1f77b4', label='Preprocessed')
    axes[0, 1].set_title(f'Time Domain: Filtered Signal')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 频域图像 - 原始
    axes[1, 0].plot(freqs[mask_freq], amps_raw[mask_freq], color='gray', label='Raw PSD')
    axes[1, 0].set_title('Frequency Domain: Raw')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].axvline(x=base_freq, color='red', linestyle='--', alpha=0.5, label=f'Target {base_freq}Hz')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 频域图像 - 预处理后
    axes[1, 1].plot(freqs[mask_freq], amps_filt[mask_freq], color='#1f77b4', label='Filtered PSD')
    
    # 标记基频及其谐波
    for i in range(1, 5):
        harm_freq = base_freq * i
        if harm_freq <= 60:
            axes[1, 1].axvline(x=harm_freq, color='red', linestyle='--', alpha=0.6, label=f'{harm_freq}Hz' if i==1 else None)

    axes[1, 1].set_title('Frequency Domain: Filtered (Harmonics Check)')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.show()


def calculate_snr_spectrum(amp_spectrum, neighbor_k=5):
    """
    实现 Chen et al., 2015 JNE 论文中的公式：
    SNR = y(f) / 邻近10个点的均值（左侧5个 + 右侧5个）

    参数:
        amp_spectrum: 幅度谱一维数组
        neighbor_k: 单侧邻域点数（默认5）
    返回:
        snr_spectrum: 一维数组（同长度），线性比值（尚未转换为dB）
    """
    n_points = len(amp_spectrum)
    snr = np.zeros_like(amp_spectrum)
    
    # 仅在两侧均有足够邻域点的位置计算SNR
    for i in range(neighbor_k, n_points - neighbor_k):
        signal_amp = amp_spectrum[i]
        
        # 求和邻域点：[i-k ... i-1] 和 [i+1 ... i+k]
        # 注意：明确排除中心点 i
        noise_sum = np.sum(amp_spectrum[i-neighbor_k : i]) + \
                    np.sum(amp_spectrum[i+1 : i+neighbor_k+1])
        
        mean_noise = noise_sum / (2 * neighbor_k)
        
        if mean_noise == 0:
            snr[i] = 0
        else:
            snr[i] = signal_amp / mean_noise
            
    return snr

