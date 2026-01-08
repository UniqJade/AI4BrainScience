# AI4BrainScience
AI for Brain Science 课程（FDU, 2025 Fall）的代码作业，要求使用机器学习算法分析基于SSVEP的脑机接口数据。

### 作业内容

实验内容：
- 分析一位被试者的`SSVEP`脑机接口数据，完成分类识别

实验要求：
- 本文件已提供了一个基线算法，请在后文补充你自己实现的机器学习算法
- 包含完整的数据分析过程：算法原理、模型训练、模型验证、模型调优、实验结果分析等
- 提供清晰的图表，必要的代码注释，文字说明等
- 数据集包含受试者的6组数据，要求使用6折交叉验证，即单次训练时选其中5组数据作为训练集，剩下的1组作为验证集；重复6次后，最终得到6次验证结果，计算其平均准确率；如果使用的是无须训练的算法，直接统计6组数据的平均准确率即可；

### 数据集说明

- 屏幕上的字符数量为40，分成5行8列，每个字符以不同频率闪烁

- 包含单个受试者的6组数据，每组数据包含40个trial(分别对应40个字符)

- 单个trial说明：屏幕上随机位置闪烁红色光标，引导受试者注视该字符，采样数据记录的是红色光标消失后一小段时间内，受试者持续注视该字符5s的脑电信号

- 数据格式：64个通道，采样频率250Hz，持续时长5s，即包含1250个采样点

<center>
<img src='https://github.com/UniqJade/AI4BrainScience/blob/main/image/BCI_01.jpg' style='width: 600px' />
<img src='https://github.com/UniqJade/AI4BrainScience/blob/main/image/BCI_10_20_system.png' style='width: 450px' />
</center>

<center>
<img src='https://github.com/UniqJade/AI4BrainScience/blob/main/image/BCI_keyboard.jpg' style='width: 500px;margin-right: 20px' />
<img src='https://github.com/UniqJade/AI4BrainScience/blob/main/image/BCI_02.jpg' style='width: 500px;' />
</center>



### 数据

实验数据不公开

### 实验思路

1. 使用线性核的SSVEP进行初步分类，并验证文献中的通道选择是否合理
2. 采用CCA/FBCCA/eTRCA算法进行分类，并进行参数扫描，得到最优参数
3. 若有时间，尝试最新的基于深度学习的模型，比较（正在复现TRCA-net）

所有代码均位于<code>notebook/ssvep.ipynb</code> 中

