# **Neural Manifold Operators** (KDD 2024)

**Neural Manifold Operators for Learning the Evolution of Physical Dynamics** [[paper]](https://doi.org/10.1145/3637528.3671779)

We propose Neural Manifold Operator (NMO), an operator learning paradigm for learning the intrinsic dimension representation of the underlying operator.

<p align="center">
<img src=".\fig\fig2.png" height = "280" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overview of NMO.
</p>

**Intrinsic dimension representation:** NMO learns the intrinsic dynamics of the physics system by learning the invariant subspace of the underlying infinite-dimensional operators with intrinsic dimension.

**Generic operator learning paradigm:** NMO is a generic operator learning paradigm for various network structure implementations including **Multi-Layer Perceptron, Convolutional Neural Network, and Transformer**.

**Benefits in multi-disciplinary areas:** NMO achieves **state-of-the-art** performance in several **real-world and equation-governed scenarios** (Fig 2), ranging from mathematics, physics, chemistry and earth science.

**Efficiency and Accuracy:** By intrinsic dimension projection, NMO significantly reduces the training parameters and effectively improves the capability of generalization and physical consistency.

<p align="center">
<img src=".\fig\fig1.png" height = "220" alt="" align=center />
<br><br>
<b>Figure 2.</b> The experiment scenarios of NMO.
</p>

## Get Started

Our complete code will be released in early September.

## Experiments Result

### 1. Main Result

**State-of-the-art performance.** We compare our model with nine baseline models in seven scenarios. NMO achieves **state-of-the-art performance** in statistical and physical metrics and gains **23.35\% average improvement** on three real-world scenarios and four equation-governed scenarios across a wide range of multi-disciplinary fields.

<p align="center">
<img src=".\fig\table1.png" height = "450" alt="" align=center />
<br><br>
</p>

**The best physical performance.** NMO achieves the best performance in physical metrics without specific inductive bias.

<p align="center">
<img src=".\fig\fig4.png" height = "350" alt="" align=center />
<br><br>
<b>Figure 3.</b> Left: Relative mass error at each time step and visualization of prediction results of each model on the Shallow-Water equations scenario. Mid: Turbulence energy spectrum on the Rayleigh-Bénard convection scenario. Right: The average of absolute divergence convection at each time step and RMSE associated with the prediction step of each model on the Rayleigh-Bénard convection scenario.
</p>

**Accuracy and efficiency.** Our model achieves the best balance among training speed, parameter size, and performance.

<p align="center">
<img src=".\fig\fig3.png" height = "280" alt="" align=center />
<br><br>
<b>Figure 4.</b> The training time and RMSE performance rankings of various models on SEVIR and Navier-Stokes equation scenario.
</p>

### 2. Dimension Experiments

It is shown that the intrinsic dimension calculated by our paradigm is the optimal dimensional representation of the underlying operators.

<p align="center">
<img src=".\fig\table2.png" height = "320" alt="" align=center />
<br><br>
</p>

We have implemented it across various scenarios, further demonstrating the effectiveness of our algorithm.

<p align="center">
<img src=".\fig\fig5.png" height = "400" alt="" align=center />
<br><br>
<b>Figure 5.</b> The prediction performance of various dimensions of the time evolution operator. The dotted lines represent the ID calculated by our algorithm in each scenario.
</p>

## Citation

If our paper or code is helpful to your academic research, please cite our paper.

```
@inproceedings{Wu2024NMO,
author = {Wu, Hao and Weng, Kangyu and Zhou, Shuyi and Huang, Xiaomeng and Xiong, Wei},
title = {Neural Manifold Operators for Learning the Evolution of Physical Dynamics},
year = {2024},
isbn = {9798400704901},
publisher = {Association for Computing Machinery},
url = {https://doi.org/10.1145/3637528.3671779},
doi = {10.1145/3637528.3671779},
booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {3356–3366},
numpages = {11},
series = {KDD '24}
}
```

## Contact

If you have any questions about our paper or code, please contact Hao Wu (wuhao2022@mail.ustc.edu.cn), Wei Xiong (xiongw21@mails.tsinghua.edu.cn; wei.xiong@yale.edu) or any author of this paper.
