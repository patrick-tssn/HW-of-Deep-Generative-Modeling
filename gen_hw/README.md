# HOMEWORK based on StyleCLIP
**代码主要框架来自于[StyleCLIP](https://github.com/patrick-tssn/StyleCLIP)**
## 改进
### encode
我们使用[e4e](https://github.com/omertov/encoder4editing)提供的编码方法，实现了一个编码器为实验所有图像提供了一个映射到StyleGAN$w$空间的编码器。
### 第一种方法添加了id loss.
原代码第一种方法的实现没有加入id loss，我们添加了id loss的代码。

### 稀疏优化

#### 一范数优化
原模型直接使用二范数作为生成图片的损失会使得除了文本关注的地方也会发生较大改变。我们借鉴基追踪的思想，为优化函数添加了一范数惩罚项，使得图片的改动尽可能小。
#### 低秩矩阵分解
我们对生成的图片与原图片的latent code的差值进行低秩矩阵分解，然后对不同尺度的近似矩阵进行实验。

