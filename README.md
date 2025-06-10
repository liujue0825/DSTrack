# DSTrack
Official PyTorch implementation of "Cross-Modal Object Tracking via Domain-Adversarial Learning and Switchable Adapter". 

## System Requirements are the same as OSTrack

# 开发版本
架构设计:
1. 1-6层除了主干外还有一个并行的分支，这个并行分支学习模态不变特征。
2. 7-12层除了主干外还有两个可选择的并行的分支，这个可选择并行分支学习模态特有特征。
3. 1-12层主干学习目标的通用表征。

# 待验证
1. 要求blk和self.shared_adapter[i]的反向传播的梯度分离开，也就是说blk会被一个loss优化，而self.shared_adapter[i]会被另一个loss进行优化
2. 这里的1-6层的并行是完全并行开的，只在最后一层进行融合