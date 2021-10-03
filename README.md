# Negative Sampling for NER

## 运行
+ 训练： bash run_train.sh
+ 预测：预测暂时封装为http接口，请根据需要自己修改

## 项目说明
本项目为[原项目地址](https://github.com/LeePleased/NegSampling-NER)的改进版本：
1. 修改解码方案，将片段分类+负采样方案用于嵌套实体识别。[此处](https://zhuanlan.zhihu.com/p/378684128)为详细介绍。
2. 优化显存占用，源代码直接使用指针标注网络，空间复杂度为$O(L^2)$，实验发现在显存10G内的设备上，对文本长度限制很不友好。
本项目将片段分类修改为分堆并行，空间复杂度仅由BERT决定。
3. 片段分类增加相对位置编码。原方案的片段分类仅有BERT嵌入层的绝对位置编码，本项目通过加入相对位置编码，提升实体识别的准确率。

## 权利声明
1. 上述改进优化点作者保留原创权利。

## 原项目Citation
```
@inproceedings{li2021empirical,
    title={Empirical Analysis of Unlabeled Entity Problem in Named Entity Recognition},
    author={Yangming Li and lemao liu and Shuming Shi},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=5jRVa89sZk}
}
```
