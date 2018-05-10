# 说明
从语料中自动抽取词库

1. 构建ngrams

2. 使用min_freq、SSR算法对ngrams剪枝，取得有效的候选词

3. 计算候选词的外部边界值 Exterior Boundary Values
    EBV(w) = LBV(w) ∗ RBV(w)

4. 计算候选词的内部边界值 Interior Boundary Value
    IBV(w) = min(MI(x:y), ...)
    MI(x:y) = log2( p(x,y) / (p(x)*p(y)) )

5. 计算候选词的wordrank
    WR(w) = EBV(w) ∗ f(IBV(w)) = LBV(w) ∗ RBV(w)∗ f(IBV(w))

6. 根据wordrank排序，取得词库

# 使用
```python
from wordrank.wordrank import build

build('data/data.txt', 'data/words.csv')
```

# 参考
- 《A Simple and Effective Unsupervised Word Segmentation Approach》
- 《Statistical Substring Reduction in Linear Time》