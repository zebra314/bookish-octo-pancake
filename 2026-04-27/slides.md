---
marp: true
theme: default
class: invert
paginate: true
math: mathjax
---

# Weekly Meeting - 3
林穎沛
2026-04-27

---

# Overview

1. New definition
    - [Renyi DP](https://arxiv.org/abs/1702.07476)
    - [Gaussian DP](https://arxiv.org/abs/1905.02383)
2. Tighter accounting
    - [DP-SGD](https://arxiv.org/abs/1607.00133)
    - [Adaptive clipping](https://arxiv.org/abs/1905.03871)
3. Efficiency
    - [Ghost Clipping](https://arxiv.org/abs/2205.10683)
    - [FlashDP](https://arxiv.org/abs/2507.01154)

---

## RDP

---

## DP-SGD

---

## Adaptive clipping

---

## Extra - Dataset Distillation

- [DP-GenG: Differentially Private Dataset Distillation Guided by DP-Generated Data](https://arxiv.org/abs/2511.09876)
    - 不從完全隨機的雜訊圖片開始蒸餾
    - 利用 DP 生成模型產生具備基礎輪廓的圖片來引導
- [Improving Noise Efficiency in Privacy-preserving Dataset Distillation](https://arxiv.org/abs/2508.01749)
    - 將私密資料的訊號抽取出來後，投影到主成分空間
    - 這樣只需提取一次加噪訊號就能重複使用
    
---

## Next week

- explore more papers that deal with how to better estimate the privacy
- Alignment
- Perhaps dataset distillation
