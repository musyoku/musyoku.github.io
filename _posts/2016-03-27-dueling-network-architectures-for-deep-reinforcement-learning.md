---
layout: post
title: Dueling Network Architectures for Deep Reinforcement Learning [arXiv:1511.06581]
category: 論文
tags:
- Chainer
- 実装
- 論文読み
- 強化学習
- Dueling Network
excerpt_separator: <!--more-->
---

## 概要

- [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581) を読んだ
- Double DQNにDueling Networkを組み込んだ
- DQN・Double DQNと比較した

<!--more-->

**DQN**

```
Conv +--> 512 +--> 512 +--> 4
```

**Double DQN**

```
Conv +--> 512 +--> 512 +--> 4
```

**Double DQN + Dueling Network**

```
                 +--> 256 +--+
Conv +--> 512 +--+           +--> 4
                 +--> 256 +--+
```