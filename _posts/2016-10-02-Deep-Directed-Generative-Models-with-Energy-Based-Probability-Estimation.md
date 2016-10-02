---
layout: post
title: Deep Directed Generative Models with Energy-Based Probability Estimation [arXiv:1606.03439]
category: 論文
tags:
- Chainer
excerpt_separator: <!--more-->
---

## 概要

- [Deep Directed Generative Models with Energy-Based Probability Estimation](https://arxiv.org/abs/1606.03439) を読んだ
- Chainer 1.12で実装した

<!--more-->

## はじめに

提案モデル（略してDDGMと呼ぶことにします）は、[Restricted Boltzmann Machine（RBM）](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)（RBM）と同じくエネルギー関数を用いた生成モデルです。

そういったモデルは学習の際、モデルからのサンプリングが必要になるのですが、一般的にはMCMCなどのコストのかかる方法が用いられてきました。

DDGMではエネルギーモデル（Deep Energy Model）とは別に生成モデル（Deep Generative Model）を用い、[Generative Adversarial Networks（GAN）](https://arxiv.org/abs/1406.2661)のアイディアを用いてモデルからのサンプリングをコストの低い伝承サンプリングで行うことができるようになっています。

またDDGMを畳み込みニューラルネットで実装した場合、生成モデル部分は[Deep Convolutional Generative Adversarial Networks（DCGAN）](https://arxiv.org/abs/1511.06434)と同じ働きをします。

論文によるとDDGMはGANよりも優れているそうなので、DCGANよりも綺麗な画像を生成できるかもしれません。（後日記事にします）

