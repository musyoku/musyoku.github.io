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

提案モデル（略してDDGMと呼ぶことにします）は、[Restricted Boltzmann Machine（RBM）](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)と同じくエネルギー関数を用いた生成モデルです。

そういったモデルは学習の際、モデルからのサンプリングが必要になるのですが、一般的にはMCMCなどのコストのかかる方法が用いられてきました。

DDGMではエネルギーモデル（Deep Energy Model）とは別に生成モデル（Deep Generative Model）を用い、[Generative Adversarial Networks（GAN）](https://arxiv.org/abs/1406.2661)のアイディアを用いてモデルからのサンプリングをコストの低い伝承サンプリングで近似することができるようになっています。

またDDGMを畳み込みニューラルネットで実装した場合、生成モデル部分は[Deep Convolutional Generative Adversarial Networks（DCGAN）](https://arxiv.org/abs/1511.06434)と同じ働きをします。

論文によるとDDGMはGANよりも優れているそうなので、DCGANよりも綺麗な画像を生成できるかもしれません。

## モデル

DDGMでは訓練データ$\boldsymbol x$の尤度はボルツマン分布で表されます。

$$
	\begin{align}
		P_{\Theta}(\boldsymbol x) &= \frac
			{
				e^{-E_{\Theta}(\boldsymbol x)}
			}
			{
				Z_{\Theta}
			}\\
		Z_{\Theta} &= \sum_{\boldsymbol x}e^{-E_{\Theta}(\boldsymbol x)}
	\end{align}\
$$

$E_{\Theta}(\boldsymbol x)$はエネルギー関数、$Z_{\Theta}$は正規化項（分配関数）です。

またエネルギー関数は"expert"と呼ばれる項の和になっています。

$$
	\begin{align}
		E_{\Theta}(\boldsymbol x) = \sum_{i}\tilde{E}_{\theta_i}(\boldsymbol x)
	\end{align}\
$$

### Deep Energy Model

DDGMでは入力$\boldsymbol x$をそのままエネルギー関数$E_{\Theta}(\boldsymbol x)$に渡すのではなく、特徴抽出器（feature extractor）$f_{\psi}$を用いて特徴量を取り出し、それをエネルギー関数に入力します。

$E_{\Theta}(\boldsymbol x)$は単層ネットワークですが、$f_{\psi}(\boldsymbol x)$はCNNでも全結合でも良く、深いネットワークで構成されます。

これらを用いてDeep Energy Modelのエネルギー関数を以下のように定義します。

$$
	\begin{align}
		E_{\Theta}(\boldsymbol x) = E_{\Theta'}(\boldsymbol x, f_{\psi}(\boldsymbol x)) = 
		\frac{1}{\sigma^2}\boldsymbol x^T\boldsymbol x-\boldsymbol b^T\boldsymbol x - 
		\sum_i {\rm log}(1+e^{W_i^Tf_{\psi}(\boldsymbol x) + c_i})
	\end{align}\
$$

また各expertは

$$
	\begin{align}
		\tilde{E}_{\theta_i}(f_{\psi}(\boldsymbol x)) = -{\rm log}(1+e^{W_i^Tf_{\psi}(\boldsymbol x) + c_i})
	\end{align}\
$$

となります。

論文で$b_i$となっている部分は$c_i$の間違いだと思います。（$\boldsymbol b^T$と$b_i$は別物のはずです）

### Deep Generative Model

生成モデル部分は隠れ変数$\boldsymbol z$をとる生成関数$G_{\phi}(\boldsymbol z)$になっています。

$\boldsymbol z$は一様分布などからサンプリングしますが、$\boldsymbol x$の生成モデルからのサンプリングは$\boldsymbol x = G_{\phi}(\boldsymbol z)$と決定的に決まります。

## 学習