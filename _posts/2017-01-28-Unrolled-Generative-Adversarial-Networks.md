---
layout: post
title: Unrolled Generative Adversarial Networks [arXiv:1611.02163]
category: 実装
tags:
- Chainer
excerpt_separator: <!--more-->
---

## 概要

- [Unrolled Generative Adversarial Networks](https://arxiv.org/abs/1611.02163)を読んだ
- Chainerで実装した

<!--more-->

## はじめに

Unrolled GANはGeneratorの安定性を高め、"mode collapse"を回避するための手法です。

Discriminatorに関しては何もしません。

実装も非常にシンプルです。

## Unrolling GANs

Generatorのパラメータを$\theta_G$、Discriminatorのパラメータを$\theta_D$とします。

またGANの目的関数はよく使われる以下を用います。

$$
	\begin{align}
		f(\theta_G, \theta_D) &= \double E_{x \sim p_{data}} \left[{\rm log}(D(x;\theta_D)) \right]+
		\double E_{z \sim {\cal N}(0, I)} \left[{\rm log}(1-D(G(z;\theta_G);\theta_D)) \right]\\
	\end{align}\
$$

この時、現在のGeneratorに絶対騙されない理想的なDiscriminatorのパラメータ$\theta_D^*$は以下のようにパラメータ更新を繰り返すことで得られます。

$$
	\begin{align}
		\theta_D^{(0)} &\gets \theta_D\\
		\theta_D^{(k+1)} &\gets \eta^{(k)}\frac{df(\theta_G, \theta_D^{(k)})}{d\theta_D^{(k)}} \\
		\theta_D^* &= \lim_{k \to \infty}\theta_D^{(k)} \\
	\end{align}\
$$

あくまで現在のGeneratorに対するDiscriminatorのパラメータですので、$\theta_G$は値を固定しておきます。

このように$\theta_D^{(k+1)}$を求める操作を"unrolling"と呼び、式(4)は無限回のunrollingを表しています。

次に$K$回のunrollingで得られる$\theta_D^{(K)}$に対して、Generatorの目的関数を以下のように定義します。

$$
	\begin{align}
		f_K(\theta_G, \theta_D^{(K)}) &= \double E_{x \sim p_{data}} \left[{\rm log}(D(x;\theta_D^{(K)})) \right]+
		\double E_{z \sim {\cal N}(0, I)} \left[{\rm log}(1-D(G(z;\theta_G);\theta_D^{(K)})) \right]\\
	\end{align}\
$$

$K=0$とすれば通常のGANの目的関数に一致するため、Unrolled GANは通常のGANを特殊なケースとして含みます。

## パラメータ更新

現在の各パラメータを$\theta_G \gets \theta_G^{(0)}, \theta_D \gets \theta_D^{(0)}$とします。

$\theta_G, \theta_D$の更新は以下の通りです。

まず上記の$K$ステップのunrollingを行ない$\theta_D^{(1)}, \theta_D^{(2)}, ..., \theta_D^{(K-1)}, \theta_D^{(K)}$を得ます。

Generatorはこの$K$ステップだけ先に強くなってしまったDiscriminatorに対し以下のようにパラメータを更新します。

$$
	\begin{align}
		\theta_G^{(1)} &\gets \eta^{(0)}\frac{df(\theta_G^{(0)}, \theta_D^{(K)})}{d\theta_G^{(0)}} \\
		\theta_G &\gets \theta_G^{(1)}
	\end{align}\
$$

最後にDiscriminatorを以下のように更新します。

$$
	\begin{align}
		\theta_D \gets \theta_D^{(1)}
	\end{align}\
$$

注意点として、$\theta_D$は最終的に$\theta_D^{(1)}$になり、$\theta_D^{(K)}$にはなりません。

この部分を間違えてしまうとGeneratorはDiscriminatorに勝てなくなり学習に失敗します。

Unrolled GANのこのような更新手法は、未来のDiscriminatorをGeneratorが先取りして対策しておくようなイメージです。

## 実装

Chainer 1.20で実装しました。

[https://github.com/musyoku/unrolled-gan](https://github.com/musyoku/unrolled-gan)

計算の流れは以下のようになります。（$K=2$）

![image](/images/post/2017-01-29/unrolled_gan.png)

実装も既存のGANがあれば十数行追加するだけでです。

私は重みの退避と再設定機能だけ追加しました。

```
def cache_discriminator_weights(self):
	self.cached_weights = {}
	xp = self.xp
	optimizer = self.discriminator.optimizer
	for name, param in optimizer.target.namedparams():
		with cuda.get_device(param.data):
			self.cached_weights[name] = xp.copy(param.data)

def restore_discriminator_weights(self):
	optimizer = self.discriminator.optimizer
	for name, param in optimizer.target.namedparams():
		with cuda.get_device(param.data):
			if name not in self.cached_weights:
				raise Exception()
			param.data = self.cached_weights[name]
```

パラメータの学習ですが、まず$K$回Discriminatorを更新します。

この時1回目の更新で得られた重みをコピーして保存しておきます。

その後Generatorを更新してから保存したDiscriminatorの重みで現在のDiscriminatorの重みを上書きします。

## 実験

今回の手法はGANの拡張と言うよりは、従来のGANが今回の手法の特殊なケースだっというものなので、すでに実装しているGANをほとんど変えずに使うことができます。

そこで前回実装した[Improved Techniques for Training GANs](/2016/12/23/Improved-Techniques-for-Training-GANs/)のタスクをそのままUnrolled GANで行いました。

## Mixture of Gaussians Dataset

まず論文で主張されている、Unrolled GANは"mode collapse"が起こりにくい、ということを確認します。

用いるデータは平均をずらした正規分布の混合分布から生成させたデータです。

Generatorの入力ノイズは論文通りに256次元とします。

1つめのデータは8つの正規分布の混合分布で以下のような形をしています。

![image](/images/post/2017-01-29/circle_true.png)

左が散布図で右がKDE（Kernel Density Estimation）です。

学習結果は以下のようになりました。

![image](/images/post/2017-01-29/circle_256.png)

上の2段が通常のGANで下の2段がUnrolled GANです。

通常のGANは学習に失敗し、最頻値（mode）のうちの1つに集まってしまいました。

これが"mode collapse"と呼ばれる現象だと思います。

それに対して5回のunrollingを行うGANでは学習初期から生成点が分散し、目標分布をうまく捉えることができています。

次に以下のような10個の正規分布の混合分布のデータを用いてGANを学習させました。

![image](/images/post/2017-01-29/double_true.png)

結果は以下のとおりです。

![image](/images/post/2017-01-29/double_256.png)

このデータでもUnrolled GANは安定して目標分布を捉えています。

ただしこの実験には一つ疑問があり、データ$x$が2次元なのに対しノイズ$z$が256次元なのは多すぎであると考えられます。

そこで$z$を16次元にして上記2つのデータセットで再実験した結果が以下になります。

![image](/images/post/2017-01-29/circle_16.png)

![image](/images/post/2017-01-29/double_16.png)

通常のGANが謎の粘りを見せ最終的には両者とも分布を正しく捉えることができました。

しかしここでも通常GANは不安定な部分があるのに対し、Unrolled GANは学習初期から安定していることがわかります。


## MNISTの半教師あり学習

前回の[Improved Techniques for Training GANs](/2016/12/23/Improved-Techniques-for-Training-GANs/)では、MNISTの半教師あり学習をGANでやるとほぼワンショット学習でも高い認識精度が得られることが分かりました。

そこでUnrolled GANで同様にMNISTの半教師あり学習を行ないました。


