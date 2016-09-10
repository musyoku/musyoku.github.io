---
layout: post
title: Auxiliary Deep Generative Models [arXiv:1602.05473]
category: 論文
tags:
- Chainer
- ADGM
excerpt_separator: <!--more-->
---

## 概要

- [Auxiliary Deep Generative Models](http://arxiv.org/abs/1602.05473) を読んだ
- Chainer 1.12でADGMとSDGMを実装した

## はじめに

Auxiliary Deep Generative Models(ADGM)は半教師ありのMNISTのクラス分類(100 labels)において、現在世界最高精度のエラー率0.96%を達成したモデルです。

また著者自身による[Lasagne実装](https://github.com/larsmaaloee/auxiliary-deep-generative-models)が公開されていますので、試したい方はそちらを利用することもできます。

## モデル

ADGMは[VAE](/2016/07/02/semi-supervised-learning-with-deep-generative-models/)に補助変数$a$を入れて拡張したものになっています。

VAEのM1+M2と同じくADGMも隠れ変数は２つ（$\boldsymbol z$と$\boldsymbol a$）ですが、依存関係が大きく異なっています。

ADGMでは生成モデルと推論モデルはそれぞれ

$$
	\begin{align}
		p_{\boldsymbol \theta}(\boldsymbol a, \boldsymbol x, y, \boldsymbol z) &= p_{\boldsymbol \theta}(\boldsymbol a \mid \boldsymbol x,y,\boldsymbol z)p_{\boldsymbol \theta}(\boldsymbol x\mid y,\boldsymbol z)p(y)p(\boldsymbol z)\\
		q_{\boldsymbol \phi}(\boldsymbol a, y, \boldsymbol z \mid \boldsymbol x) &= q_{\boldsymbol \phi}(\boldsymbol a \mid \boldsymbol x)q_{\boldsymbol \phi}(y \mid \boldsymbol a, \boldsymbol x)q_{\boldsymbol \phi}(\boldsymbol z \mid \boldsymbol a,\boldsymbol x, y)
	\end{align}\
$$

と定義します。

$\boldsymbol \theta$と$\boldsymbol \phi$はニューラルネットのパラメータを表します。

VAEのM1+M2モデルとの比較を以下に示します。

![ADGMとVAEの比較](/images/post/2016-09-10/adgm_vae.png)

VAEは2つのモデルが積み重なっており、M1をあらかじめ学習させてからM2を学習する必要があるのですが、ADGMではend-to-endで学習を行うことが可能です。

また論文ではSkip Deep Generative Model (SDGM)と呼ばれるもう一つのモデルが提案されています。

これは以下に示す通り、推論モデルの矢印を逆にしたものになっています。

![ADGMとSDGM](/images/post/2016-09-10/adgm_model.png)

## 実装

ADGMとSDGMはともに以下に示す5つのニューラルネットから構成されています。

![ADGMとSDGM](/images/post/2016-09-10/adgm_arch.png)

### 誤差関数

VAEの時と同様のやり方で変分下限にマイナスをかけたものを誤差とします。

まずラベルありの場合のデータの対数尤度は

$$
	\begin{align}
		{\rm log}p(\boldsymbol x,y) &= {\rm log}\int_{\boldsymbol a}\int_{\boldsymbol z}p(\boldsymbol a, \boldsymbol x, y, \boldsymbol z)d\boldsymbol z d\boldsymbol a\\
		&\geq \double E_{\boldsymbol a, \boldsymbol z \sim q_{\phi}(\boldsymbol a, \boldsymbol z \mid \boldsymbol x, y)}\left[{\rm log}\frac{p_{\boldsymbol \theta}(\boldsymbol a,\boldsymbol x, y,\boldsymbol z)}{q_{\boldsymbol \phi}(\boldsymbol a, \boldsymbol z \mid \boldsymbol x, y)}\right]\\
		&=\double E_{\boldsymbol a, \boldsymbol z \sim q_{\phi}(\boldsymbol a, \boldsymbol z \mid \boldsymbol x, y)}\left[
		{\rm log}\frac{
			p_{\boldsymbol \theta}(\boldsymbol a \mid \boldsymbol x,y,\boldsymbol z)p_{\boldsymbol \theta}(\boldsymbol x\mid y,\boldsymbol z)p(y)p(\boldsymbol z)
		}{
			q_{\boldsymbol \phi}(\boldsymbol a \mid \boldsymbol x)q_{\boldsymbol \phi}(\boldsymbol z \mid \boldsymbol a,\boldsymbol x, y)
		}\right]\\
		&\equiv -{\cal L}(\boldsymbol x, y)
	\end{align}\
$$

となります。

${\cal L}(\boldsymbol x, y)$はラベルありの場合の誤差関数です。

次にラベルがない場合のデータの対数尤度は

$$
	\begin{align}
		{\rm log}p(\boldsymbol x) &= {\rm log}\sum_y\int_{\boldsymbol a}\int_{\boldsymbol z}p(\boldsymbol a, \boldsymbol x, y, \boldsymbol z)d\boldsymbol z d\boldsymbol a\\
		&\geq \double E_{\boldsymbol a, y, \boldsymbol z \sim q_{\phi}(\boldsymbol a, y, \boldsymbol z \mid \boldsymbol x)}\left[{\rm log}\frac{p_{\boldsymbol \theta}(\boldsymbol a,\boldsymbol x, y,\boldsymbol z)}{q_{\boldsymbol \phi}(\boldsymbol a, \boldsymbol z \mid \boldsymbol x, y)}\right]\\
		&=\double E_{\boldsymbol a, y, \boldsymbol z \sim q_{\phi}(\boldsymbol a, y, \boldsymbol z \mid \boldsymbol x)}\left[
		{\rm log}\frac{
			p_{\boldsymbol \theta}(\boldsymbol a \mid \boldsymbol x,y,\boldsymbol z)p_{\boldsymbol \theta}(\boldsymbol x\mid y,\boldsymbol z)p(y)p(\boldsymbol z)
		}{
			q_{\boldsymbol \phi}(\boldsymbol a \mid \boldsymbol x)q_{\boldsymbol \phi}(\boldsymbol z \mid \boldsymbol a,\boldsymbol x, y)q_{\phi}(y \mid \boldsymbol a, \boldsymbol x)
		}\right]\\
	\end{align}\
$$

となります。

ここで

$$
	\begin{align}
		f(\cdot) = {\rm log}\frac{p_{\boldsymbol \theta}(\boldsymbol a,\boldsymbol x, y,\boldsymbol z)}{q_{\boldsymbol \phi}(\boldsymbol a, \boldsymbol z \mid \boldsymbol x, y)}
	\end{align}\
$$

とおくと、ラベル無しデータの対数尤度は(9)式をさらに変形し

$$
	\begin{align}
		{\rm log}p(\boldsymbol x) &= {\rm log}\sum_y\int_{\boldsymbol a}\int_{\boldsymbol z}p(\boldsymbol a, \boldsymbol x, y, \boldsymbol z)d\boldsymbol z d\boldsymbol a\\
		&\geq \double E_{\boldsymbol a, y, \boldsymbol z \sim q_{\phi}(\boldsymbol a, y, \boldsymbol z \mid \boldsymbol x)}
		\left[
			f(\cdot) - {\rm log}q_{\phi}(y \mid \boldsymbol a, \boldsymbol x)
		\right]\\
		&= \double E_{\boldsymbol a \sim q_{\phi}(\boldsymbol a \mid \boldsymbol x)}
		\Biggl[
			\sum_y
			\biggl\{
				q_{\phi}(y \mid \boldsymbol a, \boldsymbol x)
				\bigl\{
					\double E_{\boldsymbol z \sim q_{\phi}(\boldsymbol a \mid \boldsymbol x)}[f(\cdot)]
					- {\rm log}q_{\phi}(y \mid \boldsymbol a, \boldsymbol x)
				\bigr\}
			\biggr\}
		\Biggr]\\
		&= \double E_{\boldsymbol a \sim q_{\phi}(\boldsymbol a \mid \boldsymbol x)}
		\Biggl[
			\sum_y 
			\biggl\{
				q_{\phi}(y \mid \boldsymbol a, \boldsymbol x)\double E_{\boldsymbol z \sim q_{\phi}(\boldsymbol a \mid \boldsymbol x)}[f(\cdot)]
			\biggr\}
			+
			\underbrace{
				\sum_y
				\biggl\{
					q_{\phi}(y \mid \boldsymbol a, \boldsymbol x)
					\{
						- {\rm log}q_{\phi}(y \mid \boldsymbol a, \boldsymbol x)
					\}
				\biggr\}
			}_{ {\cal H}(q_{\phi}(y \mid \boldsymbol a, \boldsymbol x)) }
		\Biggr]\\
		&= \double E_{\boldsymbol a \sim q_{\phi}(\boldsymbol a \mid \boldsymbol x)}
		\left[
			\sum_y
			\biggl\{
				q_{\phi}(y \mid \boldsymbol a, \boldsymbol x)
					\double E_{\boldsymbol z \sim q_{\phi}(\boldsymbol a \mid \boldsymbol x)}[f(\cdot)]
			\biggr\}
			+
			{\cal H}(q_{\phi}(y \mid \boldsymbol a, \boldsymbol x))
		\right]\\
		&\equiv -{\cal U}(\boldsymbol x)
	\end{align}\
$$

となります。

${\cal U}(\boldsymbol x)$はラベルありの場合の誤差関数です。

### モンテカルロサンプリング

上記の${\cal L}(\boldsymbol x, y)$や${\cal U}(\boldsymbol x)$は直接計算するのが困難なのでサンプリングにより近似します。

サンプリング回数を$N_{MC}$とすると、

$$
	\begin{align}
		\boldsymbol a^{(l)} &\sim q_{\boldsymbol \phi}(\boldsymbol a \mid \boldsymbol x)\\
		\boldsymbol z^{(l)} &\sim q_{\boldsymbol \phi}(\boldsymbol z^{(l)} \mid \boldsymbol a^{(l)},\boldsymbol x, y)\\
		{\cal L}(\boldsymbol x, y) &\simeq \frac{1}{N_{MC}}
		\sum_{l=1}^{N_{MC}}
			{\rm log}\frac{
				p_{\boldsymbol \theta}(\boldsymbol a^{(l)} \mid \boldsymbol x,y,\boldsymbol z^{(l)})p_{\boldsymbol \theta}(\boldsymbol x\mid y,\boldsymbol z^{(l)})p(y)p(\boldsymbol z^{(l)})
			}{
				q_{\boldsymbol \phi}(\boldsymbol a^{(l)} \mid \boldsymbol x)q_{\boldsymbol \phi}(\boldsymbol z^{(l)} \mid \boldsymbol a^{(l)},\boldsymbol x, y)
			}\\
		{\cal U}(\boldsymbol x) &\simeq \frac{1}{N_{MC}}
		\sum_{l=1}^{N_{MC}}
		\biggl\{
			\sum_y
				\bigl\{
					q_{\phi}(y \mid \boldsymbol a^{(l)}, \boldsymbol x)f(\cdot)
				\bigr\}
				+
				{\cal H}(q_{\phi}(y \mid \boldsymbol a^{(l)}, \boldsymbol x))
		\biggr\}
	\end{align}\
$$

