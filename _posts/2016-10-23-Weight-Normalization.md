---
layout: post
title: Weight Normalization [arXiv:1602.07868]
category: 論文
tags:
- Chainer
excerpt_separator: <!--more-->
---

## 概要

- [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/abs/1602.07868)を読んだ
- Chainer 1.17で実装した（→[GitHub](https://github.com/musyoku/weight-normalization)）

<!--more-->

## はじめに

Weight Normalizationは、ニューラルネットの重みベクトル$\boldsymbol w$を

$$
	\begin{align}
		\boldsymbol w=\frac{g}{\mid\mid \boldsymbol v \mid\mid}\boldsymbol v
	\end{align}\
$$

のように、ベクトル$\boldsymbol v$とスカラー$g$に分解します。

また、これらのパラメータで誤差関数$L$を微分した時の勾配はそれぞれ

$$
	\begin{align}
		\nabla_gL &= \frac{\nabla_{\boldsymbol w}L\circ \boldsymbol v}{\mid\mid \boldsymbol v \mid\mid}\\
		\nabla_{\boldsymbol v}L &= \frac{g}{\mid\mid \boldsymbol v \mid\mid}\nabla_{\boldsymbol w}L-
			\frac{g\nabla_gL}{\mid\mid \boldsymbol v \mid\mid^2}\boldsymbol v\\
	\end{align}\
$$

となります。

ここで$\circ$は要素積の総和を表します。```numpy.dot```ではありませんのでご注意下さい。

## 実装

実装する際はchainerのfunctionを継承したクラスを作成し、式(2)の$\nabla_{\boldsymbol w}L$を親クラスのfunctionに計算させます。

その後自クラスで$\nabla_gL$と$\nabla_{\boldsymbol v}L$を計算するだけで実装は完了です。

またWeight Normalizationは**重みベクトル**に対して行う操作のようなので、**重み行列**を用いるchainerのLinkでは$g$はスカラーではなく出力ユニット数の次元を持ったベクトルになります。

畳み込みニューラルネットの場合は出力チャネルごとに$\boldsymbol v$があるとみなして$g$の次元数は出力チャネル数と同じにするようです。

## Data-Dependent Initialization

Weight Normalizationではパラメータ$g$やニューラルネットのバイアス$b$をデータから決まる値で初期化します。

私の実装では最初にデータを入力したときに自動的に初期化されるようにしてあるので、学習を始める前に一度訓練データを入力してください。

## 終わりに

最近のGenerative Adversarial Networkは、DiscriminatorにWeight Normalizationを使う例が多かったため今回作成しました。

一応テストも書き、（というかchainerのLink用テストをコピペして精度を少し落としただけ）ちゃんと通るのを確認したのでバグはないと思います。