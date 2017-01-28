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

