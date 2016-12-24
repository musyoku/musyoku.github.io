---
layout: post
title: Improved Techniques for Training GANs [arXiv:1606.03498]
category: 論文
tags:
- Chainer
excerpt_separator: <!--more-->
---

## 概要

- [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)を読んだ
- Chainer 1.18で実装した
- アニメ顔画像を学習させた
- MNISTの半教師あり学習を実験した

<!--more-->

## はじめに

この論文はGANによる画像生成と半教師あり学習の2つに焦点を当て、新たな学習テクニックを提案したものです。

この記事ではそのテクニックの中からfeature matchingとminibatch discriminationを実装します。

さらに多クラス分類器をDiscriminatorとして使うテクニックを用いてMNISTの半教師あり学習を行ないます。

以下、訓練データを$x$、Generatorが生成したデータを$\bar{x}$とします。

またDiscriminatorを$D(x)$、ノイズ$z$から$\bar{x}$を生成するGeneratorを$G(z)$と表記します。

GANでは$x$を本物のデータ、$\bar{x}$を偽物のデータを考え、Discriminatorが偽物のデータを見破れるように学習を行います。

## Feature matching

Feature matchingは、Discriminatorに$x$と$\bar{x}$を入力した時のそれぞれの中間層出力の二乗誤差を小さくすることでGeneratorがより本物に近いデータを生成できるようにするテクニックです。

実装する時は出力層に一番近い中間層出力（活性化関数を通した後の値）をマッチさせれば良いと思います。

## Minibatch discrimination

このテクニックはGeneratorにより多様性のある画像を生成させるためのものです。

論文の"collapse"という現象が具体的にどういうものか想像できなかったので読んでもあまり理解できなかったのですが、極端なことを言うとGeneratorは「Discriminatorが最も本物だと考える画像」を1つ生成できれば勝てるので、どのようなノイズ$z$からも似たような画像が生成されるということなんでしょうか。

実際、以前に実験したGANライクなモデルである[Deep Directed Generative Models with Energy-Based Probability Estimation](/2016/10/28/Deep-Directed-Generative-Models-with-Energy-Based-Probability-Estimation/)では、以下のような訓練画像で学習した際に、

![image](/images/post/2016-10-28/kb_original.png)

学習に失敗するとランダムな100個のノイズからこのような画像が生成されました。

![image](/images/post/2016-10-28/kb_fail.png)

平均的な画像を生成すれば最低限Discriminatorを騙せると学習してしまったようなのですが、これも"collapse"の一種だと考えて良いでしょう。

このような現象を抑えるためには、Discriminatorがミニバッチ中のデータの多様性を知れる仕組みが必要であり、多様性を増大させるような方向の勾配をGeneratorに伝播すれば、Generatorはより多種多様な画像を生成するようになる可能性があります。

そこでMinibatch discriminationでは、ミニバッチ中のあるデータ$x_i$をDiscriminatorに入力した時の中間層出力（特徴ベクトル）$f(x_i)$に対して、残りのデータ$x_{-i}$の特徴ベクトル$f(x_{-i})$全てとのノルムを計算し、これを$f(x_i)$に付加して上位層に伝播します。

この計算はミニバッチ中のデータそれぞれについて求める必要がありますが、論文ではテンソルを用いて「ミニバッチの多様性を表すベクトル」を計算する手法を提案しています。

まず特徴ベクトル$f(x_i) \in \double R^A$にテンソル$T \in \double R^{A \times B \times C}$を掛けることで行列$M_i \in \double R^{B \times C}$が得られます。

この行列は$C$次元のベクトルが$B$個あると解釈します。

$M_i$の$b$行目のベクトルを$M_{i,b}$で表し、以下の3つの量を定義します。

$$
	\begin{align}
		c_b(x_i, x_j) &= exp(-\mid\mid M_{i,b} - M_{j, b} \mid\mid _{L1}) \\
		o(x_i)_b &= \sum_{k=1}^{n} c_b(x_i, x_k) \\
		o(x_i) &= \left[o(x_i)_1, o(x_i)_2, ... , o(x_i)_B \right] \\
	\end{align}\
$$

（$n$はミニバッチに含まれるデータ数）



