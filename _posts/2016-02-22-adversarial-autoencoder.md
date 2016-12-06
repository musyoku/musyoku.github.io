---
layout: post
title: Adversarial Autoencoders [arXiv:1511.05644]
category: 論文
tags:
- Chainer
- Adversarial Autoencoder
excerpt_separator: <!--more-->
---

## 概要

- [Adversarial Autoencoders](http://arxiv.org/abs/1511.05644) を読んだ
- Chainerで実装した

<!--more-->

## はじめに

Adversarial Autoencoderは、通常のオートエンコーダの隠れ層の出力ベクトル$\boldsymbol z$を、任意の事前分布$p(\boldsymbol z)$に押し込むことで正則化するオートエンコーダです。

パラメータの学習は [Generative Adversarial Networks](http://arxiv.org/abs/1406.2661) の枠組みで行うので、まずGANについて簡単に説明します。

## Adversarial Networks

いま訓練データとして$\boldsymbol x$が与えられ、その生成過程を表す確率分布$p_{data}(\boldsymbol x)$を推定することを考えます。

Adversarial Networksでは、まず任意のノイズ分布$p_z(\boldsymbol z)$を考え、そこからサンプリングしたノイズ$\boldsymbol z$を偽のデータ$\boldsymbol x_{gen}$へ変換する関数$G(\boldsymbol z)$をニューラルネットで定義します。

次に任意のデータ$\boldsymbol x$に対し、それが訓練データ由来の本物のデータ$\boldsymbol x_{real}$なのか、それとも$G$が生成した偽のデータ$\boldsymbol x_{gen}$なのかを識別する$D(\boldsymbol x)$を同様にニューラルネットで定義します。

このような設定のもとで、著者らは$G$と$D$にゲームをさせました。

まず$G$は自らが生成する偽のデータ$\boldsymbol x_{gen}$をより本物のデータに近づけることで$D$を騙そうとします。

一方で$D$は本物のデータと偽物のデータを区別できるように訓練します。

このゲームを式にすると以下のような目的関数になります。

$$
	\begin{align}
		\min_G\max_DV(G,D) = \mathbb E_{\boldsymbol x \sim p_{data}(\boldsymbol x)} [{\rm log}D(\boldsymbol x)]+\mathbb E_{\boldsymbol z \sim p_z(\boldsymbol z)}[{\rm log}(1-D(G(\boldsymbol z)))]
	\end{align}
$$

これを繰り返していくと、$G$は真のデータ分布$p_{data}(\boldsymbol x)$に近づくことができ、結果的に本来の目的であるデータ分布の推定が達成されます。

## Adversarial Autoencoder

Adversarial Autoencoderは基本的には通常のオートエンコーダで、入力$\boldsymbol x$を隠れ変数ベクトル$\boldsymbol z$に符号化したり、逆に$\boldsymbol z$から$\boldsymbol x$に復号化するものですが、以下の点が変更されています。

- $\boldsymbol x$を符号化した$\boldsymbol z_{gen}$を偽のデータと考える。
- 任意の隠れ変数分布$p_z(\boldsymbol z)$からサンプリングした$\boldsymbol z_{real}$を本物のデータと考える。
- オートエンコーダの符号化部分を$G$とみなし、$G(\boldsymbol x)$と表す。
- $D$は$\boldsymbol z$を入力とする新たなニューラルネット$D(\boldsymbol z)$で定義する。
- $D$は与えられた$\boldsymbol z$が$p_z(\boldsymbol z)$由来なのか、それとも$G$が生成した$\boldsymbol z_{gen}$なのかを識別する。

以上のような変更により、このオートエンコーダで$\boldsymbol x$を$\boldsymbol z$に符号化すると、その$\boldsymbol z$は$p_z(\boldsymbol z)$に従うようになります。

パラメータの学習では、復号化誤差を最小化する通常のオートエンコーダとしての学習と、以下の目的関数を用いた2段階の学習を行います。

$$
	\begin{align}
		\min_G\max_DV(G,D) = \mathbb E_{\boldsymbol z \sim p_z(\boldsymbol z)} [{\rm log}D(\boldsymbol z)]+\mathbb E_{\boldsymbol x \sim p_{data}(\boldsymbol x)}[{\rm log}(1-D(G(\boldsymbol x)))]
	\end{align}
$$

GANの時とは$\boldsymbol x$と$\boldsymbol z$が逆になっていることに注意が必要です。

## 実装

$D$を実装する際、初めは出力ユニットを1つにして確率を出力するようにしていましたがうまくいきませんでした。

他の方の実装例を見ていると、ユニットを2つにし片方は真のデータであることを表し、もう片方は偽のデータを表すようにしてソフトマックス層として確率を求めているようでしたので、今回はそのように実装しました。

コードはGitHubにあります。

[adversarial-autoencoder](https://github.com/musyoku/adversarial-autoencoder)

ルートにimagesディレクトリを作成しそこに訓練用データを入れてください。

もしくは実行時にimage_dir引数で指定することもできます。

## 実験

[論文](http://arxiv.org/abs/1511.05644)では、教師あり学習でガウス混合分布とSwiss Roll分布（と著者が呼んでいる分布）に隠れ変数をマッチさせていたので、同様の実験を行いました。

隠れ変数$\boldsymbol z$は今回2次元とします。

まず$p_z(\boldsymbol z)$に従わせるガウス混合分布を以下のようにします。

![10 2D-Gaussian](https://github.com/musyoku/adversarial-autoencoder/blob/master/example/10_2d-gaussian_train_labeled_z.png?raw=true)

それぞれの羽に各数字の隠れ変数$\boldsymbol z$を押し込むことが目的です。

各数字100枚で学習を行い、9,000枚のテストデータの隠れ変数$\boldsymbol z$を可視化したものが以下になります。

![10 2D-Gaussian](https://github.com/musyoku/adversarial-autoencoder/blob/master/example/10_2d-gaussian_test_labeled_z.png?raw=true)


次にSwiss Roll分布は以下の様な分布になっています。

Swiss Roll（ロールケーキ）の名前通り渦を巻いた分布で、中心に近い部分から順に数字の0, 1, 2, ...を押し込みます。

![Swiss Roll](https://github.com/musyoku/adversarial-autoencoder/blob/master/example/swiss_roll_train_labeled_z.png?raw=true)

ガウス混合分布の時と同様に各数字100枚で学習を行い、9,000枚のテストデータの隠れ変数$\boldsymbol z$を可視化したものが以下になります。

![Swiss Roll](https://github.com/musyoku/adversarial-autoencoder/blob/master/example/swiss_roll_test_labeled_z.png?raw=true)

## 余談

Chainer初心者なのでラベル情報の付加のやりかたがわからなかったため、隠れ変数ベクトルの次元を10拡張してone-hotなラベルで置き換えるやり方をしていますが、正しいやり方はどうなんでしょう。

（追記）

[ChainerでVariableにラベル情報を付加する方法](/2016/03/25/ChainerでVariableにラベル情報を付加する方法/)