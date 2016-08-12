---
layout: post
title: Adversarial AutoEncoderで半教師あり学習
category: 論文
tags:
- Chainer
- 実装
- 論文読み
- Adversarial Autoencoder
excerpt_separator: <!--more-->
---

## 概要

- [Adversarial Autoencoders](http://arxiv.org/abs/1511.05644) を読んだ
- Chainer 1.12で実装した

<!--more-->

## はじめに

前回[Adversarial AutoEncoderの記事](/2016/02/22/adversarial-autoencoder/)を書きましたが、あれから3ヶ月後に論文が更新され、半教師あり学習への応用などの項目が追加されていました。

そこでChainer 1.12で実装し実験を行いました。

コードは[GitHub](https://github.com/musyoku/adversarial-autoencoder)にあります。

以下の説明は全てこのコードを元に行います。

またAdversarial AutoEncoderをこれ以降AAEと呼びます。

## Adversarial Regularization

論文2.3節では隠れ変数$\boldsymbol z$をラベルデータを用いて正則化する方法について書かれています。

コードは`supervised/regularize_z`にあります。

学習させると以下のように狙った箇所に隠れ変数を押し込めます。

![aae](/images/post/2016-08-09/supervised/regularize_z/labeled_z_10_gaussian.png)

![aae](/images/post/2016-08-09/supervised/regularize_z/labeled_z_swiss_roll.png)

## Supervised Adversarial Autoencoders

論文4章ではAAEに画像のスタイルを学習させています。

ここでは隠れ変数$\boldsymbol z$を入力画像のスタイルとみなし、ラベルデータと$\boldsymbol z$の両方をAAEのデコーダーに入力し画像を複合します。

コードは`supervised/learn_style`にあります。

学習結果は以下のようになりました。

![aae](/images/post/2016-08-09/supervised/learn_style/analogy.png)

学習時間やチューニングが足りていないのであまり良い結果とはいえませんが、数字のスタイルがちゃんと取れているように思えます。

## Semi-Supervised Adversarial Autoencoders 

論文5章はAAEを用いた半教師あり学習です。

これは[前回のVAEによる半教師あり学習](/2016/07/02/semi-supervised-learning-with-deep-generative-models/)と同じく、MNISTのごく一部のデータ（たとえば100枚）だけ教師あり学習を行い、それ以外のデータでは教師なし学習を行います。

コードは`semi-supervised/classification`にあります。

実際に100枚だけにラベルをつけた半教師あり学習を行った際のバリデーションデータの分類精度が以下になります。

![aae](/images/post/2016-08-09/semi_supervised.png)

論文では97%を超える精度が出ると書かれていますが残念ながら遠く及ばない結果になってしまいました。

私の実装に不備があるのかチューニングのせいなのかが分かりません。

## Unsupervised Clustering with Adversarial Autoencoders

論文6章は教師なしクラスタリングです。

本来MNISTのクラスタ数は10ですが、ここでは16クラスタ（または32クラスタ）あると考えて入力画像を分類します。

コードは`unsupervised/clustering`にあります。

#### 学習結果

![aae](/images/post/2016-08-09/unsupervised/clustering/clusters_16.png)

![aae](/images/post/2016-08-09/unsupervised/clustering/clusters_32.png)

一番左の画像はcluster headを表しています。

これも学習時間が足りてないせいかあまり良い結果とは言えませんが、そこそこクラスタリングされているように思えます。

## Dimensionality Reduction with Adversarial Autoencoders

論文7章は次元削減です。

普通のオートエンコーダも次元削減を行うモデルですが、ここでは6章の内容を拡張し、クラスタを考慮した次元削減を行います。

コードは`unsupervised/dim_reduction`と`semi-supervised/dim_reduction`にあります。

まず半教師あり学習による次元削減の結果です。

#### 100ラベル

![aae](/images/post/2016-08-09/semi-supervised/dim_reduction/labeled_z_100.png)

#### 1000ラベル

![aae](/images/post/2016-08-09/semi-supervised/dim_reduction/labeled_z_1000.png)

次に教師なし学習で20クラスタを仮定した次元削減です。

![aae](/images/post/2016-08-09/unsupervised/dim_reduction/labeled_z.png)

## 終わりに

学習に数日かけないと良い結果が出ない印象です。