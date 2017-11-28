---
layout: post
title: Face Alignment at 3000 FPS via Regressing Local Binary Features
category: 実装
tags:
- 機械学習
excerpt_separator: <!--more-->
---

## 概要

- [Face Alignment at 3000 FPS via Regressing Local Binary Features](https://pdfs.semanticscholar.org/d59f/b96a60168f2baec6f5c61b82393576c33fb7.pdf)を読んだ
- C++で実装した

<!--more-->

## はじめに

Dlibには目や鼻、口などの顔特徴点を検出する手法が実装されています。

[Real-Time Face Pose Estimation](http://blog.dlib.net/2014/08/real-time-face-pose-estimation.html)

<iframe width="560" height="315" src="https://www.youtube.com/embed/oWviqXI9xmI?rel=0" frameborder="0" allowfullscreen></iframe>

この機能は[One Millisecond Face Alignment with an Ensemble of Regression Trees](https://pdfs.semanticscholar.org/d78b/6a5b0dcaa81b1faea5fb0000045a62513567.pdf)の実装になっており、興味があったので関連研究を調べてみました。

この分野の研究は数多く行われていますが、今回はタイトルに惹かれたので[Face Alignment at 3000 FPS via Regressing Local Binary Features](https://pdfs.semanticscholar.org/d59f/b96a60168f2baec6f5c61b82393576c33fb7.pdf)を実装しました。

日本語で読める参考文献として[顔特徴点検出における形状回帰モデルの適応的設計](https://www.jstage.jst.go.jp/article/itej/69/3/69_J126/_article/-char/ja/)や[関東CV勉強会20140802（Face Alignment at 3000fps）](https://www.slideshare.net/tackson5/cv20140802face-alignment-at-3000fps)があります。

## 形状回帰モデル

One Millisecondや3000 FPSはともに形状回帰モデルによる特徴点検出を行っています。

これは以下のように平均顔形状から開始して、各点について画像特徴を元に移動量を推定し、点を繰り返し動かしていくことで予測顔形状を求める手法です。（これをCascaded Regressionといいます）

3000 FPSでは数回の繰り返しになりますが、One Millisecondなど他の手法では数百回の繰り返しを行うこともあります。

この繰り返しをステージと呼びます。

注意として、形状回帰モデルを用いた検出は、現時点の検出点に対する適切な移動量を推定するものであり、検出点の適切な位置を直接推定するものではありません。

学習時には現時点の検出点と正解形状から正しい移動量（正確には$x$軸と$y$軸それぞれの移動量）が分かるので、その値を回帰するために形状回帰モデルと呼ばれています。

顔形状は複数個の点から構成されますが、点の個数はデータセット作成者によって決められています。

今回用いる300-Wデータセットは68点なので、これ以降顔形状を$\boldsymbol S \in \double R^{68 \times 2}$と表します。

実装時には2次元配列を用いると便利です。

## Shape-Indexed Features



## Local Binary Features

本手法では、画像上のある2点の輝度値の差を画像特徴量として、ランダムフォレストによりそれぞれの検出点の移動量を決定します。

ある1つの検出点の移動量を推定するためには、まずランダムフォレストのそれぞれの木のルートから始め、画像特徴量に基づいて左右どちらの子ノードに降りるかを決定します。

この操作を葉ノード（つまり子ノードを持っていない終端ノード）に到達するまで行い、最後に到達した葉ノードが持つ移動量の値を推定結果として返します。

推定の流れは以上の通りなんですが、初見では何を言っているのかが全く分からないので詳しく説明します。

まず画像特徴量として2点間の輝度値の差というただのスカラーを用いることについて、たった一つの値から適切な推定を行うのは困難であることが容易に想像できます。

そのため、複数の位置の輝度値の差を組み合わせて複雑化させ、表現力を高める必要があります。

このような輝度値の組み合わせを特徴量として扱うために、本論文ではランダムフォレストを用います。

ランダムフォレストの木を構成するノードはそれぞれ、画像から輝度値を計算するために必要な2点の位置を保持しています。

仮に木が完全二分木で深さが7であるとします。

この場合ノードは127個あるため、輝度値を計算するための2点の位置が127通り存在します。

さらに各ノードは閾値を持っており、そのノードが持つ2点の位置を元に計算した輝度値が、その閾値を上回っている場合に右の子ノードに進み、下回っている場合に左の子ノードに進みます。