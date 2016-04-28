---
layout: post
title: EmbedIDで埋め込みベクトルから単語IDを逆算する
category: Chainer
tags:
- Chainer
- 実装
excerpt_separator: <!--more-->
---

## 概要

- ChainerのEmbedIDを拡張した
- コサイン類似度をもとに埋め込みベクトルから単語IDを求める

<!--more-->

## はじめに

ChainerのEmbedIDは、単語ID（整数値）から対応する埋め込みベクトル（多次元実数ベクトル）を出力します。

今回はこの埋め込みベクトルから単語IDを知るための手法の紹介です。

コードは[embed-id-extended](https://github.com/musyoku/embed-id-extended)に置いてます。

## 使い方

単語ベクトル（最初の軸はバッチ）のVariableを入力します。

```
from embed_id import EmbedID

...

embed = EmbedID(n_ids, ndim_vec)
vec = embed(id)
id = embed.reverse(vec)
```

## 実装

入力ベクトルと、EmbedID.Wのそれぞれのベクトルとのコサイン類似度を計算します。

```
vec = vec.data
W = self.W.data
xp = cuda.get_array_module(*(vec,))
w_norm = xp.sqrt(xp.sum(W ** 2, axis=1))
v_norm = xp.sqrt(xp.sum(vec ** 2, axis=1))
inner_product = W.dot(vec.T)
norm = w_norm.reshape(1, -1).T.dot(v_norm.reshape(1, -1)) + 1e-6
cosine_similarity = inner_product / norm
```
