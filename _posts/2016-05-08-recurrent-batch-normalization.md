---
layout: post
title: Recurrent Batch Normalization [arXiv:1603.09025]
category: Chainer
tags:
- LSTM
excerpt_separator: <!--more-->
---

## 概要

- [Recurrent Batch Normalization](http://arxiv.org/abs/1603.09025) を読んだ
- Chainer 1.8で実装した

<!--more-->

## はじめに

LSTMの内部のゲートやセルにバッチ正規化を適用することで、学習速度が通常のLSTMに比べて早くなります。

## Recurrent Batch Normalization

通常のLSTMは以下の演算により出力値を決定します。

$$
	\begin{align}
		\left(
			\begin{array}{ccc}
			\tilde{ {\boldsymbol {f}} _t} \\
			\tilde{ {\boldsymbol {i}} _t} \\
			\tilde{ {\boldsymbol {o}} _t} \\
			\tilde{ {\boldsymbol {g}} _t} \\
			\end{array}
		\right)
		&=\boldsymbol W_{h}\boldsymbol h_{t-1}+\boldsymbol W_{x}\boldsymbol x_{t}+\boldsymbol b\\
		\boldsymbol c_t&=\sigma(\tilde{ {\boldsymbol {f}} _t}) \odot \boldsymbol c_{t-1}+\sigma(\tilde{ {\boldsymbol {i}} _t}) \odot {\rm tanh}(\tilde{ {\boldsymbol {g}} _t})\\
		\boldsymbol h_t&=\sigma(\tilde{ {\boldsymbol {o}} _t}) \odot {\rm tanh}(\boldsymbol c_t)
	\end{align}
$$

$\tilde{ {\boldsymbol {f}} _t}$は忘却ゲート、$\tilde{ {\boldsymbol {i}} _t}$は入力ゲート、$\tilde{ {\boldsymbol {o}} _t}$は出力ゲート、$\tilde{ {\boldsymbol {g}} _t}$は入力ゲートの入力となります。

このLSTMにバッチ正規化${\rm BN}(x)$を以下のように適用します。

$$
	\begin{align}
		\left(
			\begin{array}{ccc}
			\tilde{ {\boldsymbol {f}} _t} \\
			\tilde{ {\boldsymbol {i}} _t} \\
			\tilde{ {\boldsymbol {o}} _t} \\
			\tilde{ {\boldsymbol {g}} _t} \\
			\end{array}
		\right)
		&={\rm BN}(\boldsymbol W_{h}\boldsymbol h_{t-1})+{\rm BN}(\boldsymbol W_{x}\boldsymbol x_{t})+\boldsymbol b\\
		\boldsymbol c_t&=\sigma(\tilde{ {\boldsymbol {f}} _t}) \odot \boldsymbol c_{t-1}+\sigma(\tilde{ {\boldsymbol {i}} _t}) \odot {\rm tanh}(\tilde{ {\boldsymbol {g}} _t})\\
		\boldsymbol h_t&=\sigma(\tilde{ {\boldsymbol {o}} _t}) \odot {\rm tanh}({\rm BN}(\boldsymbol c_t))
	\end{align}
$$


## 実装

コードは[GitHub](https://github.com/musyoku/recurrent-batch-normalization)にあります。

通常のLSTMはワンパスで出力値を計算できますが、バッチ正規化LSTMはセルの状態決定後にまたバッチ正規化を適用するため、二段階に分けて実装しました。

式(4)

```
lstm_in = self.bnx(self.wx(x))
if self.h is not None:
	lstm_in += self.bnh(self.wh(self.h))
if self.c is None:
	xp = self.xp
	self.c = Variable(xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype), volatile="auto")
```

式(5)

```
self.c = bn_lstm_cell(self.c, lstm_in)
```
式(6)

```
self.h = bn_lstm_state(self.bnc(self.c), lstm_in)
```

## 使い方

従来

```
from chainer import links as L
lstm = L.LSTM(n_in, n_out)
```

バッチ正規化LSTM

```
from bnlstm import BNLSTM
lstm = BNLSTM(n_in, n_out)
```

## 実験

手元にあったツイッターの投稿データ50万文を学習させてみたところ、処理速度は通常版LSTMの方が2倍ほど早かったのですが、同じ時間動かした時の収束の速さはバッチ正規化LSTMの方が早かったです。

実験に用いたコードは[GitHub](https://github.com/musyoku/lstm)にあります。