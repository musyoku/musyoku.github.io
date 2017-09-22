---
layout: post
title: Training RNNs as Fast as CNNs
category: 実装
tags:
- Chainer
excerpt_separator: <!--more-->
---

## 概要

- [Training RNNs as Fast as CNNs](https://arxiv.org/abs/1709.02755)を読んだ
- Simple Recurrent Unitを実装した

<!--more-->

## はじめに

この論文ではRNNの高速な実装方法を提案しています。

LSTMに比べて5倍から10倍程度高速化できます。

## Simple Recurrent Unit（SRU）

SRUの各ゲートの更新式は以下の通りです。

$$
  \begin{align}
  	\boldsymbol{\tilde{x}}_t &= \boldsymbol{W}\boldsymbol{x}_t\\
  	\boldsymbol{f}_t &= \sigma(\boldsymbol{W}_f\boldsymbol{x}_t + \boldsymbol{b}_f)\\
  	\boldsymbol{r}_t &= \sigma(\boldsymbol{W}_r\boldsymbol{x}_t + \boldsymbol{b}_r)\\
  	\boldsymbol{c}_t &= \boldsymbol{f}_t \odot \boldsymbol{c}_{t-1} + (1 - \boldsymbol{f}_t) \odot \boldsymbol{\tilde{x}}_t\\
  	\boldsymbol{h}_t &= \boldsymbol{r}_t \odot g(\boldsymbol{c}_t) + (1 - \boldsymbol{r}_t) \odot \boldsymbol{x}_t\\

  \end{align}\
$$

