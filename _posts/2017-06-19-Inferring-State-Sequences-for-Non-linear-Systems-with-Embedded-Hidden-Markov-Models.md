---
layout: post
title:  Inferring State Sequences for Non-linear Systems with Embedded Hidden Markov Models
category: 実装
tags:
- HMM
excerpt_separator: <!--more-->
---

## 概要

- [Inferring State Sequences for Non-linear Systems with Embedded Hidden Markov Models](https://papers.nips.cc/paper/2391-inferring-state-sequences-for-non-linear-systems-with-embedded-hidden-markov-models)を読んだ
- C++で実装した

<!--more-->

## はじめに

Embedded HMMは状態空間が連続のモデルにおいて、観測系列から隠れ状態系列をforward-backwardアルゴリズムによってサンプリングするための手法です。

無限に存在する状態を離散化し有限個の候補を各時刻ごとに作成することで、通常のHMMと同様に状態系列をサンプリングできるようになります。

以前に実装した[無限木構造HMM](/2017/03/09/無限木構造隠れMarkovモデルによる階層的品詞の教師なし学習/)の展望として今回のEmbedded HMMを適用することが挙げられていたので、まずEmbedded HMMがどのようなものか調べました。

## The Embedded HMM Algorithm

状態空間を$\cal X$、観測空間を$\cal Y$、初期状態の分布を$p(x_0)$、遷移確率を$p(x_t \mid x_{t-1})$、出力確率を$p(y_t \mid x_t)$とします。

我々の目的は観測系列から状態系列をサンプリングすることです。

$$
  \begin{align}
    \boldsymbol x \sim p(x_0, ..., x_{n-1} \mid y_0, ..., y_{n-1})
  \end{align}\
$$

この条件付き確率を$\pi(\boldsymbol x)$とします。

$$
  \begin{align}
    \pi(\boldsymbol x) \equiv p(x_0, ..., x_{n-1} \mid y_0, ..., y_{n-1})
  \end{align}\
$$

$\pi(\boldsymbol x)$から直接サンプリングすることができないので、$\pi(\boldsymbol x)$を不変分布とするマルコフ連鎖$\boldsymbol x^{(0)}, \boldsymbol x^{(1)}, ...$を考えます。

MCMCについては[マルコフ連鎖入門](http://ebsa.ism.ac.jp/ebooks/sites/default/files/ebook/1881/pdf/vol3_ch10.pdf)が詳しいです。

このマルコフ連鎖の遷移確率を$Q(\boldsymbol x^{(i)} \mid \boldsymbol x^{(i-1)})$とすると、詳細釣り合い条件は以下のように書けます。

$$
  \begin{align}
    \pi(\boldsymbol x)Q(\boldsymbol x' \mid \boldsymbol x) = \pi(\boldsymbol x')Q(\boldsymbol x \mid \boldsymbol x'),\text{ for all } \boldsymbol x \text{ and } \boldsymbol x' \text{ in }{\cal X}^n
  \end{align}\
$$

Embedded HMMでは$Q(\boldsymbol x^{(i)} \mid \boldsymbol x^{(i-1)})$から直接サンプリングするのではなく、プールと呼ばれる候補状態の集合から状態を取ってくることでサンプリングの代わりをします。

このプールにはマルコフ連鎖での現在の状態$x_t^{(i)}$が必ず入っており、残りの候補状態はプール分布$\rho_t$から生成します。

ただしこの$\rho_t$もどういう分布なのかがわからないので、$\rho_t$を不変分布とするまた別のマルコフ連鎖を考えます。

このマルコフ連鎖の遷移確率を$R_t(\cdot \mid \cdot)$、逆方向への遷移確率を$\tilde{R}_t(\cdot \mid \cdot)$とし、これらが以下の条件を満たすように定義します。

$$
  \begin{align}
    \rho_t(x_t)R_t(x_t' \mid x_t) = \rho_t(x_t')\tilde{R}_t(x_t \mid x_t'),\text{ for all }x_t\text{ and }x_t'\text{ in }{\cal X}^n
  \end{align}\
$$

論文で"inner" Markov chainと書かれていますが、上記のように$Q(\boldsymbol x^{(i)} \mid \boldsymbol x^{(i-1)})$によるマルコフ連鎖の中に$R_t(\cdot \mid \cdot)$によるマルコフ連鎖が組み込まれていることを意味しています。

$R_t$が詳細釣り合い条件を満たしている場合、式(4)から$R_t$と$\tilde{R}_t$は同じです。

この$R_t(x' \mid x)$は表記上$x$から次の候補状態を生成しそうに見えるのですが、現在の状態$\boldsymbol x^{(i)}$から候補状態を生成してはいけません。

これは後で説明しますが、式(3)の詳細釣り合い条件が満たされなくなるためです。

そのため候補状態の生成の際には$\rho_t$から独立にサンプリングしたり、観測$\boldsymbol y$からサンプリングすると良いそうです。

（この"inner"なマルコフ連鎖はもはや連鎖ではありません。）