---
layout: post
title: Auto-Encoding Variational Bayes [arXiv:1312.6114]
category: Chainer
tags:
- 変分ベイズ
excerpt_separator: <!--more-->
---

## 概要

- [Auto-Encoding Variational Bayes](http://arxiv.org/abs/1312.6114) を読んだ

<!--more-->

## はじめに

最近名前をよく聞くようになった変分オートエンコーダの基礎となる**確率的勾配変分ベイズ（SGVB）**について自分なりにまとめます。

## 参考

- [20150717-suzuki.pdf](http://deeplearning.jp/wp-content/uploads/2014/04/20150717-suzuki.pdf)
	- SGVB
- [vb-nlp-tutorial.pdf](http://chasen.org/~daiti-m/paper/vb-nlp-tutorial.pdf)
	- 変分下限

## 問題設定

データを$\boldsymbol x$、隠れ変数を$\boldsymbol z$、パラメータを$\boldsymbol \theta$とし、同時確率分布$p_{\boldsymbol \theta}(\boldsymbol x, \boldsymbol z) = p_{\boldsymbol \theta}(\boldsymbol x\mid\boldsymbol z)p_{\boldsymbol \theta}(\boldsymbol z)$を推定します。

周辺尤度$p_{\boldsymbol \theta}(\boldsymbol x) = \int p_{\boldsymbol \theta}(\boldsymbol x\mid\boldsymbol z)p_{\boldsymbol \theta}(\boldsymbol z)d\boldsymbol z$が計算困難な場合、$p_{\boldsymbol \theta}(\boldsymbol x)$は$\boldsymbol \theta$で微分できないため、直接$\boldsymbol \theta$を最適化することはできません。

また事後分布$p_{\boldsymbol \theta}(\boldsymbol z\mid\boldsymbol x) = p_{\boldsymbol \theta}(\boldsymbol x\mid\boldsymbol z)p_{\boldsymbol \theta}(\boldsymbol z)/p_{\boldsymbol \theta}(\boldsymbol x)$も困難であり、EMアルゴリズムを使えません。

さらにデータが大量にあるので、時間のかかるサンプリングベースな手法は使いたくありません。

本論文ではこれらの問題を解決するための手法を提案します。

## 認識モデル

上記の問題の解決に関して、認識モデル$q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)$を導入します。

これは真の事後分布$p_{\boldsymbol \theta}(\boldsymbol z\mid\boldsymbol x)$の近似であり、パラメータ$\boldsymbol \phi$は$\boldsymbol \theta$と同時に学習させます。

## 変分下限（変分下界）

$\boldsymbol z$の対数周辺尤度は以下のように変形できます。

$$
	\begin{align}
		{\rm log}p_{\boldsymbol \theta}(\boldsymbol x) &= log\int p_{\boldsymbol \theta}(\boldsymbol x\mid \boldsymbol z)p_{\boldsymbol \theta}(\boldsymbol z)d\boldsymbol z\\
		&= log\int q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)\frac{p_{\boldsymbol \theta}(\boldsymbol x\mid \boldsymbol z)p_{\boldsymbol \theta}(\boldsymbol z)}{q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)}d\boldsymbol z\\
		&\geq\int q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x){\rm log}\frac{p_{\boldsymbol \theta}(\boldsymbol x\mid \boldsymbol z)p_{\boldsymbol \theta}(\boldsymbol z)}{q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)}d\boldsymbol z\\
		&=\int q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)
		\biggl\{
		{\rm log}\frac{p_{\boldsymbol \theta}(\boldsymbol z)}{q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)}+{\rm log}p_{\boldsymbol \theta}(\boldsymbol x\mid \boldsymbol z)
		\biggr\}d\boldsymbol z\\
		&=\int q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x){\rm log}p_{\boldsymbol \theta}(\boldsymbol x\mid\boldsymbol z)d\boldsymbol z-\int q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x){\rm log}\frac{q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)}{p_{\boldsymbol \theta}(\boldsymbol z)}d\boldsymbol z\\
		&=\double E_{\boldsymbol z \sim q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)}[{\rm log}p_{\boldsymbol \theta}(\boldsymbol x\mid\boldsymbol z)] - D_{KL}(q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)||p_{\boldsymbol \theta}(\boldsymbol z))
	\end{align}
$$

式(2)から(3)への変形にはイェンセンの不等式を用います。

$\double E[\cdot]$は期待値、$D_{KL}$はKLダイバージェンスを表します。

式(6)は${\rm log}p_{\boldsymbol \theta}(\boldsymbol x)$の下限値を表しているため、これを増加させる$\boldsymbol \theta$を探せばよいことになります。

### 第一項について

$\double E_{\boldsymbol z \sim q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)}[{\rm log}p_{\boldsymbol \theta}(\boldsymbol x\mid\boldsymbol z)]$は、あるデータ$\boldsymbol x^{(i)}$があるときに、$q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)$からサンプリングした$\boldsymbol z^{(i)}$を用いて、$p_{\boldsymbol \theta}(\boldsymbol x\mid\boldsymbol z)$からサンプリングして得られた$\boldsymbol x'$が、もとの$\boldsymbol x^{(i)}$である度合いを表しています。

つまり、$\boldsymbol x \to \boldsymbol z \to \boldsymbol x$のようなオートエンコーダとしてうまく機能する度合いを表しているため、この項は復号誤差と呼ばれたりします。

### 第二項について

$D_{KL}(q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)\|\|p_{\boldsymbol \theta}(\boldsymbol z))$は常に0以上の値を取ります。

これは$q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)$が事前分布$p_{\boldsymbol \theta}(\boldsymbol z)$からどれだけ離れているかを表しています。つまり、この項が0になれば、$q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)$は$p_{\boldsymbol \theta}(\boldsymbol z)$に一致します。

$q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)$を、$\boldsymbol x$に無関係な$\boldsymbol z$の事前分布$p_{\boldsymbol \theta}(\boldsymbol z)$に近づけることで過学習を防ぐことができると考えられるので、この項は正則化項と呼ばれています。

### 第二項について


