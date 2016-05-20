---
layout: post
title: Semi-Supervised Learning with Deep Generative Models [arXiv:1406.5298]
category: Chainer
tags:
- VAE
excerpt_separator: <!--more-->
---

## 概要

- [Semi-Supervised Learning with Deep Generative Models](http://arxiv.org/abs/1406.5298) を読んだ
- Chainer 1.8で実装した
- モデルM1、M2、M1+M2の実装方法の解説
- モデルM2で100ラベルのエラー率9%を達成した

<!--more-->

## はじめに

Variational AutoEncoder(VAE)は、半教師あり学習に用いることのできるオートエンコーダです。

学習のベースとなる確率的勾配変分ベイズ(SGVB)については[以前の記事](/2016/04/29/auto-encoding-variational-bayes/)をお読みください。

この論文では3つのVAEのモデル、M1、M2、M1+M2が提案されています。

### M1

M1は教師なし学習のためのモデルです。

Chainerでの実装も多く見られ、公式サンプルにも追加されました。

### M2

M2は半教師なし学習のためのモデルです。

MNISTを用いた場合、50000枚の訓練画像のうち、たった100枚にだけ正解ラベルを与え、それ以外の画像では正解ラベルを与えない学習を行っても、クラス分類精度が90%を超えます。

このM2の実装は現時点で著者のKingma氏によるTheano実装しか公開されておらず、論文もやや説明不足な部分があり難易度が高いです。

### M1+M2

このモデルはM1を教師なし学習させ、画像から隠れ変数$z$を出力させます。

その後M2で$z$を用いた半教師あり学習を行ます。

MNISTで100枚だけに正解ラベルを与えた半教師あり学習でも、クラス分類精度が97%を超える結果が出ます。

## コード

すべての実装は[GitHub](https://github.com/musyoku/variational-autoencoder)にあります。

## M1の実装

以下、入力画像を$\boldsymbol x$、隠れ変数を$\boldsymbol z$とします。両方ともベクトルです。

生成モデルを以下のように定義します。

$$
	\begin{align}
		p(\boldsymbol z) &= {\cal N}(\boldsymbol z \mid \boldsymbol 0, \boldsymbol 1)\\
		p_{\theta}(\boldsymbol x \mid \boldsymbol z) &= f(\boldsymbol x;\boldsymbol z,\boldsymbol \theta)\\
		p(\boldsymbol x, \boldsymbol z) &= p_{\theta}(\boldsymbol x \mid \boldsymbol z)p(\boldsymbol z)
	\end{align}\
$$

$f(\boldsymbol x;\boldsymbol z,\boldsymbol \theta)$は$\boldsymbol z$の関数なので尤度関数と呼びます。

これには正規分布やベルヌーイ分布が用いられます。

$x$が画像の場合、画素値は$[0,1]$の実数を取りますが、これは確率値とみなすことができるのでベルヌーイ分布が用いられます。

$\theta$はニューラルネットのパラメータを表します。

${\cal N}(\boldsymbol z \mid \boldsymbol 0, \boldsymbol 1)$は平均が0、分散が1の正規分布です。

また$\boldsymbol z$の真の事後分布$p(\boldsymbol z \mid \boldsymbol x)$の近似である$q_{\phi}(\boldsymbol z \mid \boldsymbol x)$を以下のように定義します。

$$
	\begin{align}
		q_{\phi}(\boldsymbol z \mid \boldsymbol x) ={\cal N}(\boldsymbol z \mid \boldsymbol \mu_{\phi}(\boldsymbol x), {\rm diag}(\boldsymbol \sigma^2_{\phi}(\boldsymbol x)))
	\end{align}\
$$

$\boldsymbol \mu_{\phi}(\boldsymbol x)$と$\boldsymbol \sigma^2_{\phi}(\boldsymbol x)$がニューラルネットであり、$\boldsymbol x$を入力するとそれぞれ$\boldsymbol z$の平均と分散を出力します。

VAEはオートエンコーダであり、符号化を$q_{\phi}(\boldsymbol z \mid \boldsymbol x)$が行い入力を隠れ変数に符号化します。

隠れ変数の入力への復号化には$p_{\theta}(\boldsymbol x \mid \boldsymbol z)$を用います。

### 目的関数

VAEの目的は$\boldsymbol z$の対数周辺尤度${\rm log}p_{\boldsymbol \theta}(\boldsymbol x)$を最大化することです。

これはつまり、訓練データ$\boldsymbol x$を与える$\boldsymbol z$として尤もらしいものを求めることです。

[以前の記事](/2016/04/29/auto-encoding-variational-bayes/)にも載せていますが、イェンゼンの不等式を用いて以下のように変形することで、${\rm log}p_{\boldsymbol \theta}(\boldsymbol x)$の下限値を求めることができます。

$$
	\begin{align}
		{\rm log}p_{\boldsymbol \theta}(\boldsymbol x) &= log\int p_{\boldsymbol \theta}(\boldsymbol x\mid \boldsymbol z)p_{\boldsymbol \theta}(\boldsymbol z)d\boldsymbol z\nonumber\\
		&= log\int q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)\frac{p_{\boldsymbol \theta}(\boldsymbol x\mid \boldsymbol z)p_{\boldsymbol \theta}(\boldsymbol z)}{q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)}d\boldsymbol z\nonumber\\
		&\geq\int q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x){\rm log}\frac{p_{\boldsymbol \theta}(\boldsymbol x\mid \boldsymbol z)p_{\boldsymbol \theta}(\boldsymbol z)}{q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)}d\boldsymbol z\nonumber\\
		&=\int q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)
		\biggl\{
		{\rm log}\frac{p_{\boldsymbol \theta}(\boldsymbol z)}{q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)}+{\rm log}p_{\boldsymbol \theta}(\boldsymbol x\mid \boldsymbol z)
		\biggr\}d\boldsymbol z\nonumber\\
		&=\int q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x){\rm log}p_{\boldsymbol \theta}(\boldsymbol x\mid\boldsymbol z)d\boldsymbol z-\int q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x){\rm log}\frac{q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)}{p_{\boldsymbol \theta}(\boldsymbol z)}d\boldsymbol z\nonumber\\
		&=\double E_{\boldsymbol z \sim q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)}[{\rm log}p_{\boldsymbol \theta}(\boldsymbol x\mid\boldsymbol z)] - D_{KL}(q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)||p_{\boldsymbol \theta}(\boldsymbol z))\\
		&\simeq {\rm log}p_{\boldsymbol \theta}(\boldsymbol x\mid\boldsymbol z^{(l)}) - D_{KL}(q_{\boldsymbol \phi}(\boldsymbol z\mid\boldsymbol x)||p_{\boldsymbol \theta}(\boldsymbol z))
	\end{align}
$$

VAEでは${\rm log}p_{\boldsymbol \theta}(\boldsymbol x)$を直接最大化するのが困難なので、その下限値を最大化します。

式(5)が論文中の式(5)に対応します。

式(6)は$L=1$とした時の近似です。ミニバッチ数を100などの大きな値にしている場合はこのような粗い近似でもかまいません。

式(6)の第一項はchainer.functions.bernoulli_nllまたはchainer.functions.gaussian_nllで求めることができます。

$\boldsymbol x$が正規分布に従うと仮定してデコーダを作った場合、$p_{\theta}(\boldsymbol x \mid \boldsymbol z)$は以下の様なニューラルネットになります。

$$
	\begin{align}
		p_{\theta}(\boldsymbol x \mid \boldsymbol z) ={\cal N}(\boldsymbol x \mid \boldsymbol \mu_{\theta}(\boldsymbol z), {\rm diag}(\boldsymbol \sigma^2_{\theta}(\boldsymbol z)))
	\end{align}\
$$

chainer.functions.gaussian_nllは引数として$\boldsymbol x$、$\boldsymbol \mu_{\theta}(\boldsymbol z)$、$$\boldsymbol \sigma^2_{\theta}(\boldsymbol z)$$の3つを取りますが、$\boldsymbol \sigma^2_{\theta}(\boldsymbol z)$の入力は注意が必要です。

分散$\boldsymbol \sigma^2$は負の値を取ってはいけませんが、ニューラルネットの出力である$\boldsymbol \sigma^2_{\theta}(\boldsymbol z)$は負の値を取ります。（シグモイド関数を使えば別ですが）

そこで$\boldsymbol \sigma^2_{\theta}(\boldsymbol z)$を、$\boldsymbol \sigma^2$ではなく${\rm log}(\boldsymbol \sigma^2)$とみなすことで負の値を許容します。

従って、chainer.functions.gaussian_nllに$$\boldsymbol \sigma^2_{\theta}(\boldsymbol z)$$の出力値を入力するときは負の値を気にする必要はありません。

式(6)の第2項はchainer.functions.gaussian_kl_divergenceを使うと求めることができます。

こちらも同様に負の値を気にせず$$\boldsymbol \sigma^2_{\phi}(\boldsymbol x)$$の出力を引数に渡します。

実際のコードは以下のようになります。

```
def train(self, x, L=1, test=False):
	z_mean, z_ln_var = self.encoder(x, test=test, sample_output=False)
	reconstuction_loss = 0
	for l in xrange(L):
		# Sample z
		z = F.gaussian(z_mean, z_ln_var)
		# Decode
		x_reconstruction_mean, x_reconstruction_ln_var = self.decoder(z, test=test, output_pixel_value=False)
		# E_q(z|x)[log(p(x|z))]
		reconstuction_loss += F.gaussian_nll(x, x_reconstruction_mean, x_reconstruction_ln_var)
	loss = reconstuction_loss / (L * x.data.shape[0])
	# KL divergence
	kld_regularization_loss = F.gaussian_kl_divergence(z_mean, z_ln_var)
	loss += kld_regularization_loss / x.data.shape[0]

	self.zero_grads()
	loss.backward()
	self.update()
```

## M2の実装