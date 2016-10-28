---
layout: post
title: Deep Directed Generative Models with Energy-Based Probability Estimation [arXiv:1606.03439]
category: 論文
tags:
- Chainer
excerpt_separator: <!--more-->
---

## 概要

- [Deep Directed Generative Models with Energy-Based Probability Estimation](https://arxiv.org/abs/1606.03439) を読んだ
- Chainer 1.17で実装した

<!--more-->

## はじめに

提案モデル（略してDDGMと呼ぶことにします）は、[Restricted Boltzmann Machine（RBM）](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)と同じくエネルギー関数を用いた確率モデルです。

エネルギー関数を用いるモデルは学習の際、モデルからのサンプリングが必要になるのですが、一般的にはMCMCなどのコストのかかる方法が用いられてきました。

DDGMではエネルギーモデル（Deep Energy Model, DEM）とは別に生成モデル（Deep Generative Model, DGM）を用い、[Generative Adversarial Networks（GAN）](https://arxiv.org/abs/1406.2661)のアイディアを用いてモデルからのサンプリングをコストの低い伝承サンプリングで近似することができるようになっています。

またDDGMを畳み込みニューラルネットで実装した場合、生成モデル部分は[Deep Convolutional Generative Adversarial Networks（DCGAN）](https://arxiv.org/abs/1511.06434)と同じ働きをします。

## モデル

DDGMでは訓練データ$\boldsymbol x$の尤度はボルツマン分布で表されます。

$$
	\begin{align}
		P_{\Theta}(\boldsymbol x) &= \frac
			{
				e^{-E_{\Theta}(\boldsymbol x)}
			}
			{
				Z_{\Theta}
			}\\
		Z_{\Theta} &= \sum_{\boldsymbol x}e^{-E_{\Theta}(\boldsymbol x)}
	\end{align}\
$$

$E_{\Theta}(\boldsymbol x)$はエネルギー関数、$Z_{\Theta}$は正規化項（分配関数）です。

上でも書きましたが、この分配関数を計算するにはあらゆるデータ$\boldsymbol x$のエネルギーを計算する必要があります。

またエネルギー関数は"expert"と呼ばれる項の和になっています。

$$
	\begin{align}
		E_{\Theta}(\boldsymbol x) = \sum_{i}\tilde{E}_{\theta_i}(\boldsymbol x)
	\end{align}\
$$

### Deep Energy Model（DEM）

DDGMでは入力$\boldsymbol x$をそのままエネルギー関数$E_{\Theta}(\boldsymbol x)$に渡すのではなく、特徴抽出器（feature extractor）$f_{\psi}$を用いて特徴量を取り出し、それをエネルギー関数に入力します。

$E_{\Theta}(\boldsymbol x)$は単層ネットワークですが、$f_{\psi}(\boldsymbol x)$はCNNでも全結合でも良く、深いネットワークで構成されます。

これらを用いてDeep Energy Modelのエネルギー関数を以下のように定義します。

$$
	\begin{align}
		E_{\Theta}(\boldsymbol x) = E_{\Theta'}(\boldsymbol x, f_{\psi}(\boldsymbol x)) = 
		\frac{1}{\sigma^2}\boldsymbol x^T\boldsymbol x-\boldsymbol b^T\boldsymbol x - 
		\sum_i {\rm log}(1+e^{W_i^Tf_{\psi}(\boldsymbol x) + c_i})
	\end{align}\
$$

また各expertは

$$
	\begin{align}
		\tilde{E}_{\theta_i}(f_{\psi}(\boldsymbol x)) = -{\rm log}(1+e^{W_i^Tf_{\psi}(\boldsymbol x) + c_i})
	\end{align}\
$$

となります。

論文で$b_i$となっている部分は$c_i$の間違いだと思います。（$\boldsymbol b^T$と$b_i$は別物のはずです）

### Deep Generative Model（DGM）

生成モデル部分は隠れ変数$\boldsymbol z$をとる生成関数$G_{\boldsymbol \Phi}(\boldsymbol z)$になっています。

$\boldsymbol z$は一様分布などからサンプリングしますが、$\boldsymbol x$の生成モデルからのサンプリングは$\boldsymbol x = G_{\boldsymbol \Phi}(\boldsymbol z)$と決定的に決まります。

このあたりはGANと同じです。

## 学習

DEMのパラメータを$\boldsymbol \Theta$とすると、誤差関数は訓練データの負の対数尤度となります。

$$
	\begin{align}
		{\cal L}(\boldsymbol \Theta) &= \double E_{\boldsymbol x \sim P_D(\boldsymbol x)}[-{\rm log}P_{\boldsymbol \Theta}(\boldsymbol x)]\nonumber\\
		&\simeq -\frac{1}{N}\sum_{i=1}^{N}{\rm log}P_{\boldsymbol \Theta}(\boldsymbol x^{(i)})
	\end{align}\
$$

$P_D(\boldsymbol x)$はデータ分布です。

$\double E$は期待値を表します。

$$
	\begin{align}
		\double E_{\boldsymbol x \sim P_D(\boldsymbol x)}[-{\rm log}P_{\boldsymbol \Theta}(\boldsymbol x)]
		= \int -{\rm log}P_{\boldsymbol \Theta}(\boldsymbol x) P_D(\boldsymbol x) d\boldsymbol x\nonumber
	\end{align}\
$$

式(6)は式(1)より

$$
	\begin{align}
		{\cal L}(\boldsymbol \Theta) 
		&\simeq -\frac{1}{N}\sum_{i=1}^{N}{\rm log}P_{\boldsymbol \Theta}(\boldsymbol x^{(i)})\nonumber\\
		&= \frac{1}{N}\sum_{i=1}^{N}E_{\boldsymbol \Theta}(\boldsymbol x^{(i)})
		-\double E_{\boldsymbol x \sim P_{\boldsymbol \Theta}(\boldsymbol x)}\left[E_{\boldsymbol \Theta}(\boldsymbol x)\right]\nonumber\\
		&\simeq \underbrace{\double E_{\boldsymbol x^{+} \sim P_D(\boldsymbol x)}\left[E_{\boldsymbol \Theta}(\boldsymbol x^{+})\right]}_{Positive\ Phase}
		-\underbrace{\double E_{\boldsymbol x^{-} \sim P_{\boldsymbol \Theta}(\boldsymbol x)}\left[E_{\boldsymbol \Theta}(\boldsymbol x^{-})\right]}_{Negative\ Phase}
	\end{align}\
$$

のように変形することができます。

Positive Phaseは訓練データがあれば計算できますが、Negative PhaseはDEMからサンプリングする必要があります。

そこでDDGMでは、DGMによるサンプリング可能な分布$P_{\boldsymbol \Psi}(\boldsymbol x)$を用意し、Negative Phaseを

$$
	\begin{align}
		\double E_{\boldsymbol x^{-} \sim P_{\boldsymbol \Theta}(\boldsymbol x)}\left[E_{\boldsymbol \Theta}(\boldsymbol x^{-})\right]
		&\simeq \double E_{\boldsymbol x \sim P_{\boldsymbol \Phi}(\boldsymbol x)}\left[E_{\boldsymbol \Theta}(\boldsymbol x)\right]\nonumber\\
		&= \double E_{\boldsymbol z \sim P(\boldsymbol z)}\left[E_{\boldsymbol \Theta}\left(G_{\boldsymbol \Phi}(\boldsymbol z)\right)\right]\nonumber\\
		&\simeq \frac{1}{N}\sum_{i=1}^{N}E_{\boldsymbol \Theta}\left(G_{\boldsymbol \Phi}(\boldsymbol z^{(i)})\right) \hspace{20pt} {\rm where}\ \boldsymbol z \sim P(\boldsymbol z)
	\end{align}\
$$

のように近似します。

ただし、このままでは$P_{\boldsymbol \Psi}(\boldsymbol x)$は$P_{\boldsymbol \Theta}(\boldsymbol x)$とは無関係な分布のままですので、両者のKLダイバージェンスを最小化することで$P_{\boldsymbol \Psi}(\boldsymbol x)$を$P_{\boldsymbol \Theta}(\boldsymbol x)$に近づけます。

このKLダイバージェンスは以下のように定義されます。

$$
	\begin{align}
		D_{KL}(P_{\boldsymbol \Psi}(\boldsymbol x) \mid\mid P_{\boldsymbol \Theta}(\boldsymbol x))
		&=\double E_{\boldsymbol x^{-} \sim P_{\boldsymbol \Phi}(\boldsymbol x)}\left[-{\rm log}P_{\boldsymbol \Theta}(\boldsymbol x^{-}) \right]-H(P_{\boldsymbol \Phi}(\boldsymbol x))\nonumber\\
		&=\double E_{\boldsymbol z \sim P(\boldsymbol z)}\left[-{\rm log}P_{\boldsymbol \Theta}(G_{\boldsymbol \Phi}(\boldsymbol z)) \right]-H(P_{\boldsymbol \Phi}(\boldsymbol x))\nonumber\\
		&\simeq \frac{1}{N}\sum_{i=1}^{N}E_{\boldsymbol \Theta}\left(G_{\boldsymbol \Phi}(\boldsymbol z^{(i)})\right)-H(P_{\boldsymbol \Phi}(\boldsymbol x)) \hspace{20pt} {\rm where}\ \boldsymbol z \sim P(\boldsymbol z)
	\end{align}\
$$

$H(P_{\boldsymbol \Phi}(\boldsymbol x))$はエントロピーです。

負のエントロピーを減少させる（つまりエントロピーを増大させる）ことで、生成されるサンプルが多様性を持つようになるため、この項は局所解を避ける正則化項として働きます。

ただしこのエントロピーは解析的に求められないのですが、論文によるとBatch Normalizationレイヤーのスケールパラメータを正規分布の標準偏差とみなし、正規分布のエントロピーを計算することで近似することができるそうです。

## 実装

論文に実験の詳細がほとんどかかれていないため試行錯誤を要しましたが、今のところ得られた知見は以下のとおりです。

- DEMの中間層の活性化関数はLeaky ReLUやELUなどを用いて誤差をより多く逆伝播できるようにする
- DEMのfeature extractorの出力層の活性化関数はtanhを使う
	- 最初は$f_{\psi}$の出力はマイナスの値も取れる必要があると考えていた
	- どちらかというと活性化関数の出力が大きくなりすぎなければ問題がないように思える
	- sigmoidでもいいかもしれない
	- 活性化関数を通さなかったらどうなるかは実験していない
- DGMの中間層の活性化関数はReLUでよい
- DEMにBatch Normalizationを使うとDGMの学習に失敗することがある
	- 訓練データの平均をとったようなものが出てくる（後述）

またエントロピーの計算ではBatch Normalizationのスケールパラメータを使うのですが、スケールにも誤差を逆伝播し値を更新する必要があります。

私はchainerのBatchNormalizationレイヤーのgammaを取ってきて計算していますが、正しく学習できているか不明です。

実装はGithubで公開しています。

[https://github.com/musyoku/ddgm](https://github.com/musyoku/ddgm)

## 実験

### 2次元データ

Adversarial AutoEncoderのときに使ったガウス混合分布と渦巻き型の分布を学習させてみました。

学習過程を動画にしたものがこちらです。

[https://gfycat.com/DarlingShowyHypsilophodon](https://gfycat.com/DarlingShowyHypsilophodon)

[https://gfycat.com/UnrulyMisguidedHornedviper](https://gfycat.com/UnrulyMisguidedHornedviper)

### MNIST

#### 成功例

![success](/images/post/2016-10-28/mnist_success.png)

あまり綺麗ではないですね。

#### 失敗例

![fall](/images/post/2016-10-28/mnist_fall.png)

平均を取ったようなデータしか生成できません。

### キルミーベイベー

[公式が配布しているアイコン686枚](http://killmebaby.tv/special_icon.html)を学習させました。

画像は全て$64 \times 64$にリサイズしました。

#### オリジナル

![kb_original](/images/post/2016-10-28/kb_original.png)

#### 生成例

![kb_gen](/images/post/2016-10-28/kb_gen.png)

今回のデータは顔の位置が揃っていないため綺麗な画像が生成できませんでしたが、それっぽいのが生成されているので良しとします。

もう少し工夫すればうまくいきそうな気がします。

ちなみにキルミーベイベーの学習でも局所解に落ちてしまった例を用意していたのですが、いつの間にかデータが消えていたので載せることができません。

## おわりに

体力を使い果たしたのでアナロジーの実験ができませんでした。

feature extractorの出力とノイズ$\boldsymbol z$の関係なども気になります。