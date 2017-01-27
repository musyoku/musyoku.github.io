---
layout: post
title: A Fully Bayesian Approach to Unsupervised Part-of-Speech Tagging
category: 実装
tags:
- 自然言語処理
excerpt_separator: <!--more-->
---

## 概要

- [A Fully Bayesian Approach to Unsupervised Part-of-Speech Tagging](http://homepages.inf.ed.ac.uk/sgwater/papers/acl07-bhmm.pdf)を読んだ
- C++で実装しPythonのラッパーを作った
- 日本語・英語のデータセットで教師なし品詞推定の実験を行った

<!--more-->

## はじめに

[隠れセミマルコフモデルに基づく教師なし完全形態素解析](http://www.anlp.jp/proceedings/annual_meeting/2015/pdf_dir/C6-3.pdf)を実装していたところ、先行研究としてこの論文が出ていたため先に実装しました。

この論文では単語の品詞を隠れ状態とした隠れマルコフモデルを考え、ギブスサンプリングによってモデルを更新していきます。

またこの論文は理解を読者に丸投げするタイプのもので、私のような初学者にとっては実装のハードルが高いです。

## 必要な式

ベータ関数とガンマ関数について以下の関係をこれ以降の説明で用います。

$$
	\begin{align}
		B(x, y) &= \int_0^1 t^{x-1}(1-t)^{y-1}dt\\
				&= \frac{\Gamma(x)\Gamma(y)}{\Gamma(x+y)}\\
		\Gamma(z + 1) &= z\Gamma(z)\\
					&= z!\\
	\end{align}\
$$

また、一般に以下の条件のもとで、

$$
	\begin{align}
		\sum_{k=1}^{K} \theta_k = 1\nonumber\\
	\end{align}\
$$

以下が成り立ちます。

$$
	\begin{align}
		\int \theta_1^{\alpha_1-1}\cdot\cdot\cdot\theta_k^{\alpha_k-1}d\boldsymbol \theta = \frac{\Gamma(\alpha_1)\cdot\cdot\cdot\Gamma(\alpha_k)}{\Gamma(\alpha_1+\cdot\cdot\cdot+\alpha_k)}\\
	\end{align}\
$$

## コインの例

ベイズ的なアプローチの利点を説明するために論文ではコインを用いた例題が紹介されています。

この項目は品詞推定とは無関係なので飛ばしても構いません。

ここで用いるコインは2種類あり、バイアスがかかっているもの（$t=1$）とそうでないもの（$t=0$）があります。

バイアスがかかっている確率は50%で、かかっていない確率も50%に設定されています。

コインの表が出る確率を$\theta$とすると、バイアスがかかっている場合$\theta$はある一様分布から決定されており、バイアスがかかっていない場合は$\theta=0.5$となっています。

ちなみにコインの表を英語でheads、裏をtailsと言うそうです。

次にコインを10回投げた結果$\boldsymbol w$を得たとします。

表を$\rm H$、裏を$\rm T$で表すと、10回投げた時の結果はたとえば$\boldsymbol w={\rm HTTHTHHTHT}$のように表します。

次に得られた結果とコインの表が出る確率から、そのコインがバイアスのかかったものか、それとも普通のコインなのかを予測します。

ここではコインにバイアスがかかっている確率$p(t=1 \mid \boldsymbol w, \theta)$を求めます。

表が出た回数を$n_H$とすると最尤推定では$\theta=\frac{n_H}{10}$としますが、ベイズ的なアプローチでは$\theta$を積分消去した分布$p(t=1 \mid \boldsymbol w)$を考えることで、$\theta$によらないロバストな推定を行うことができます。

論文ではいきなり答えが出てきていますが、導出過程は以下のようになります。

まずベイズの定理を用いて変形します。

$$
	\begin{align}
		P(t=1 \mid \boldsymbol w)&=\frac{P(\boldsymbol w \mid t=1)P(t=1)}{P(\boldsymbol w)}\nonumber\\
		&=\frac{P(\boldsymbol w \mid t=1)P(t=1)}{\sum_tP(\boldsymbol w \mid t)P(t)}\nonumber\\
		&=\frac{P(\boldsymbol w \mid t=1)P(t=1)}{P(\boldsymbol w \mid t=0)P(t=0) + P(\boldsymbol w \mid t=1)P(t=1)}\nonumber\\
	\end{align}\
$$

問題設定より

$$
	\begin{align}
		P(t=0) = P(t=1) = 0.5\nonumber\\
	\end{align}\
$$

なので

$$
	\begin{align}
		P(t=1 \mid \boldsymbol w)=\frac{P(\boldsymbol w \mid t=1)}{P(\boldsymbol w \mid t=0) + P(\boldsymbol w \mid t=1)}\\
	\end{align}\
$$

となります。

右辺の各項は以下のように$\theta$に関して積分消去されています。

$$
	\begin{align}
		P(\boldsymbol w \mid t=0) &= \int_0^1 P(\boldsymbol w \mid t=0, \theta)P(\theta)d\theta\\
		P(\boldsymbol w \mid t=1) &= \int_0^1 P(\boldsymbol w \mid t=1, \theta)P(\theta)d\theta\\
	\end{align}\
$$

問題設定より以下のことが言えますので、

$$
	\begin{align}
		P(\boldsymbol w \mid t=1, \theta) &= \theta^{n_H}(1-\theta)^{n_T}\nonumber\\
		P(\boldsymbol w \mid t=0, \theta) &= \left(\frac{1}{2}\right)^{n_H}\left(\frac{1}{2}\right)^{n_T} = \left(\frac{1}{2}\right)^{10}\nonumber\\
		P(\theta) &= 1\nonumber\\
	\end{align}\
$$

これらを式(8)に代入すると

$$
	\begin{align}
		P(\boldsymbol w \mid t=1) &= \int_0^1 P(\boldsymbol w \mid t=1, \theta)P(\theta)d\theta\nonumber\\
		&= \int_0^1 \theta^{n_H}(1-\theta)^{n_T}d\theta\nonumber\\
		&= B(n_H+1, n_T+1)\nonumber\\
		&= \frac{\Gamma(n_H+1)\Gamma(n_T+1)}{\Gamma(n_H+n_T+2)}\nonumber\\
		&= \frac{n_H!n_T!}{(n_H+n_T+1)!}\nonumber\\
		&= \frac{n_H!n_T!}{11!}\\
	\end{align}\
$$

となります。

これを式(6)に代入すると答えが求まります。

$$
	\begin{align}
		P(t=1 \mid \boldsymbol w) &= \frac{\frac{n_H!n_T!}{11!}}{\frac{n_H!n_T!}{11!}+\frac{1}{2}^{10}}\nonumber\\
		&=1/\left(1+\frac{11!}{n_H!n_T!2^{10}}\right)\nonumber\\
	\end{align}\
$$

## 多項分布

一般的に言語モデルでは単語分布などに多項分布を用います。（と論文に書いてあります）

例えば$K$種類の単語があったとき、それぞれの単語が出現する確率を、パラメータ$\boldsymbol \theta=(\theta_1,...,\theta_K)$の多項分布で表します。

例えば単語$w_k$が出現する確率は$\theta_k$になります。

これはDeep Learningのソフトマックス層をイメージするとわかりやすいです。

そしてパラメータ$\boldsymbol \theta$の事前分布として広く用いられるのがディリクレ分布で、以下のように表されます。

$$
	\begin{align}
		{\rm Dir}(\alpha_1,...,\alpha_k)=\frac{\Gamma(\alpha)}{\Gamma(\alpha_1)\cdot\cdot\cdot\Gamma(\alpha_K)}{\theta_1}^{\alpha_1-1}\cdot\cdot\cdot{\theta_K}^{\alpha_K-1}\\
	\end{align}\
$$

ただし、

$$
	\begin{align}
		\alpha = \sum_{i=1}^K \alpha_i
	\end{align}\
$$

であり、$\alpha_1,...,\alpha_k$はハイパーパラメータです。

論文では簡単のためにsymmetricなディレクレ分布を使うとありますが、これはおそらく$\alpha_1,...,\alpha_k$の値が全て同じことを意味しています。

これらを踏まえて観測したデータのクラス確率を考えます。

クラス数を$K$とし、$i-1$個のデータ$\boldsymbol x=x_1,...,x_{i-1}$を観測したとします。

また各クラスの出現回数を$n_1,...,n_K$で表します。

クラス変数はパラメータ$\boldsymbol \theta=(\theta_1,...,\theta_K)$の多項分布に従っており、クラス$k$の出現確率は$\theta_k$で表されます。

また$\boldsymbol \theta$の事前分布としてパラメータ$\beta$の対称なディリクレ分布

$$
	\begin{align}
		P(\boldsymbol \theta) = {\rm Dir}(\beta)=\frac{\Gamma(K\beta)}{\Gamma(\beta)\cdot\cdot\cdot\Gamma(\beta)}{\theta_1}^{\beta-1}\cdot\cdot\cdot{\theta_K}^{\beta-1}\\
	\end{align}\
$$

を考えます。

この時$i$番目のデータ$x_i$がクラス$k$である確率は論文の式を用いると以下のように表されます。

$$
	\begin{align}
		P(x_i = k \mid \boldsymbol x_{-1}, \beta) &= \int P(k \mid \boldsymbol \theta)P(\boldsymbol \theta \mid \boldsymbol x_{-1}, \beta) d\boldsymbol \theta\nonumber\\
		&= \frac{n_k+\beta}{i-1+K\beta}
	\end{align}\
$$

上のコインの例と同様、パラメータ$\boldsymbol \theta$を積分消去しています。

この式の導出は論文には書かれていませんが、以下のように行ないます。

まずベイズの定理を用いて変形します。

$$
	\begin{align}
		P(x_i = k \mid \boldsymbol x_{-1}, \beta) &= \int P(k \mid \boldsymbol \theta)P(\boldsymbol \theta \mid \boldsymbol x_{-1}, \beta) d\boldsymbol \theta\nonumber\\
		&= \int \theta_k\frac{P(\boldsymbol x_{-1} \mid \boldsymbol \theta, \beta)P(\boldsymbol \theta)}{P(\boldsymbol x_{-1})}d\boldsymbol \theta\nonumber\\
		&= \int \theta_k\theta_1^{n_1}\cdot\cdot\cdot\theta_k^{n_k}\cdot\cdot\cdot\theta_K^{n_K}\frac{P(\boldsymbol \theta)}{P(\boldsymbol x_{-1})}d\boldsymbol \theta\nonumber\\
		&= \int \theta_1^{n_1}\cdot\cdot\cdot\theta_k^{n_k+1}\cdot\cdot\cdot\theta_K^{n_K}\frac{P(\boldsymbol \theta)}{P(\boldsymbol x_{-1})}d\boldsymbol \theta\\
	\end{align}\
$$

$P(\boldsymbol \theta)$は問題で定義されているので$P(\boldsymbol x_{-1})$を求めると、

$$
	\begin{align}
		P(\boldsymbol x_{-1}) &= \int P(\boldsymbol x_{-1} \mid \boldsymbol \theta)P(\boldsymbol \theta)d\boldsymbol \theta\nonumber\\
		&=\int \theta_1^{n_1}\cdot\cdot\cdot\theta_k^{n_K}\frac{\Gamma(K\beta)}{\Gamma(\beta)^K}\theta_1^{\beta-1}\cdot\cdot\cdot\theta_k^{\beta-1}\nonumber\\
		&= \frac{\Gamma(K\beta)}{\Gamma(\beta)^K} \int \theta_1^{n_1 + \beta - 1}\cdot\cdot\cdot\theta_k^{n_K + \beta - 1}\nonumber\\
		&= \frac{\Gamma(K\beta)\Gamma(n_1 + \beta)\cdot\cdot\cdot\Gamma(n_K + \beta)}{\Gamma(\beta)^K\Gamma(i-1+K\beta)}\\
		i &= n_1+\cdot\cdot\cdot+n_K-1\nonumber\\
	\end{align}\
$$

となります。

式(12)と式(15)を式(14)に代入すると、

$$
	\begin{align}
		P(x = k \mid \boldsymbol x_{-1}, \beta) &= \int \theta_1^{n_1}\cdot\cdot\cdot\theta_k^{n_k+1}\cdot\cdot\cdot\theta_K^{n_K}\frac{P(\boldsymbol \theta)}{P(\boldsymbol x_{-1})}d\boldsymbol \theta\nonumber\\
		&= \int \theta_1^{n_1}\cdot\cdot\cdot\theta_k^{n_k+1}\cdot\cdot\cdot\theta_K^{n_K}
		\frac{\Gamma(K\beta)}{\Gamma(\beta)^K}\theta_1^{\beta-1}\cdot\cdot\cdot\theta_k^{\beta-1}
		\frac{\Gamma(\beta)^K\Gamma(i-1+K\beta)}{\Gamma(K\beta)\Gamma(n_1 + \beta)\cdot\cdot\cdot\Gamma(n_K + \beta)}
		d\boldsymbol \theta\nonumber\\
		&= \frac{\Gamma(i-1+K\beta)}{\Gamma(n_1 + \beta)\cdot\cdot\cdot\Gamma(n_K + \beta)}
		\int \theta_1^{n_1+\beta-1}\cdot\cdot\cdot\theta_k^{n_k+\beta}\cdot\cdot\cdot\theta_K^{n_K+\beta-1}d\boldsymbol \theta\nonumber\\
		&= \frac{\Gamma(i-1+K\beta)}{\Gamma(n_1 + \beta)\cdot\cdot\cdot\Gamma(n_K + \beta)}
		\frac{\Gamma(n_1 + \beta)\cdot\cdot\cdot\Gamma(n_k + \beta + 1)\cdot\cdot\cdot\Gamma(n_K + \beta)}{\Gamma(i+K\beta)}\nonumber\\
		&= \frac{\Gamma(i-1+K\beta)}{\Gamma(n_k + \beta)}\frac{\Gamma(n_k + \beta + 1)}{\Gamma(i+K\beta)}\nonumber\\
		&= \frac{\Gamma(i-1+K\beta)}{\Gamma(n_k + \beta)}\frac{(n_k + \beta)\Gamma(n_k + \beta)}{(i-1+K\beta)\Gamma(i-1+K\beta)}\nonumber\\
		&= \frac{n_k+\beta}{i-1+K\beta}\nonumber\\
	\end{align}\
$$

となり式(13)が導出できます。

## 教師なし品詞推定

### モデル

提案手法は品詞トライグラムの隠れマルコフモデルで、以下のように定義します。

$$
	\begin{align}
		t_i &\mid t_{i-1}=t, t_{i-2}=t',\tau^{(t,t')} \sim {\rm Mult}(\tau^{(t,t')})\\
		w_i &\mid t_i = t, \omega^{(t)} \sim {\rm Mult}(\omega^{(t)})\\
		\tau^{(t,t')} &\mid \alpha \sim {\rm Dirichlet}(\alpha)\\
		\omega^{(t)} &\mid \beta \sim {\rm Dirichlet}(\beta)\\
	\end{align}\
$$

$t_i$は$i$番目に観測する品詞タグで、$w_i$は$i$番目に観測する単語です。

ちなみにこの縦棒の記号ですが、

$$
	\begin{align}
		t_i &\mid t_{i-1}=t, t_{i-2}=t',\tau^{(t,t')} \sim {\rm Mult}(\tau^{(t,t')})\nonumber\\
	\end{align}\
$$

は$t_{i-1}=t, t_{i-2}=t',\tau^{(t,t')}$が与えられたときに$t_i$が${\rm Mult}(\tau^{(t,t')})$から生成されることを表しています。

可能な品詞タグの総数を$T$とすると、品詞の遷移確率$\tau^{(t,t')}$は$T$個パラメータを持つ多項分布になります。

また単語の多項分布のパラメータ$\omega^{(t)}$ですが、これの要素数は単語の総数$W$ではなく、品詞$t$として可能な単語の総数$W_t$個の要素からなります。

単語は特定の品詞としか結びつかないのでこれは当然の設定です。

次に$\tau^{(t,t')}$と$\omega^{(t)}$を積分消去するのですが、これは式(13)の変数を変えるだけで求められます。

$$
	\begin{align}
		P(t_i \mid \boldsymbol t_{-i}, \alpha) &= \frac{n_{(t_{i-2}, t_{i-1}, t_i)} + \alpha}{n_{(t_{i-2}, t_{i-1})} + T\alpha}\\
		P(w_i \mid t_i, \boldsymbol t_{-i}, \boldsymbol w_{-i}, \alpha) &= \frac{n_{t_i, w_i} + \beta}{n_{t_i} + W_t\beta}\\
		t_{-i} &= t_1,...,t_{i-1},t_{i+1},...\\
		w_{-i} &= w_1,...,w_{i-1},w_{i+1},...\\
	\end{align}\
$$

$n_{(t_{i-2}, t_{i-1}, t_i)}$はトライグラムのカウントです。

たとえば$t_{i-2}=1,t_{i-1}=4,t_{i}=3$だった場合、$1 \to 4 \to 3$の順で品詞が並んで観測された回数を表しています。

同様に$n_{(t_i, w_i)}$は品詞と単語のペアのカウントです。

$n_{(t_{i-2}, t_{i-1}, t_i)}$と$n_{(t_i, w_i)}$はともに$i-1$番目までの観測結果のカウントであることに注意が必要です。

### 品詞の推定






