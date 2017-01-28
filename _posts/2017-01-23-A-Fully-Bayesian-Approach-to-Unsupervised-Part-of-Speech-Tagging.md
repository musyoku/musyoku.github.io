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

論文では簡単のためにsymmetricなディレクレ分布を使うとありますが、これはおそらく$\alpha_1,...,\alpha_K$の値が全て同じことを意味しています。

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
		t_i &\mid t_{i-1}=t, t_{i-2}=t',\boldsymbol \tau^{(t,t')} \sim {\rm Mult}(\tau^{(t,t')})\\
		w_i &\mid t_i = t, \boldsymbol \omega^{(t)} \sim {\rm Mult}(\omega^{(t)})\\
		\boldsymbol \tau^{(t,t')} &\mid \alpha \sim {\rm Dirichlet}(\alpha)\\
		\boldsymbol \omega^{(t)} &\mid \beta \sim {\rm Dirichlet}(\beta)\\
	\end{align}\
$$

$t_i$は$i$番目に観測する品詞タグで、$w_i$は$i$番目に観測する単語です。

ちなみにこの縦棒の記号ですが、

$$
	\begin{align}
		t_i &\mid t_{i-1}=t, t_{i-2}=t',\boldsymbol \tau^{(t,t')} \sim {\rm Mult}(\tau^{(t,t')})\nonumber\\
	\end{align}\
$$

は$t_{i-1}=t, t_{i-2}=t',\boldsymbol \tau^{(t,t')}$が与えられたときに$t_i$が${\rm Mult}(\boldsymbol \tau^{(t,t')})$から生成されることを表しています。

可能な品詞タグの総数を$\mid T \mid$とすると、品詞の遷移確率$\boldsymbol \tau^{(t,t')}$は$\mid T \mid$個パラメータを持つ多項分布になります。

また単語の多項分布のパラメータ$\boldsymbol \omega^{(t)}$ですが、これの要素数は単語の総数$\mid W \mid$ではなく、品詞$t$として可能な単語の総数$\mid W_t \mid$個の要素からなります。

単語は特定の品詞としか結びつかないのでこれは当然の設定です。

次に$\boldsymbol \tau^{(t,t')}$と$\boldsymbol \omega^{(t)}$を積分消去するのですが、これは式(13)の変数を変えるだけで求められます。

$$
	\begin{align}
		P(t_i \mid \boldsymbol t_{-i}, \alpha) &= \frac{n_{(t_{i-2}, t_{i-1}, t_i)} + \alpha}{n_{(t_{i-2}, t_{i-1})} + T\alpha}\\
		P(w_i \mid t_i, \boldsymbol t_{-i}, \boldsymbol w_{-i}, \alpha) &= \frac{n_{(t_i, w_i)} + \beta}{n_{t_i} + W_t\beta}\\
		t_{-i} &= t_1,...,t_{i-1},t_{i+1},...\\
		w_{-i} &= w_1,...,w_{i-1},w_{i+1},...\\
	\end{align}\
$$

$n_{(t_{i-2}, t_{i-1}, t_i)}$はトライグラムのカウントです。

たとえば$t_{i-2}=1,t_{i-1}=4,t_{i}=3$だった場合、$1 \to 4 \to 3$の順で品詞が並んで観測された回数を表しています。

同様に$n_{(t_i, w_i)}$は品詞と単語のペアのカウントです。

$n_{(t_{i-2}, t_{i-1}, t_i)}$と$n_{(t_i, w_i)}$はともに$i-1$番目までの観測結果のカウントであることに注意が必要です。

ちなみにこのカウントは添字$i$とは無関係です。

たとえば$t_{i-2}=1,t_{i-1}=4,t_{i}=3$の場合と、$t_{i-9997}=1,t_{i-9998}=4,t_{i-9999}=3$の場合では、ともに同じカウント$n_{(1,4,3)}$を参照します。

### 品詞の推定

今回のモデルで品詞を推定するにはギブスサンプリングを用いて品詞タグの事後分布からサンプリングします。

$$
	\begin{align}
		P(\boldsymbol t \mid \boldsymbol w, \alpha, \beta) \propto P(\boldsymbol w \mid \boldsymbol t, \beta)P(\boldsymbol t \mid \alpha)\\
	\end{align}\
$$

ただし$\boldsymbol t$の要素を一つづつサンプリングして更新していくため、実際は以下の$t_i$の条件付き確率から品詞をサンプリングします。

$$
	\begin{align}
		P(t_i \mid \boldsymbol t_{-1}, \boldsymbol w, \alpha, \beta) \propto 
		\frac{n_{t_i, w_i} + \beta}{n_{t_i} + W_t\beta}\cdot
		\frac{n_{(t_{i-2}, t_{i-1}, t_i)} + \alpha}{n_{(t_{i-2}, t_{i-1})} + T\alpha}\cdot
		\frac{n_{(t_{i-1}, t_{i}, t_{i+1})} + I(t_{i-2} = t_{i-1} = t_i = t_{i+1}) + \alpha}{n_{(t_{i-1}, t_{i})} + I(t_{i-2} = t_{i-1} = t_i) + T\alpha}\cdot\nonumber\\
		\frac{n_{(t_{i}, t_{i+1}, t_{i+2})} + I(t_{i-2} = t_i = t_{i+2} , t_{i-1} = t_{i+1}) + I(t_{i-1} = t_i = t_{i+1} = t_{i+2}) + \alpha}{n_{(t_{i}, t_{i+1})} + I(t_{i-2} = t_i , t_{i-1} = t_{i+1}) + T\alpha}\cdot
	\end{align}\
$$

$I(\cdot)$は引数が真のときに1を返し、それ以外の場合は0を返す関数であり、トライグラムカウントや品詞-単語ペアのカウントを補正する項になっています。

この部分の説明は後で行ないます。

### $\alpha,\beta$の推定

ディリクレ分布を調整するハイパーパラメータの$\alpha$と$\beta$は定数にしてもよいのですが、メトロポリス・ヘイスティングス法で最適な値に更新することができます。

これについては以下の文献が詳しいです。

[http://ebsa.ism.ac.jp/ebooks/sites/default/files/ebook/1881/pdf/vol3_ch10.pdf](http://ebsa.ism.ac.jp/ebooks/sites/default/files/ebook/1881/pdf/vol3_ch10.pdf)

メトロポリス・ヘイスティングス法では提案分布と呼ばれる分布を任意に設定し、そこからのサンプルを採択確率によって採択するか棄却するかを決定します。

論文によると$\alpha$の推定では提案分布を平均$\alpha$、分散$(0.1\cdot\alpha)^2$の正規分布としています。

この提案分布を用いて、真の$\alpha$の分布である$\pi(\alpha)$からサンプリングを行う手順は以下のとおりです。

- 新しい$\alpha^{new}$を提案分布$q(\alpha^{new} \mid \alpha^{(t)}) = {\cal N}(\alpha^{(t)}, (0.1\cdot\alpha^{(t)})^2)$から発生させる
- $u$を一様分布${\cal U}(0,1)$から発生させ、以下に従って更新する

$$
	\begin{align}
		\alpha^{t+1} \gets
			\begin{cases}
				\alpha^{new} & {\rm if} \  u \leq {\cal A}(\alpha^{(t)}, \alpha^{new})\\
				\alpha^{(t)} & {\rm otherwise}.
			\end{cases}\
	\end{align}\
$$

${\cal A}(\alpha^{(t)}, \alpha^{new})$が採択確率で、以下のように求めます。

$$
	\begin{align}
		{\cal A}(\alpha^{(t)}, \alpha^{new}) = {\rm min}\left\{1, \frac{\pi(\alpha^{new})q(\alpha^{(t)} \mid \alpha^{new})}{\pi(\alpha^{(t)})q(\alpha^{new}\mid\alpha^{(t)})}\right\}
	\end{align}\
$$

論文によると$\pi(\alpha) = p(\boldsymbol t \mid \boldsymbol w, \alpha)$にするというニュアンスの書き方になっているため、確率ではなく$\alpha$の尤度を用いるようです。

（$p(\boldsymbol t \mid \boldsymbol w, \alpha)$は$\boldsymbol t$と$\boldsymbol w$が固定されているため$\alpha$の関数とみなすことができますが、$\alpha$を動かすことで$\boldsymbol t$を与えた$\alpha$として尤もらしい値が何なのかを計算することができ、これを尤度と言います）

また論文の"a term correcting for the asymmetric proposal distribution"とあるのはおそらく$\frac{q(\alpha^{(t)} \mid \alpha^{new})}{q(\alpha^{new}\mid\alpha^{(t)})}$のことであると考えられます。

なぜ"asymmetric"なのかというと、もし両方とも分散が同じであれば（"symmetric"であれば）$q(\alpha^{(t)} \mid \alpha^{new}) = q(\alpha^{new}\mid\alpha^{(t)})$となって消えるからです。

この項は正規分布の密度関数を考えると簡単に値を求めることができます。

$$
	\begin{align}
		\sigma^{new} &= 0.1\cdot\alpha^{new}\\
		\sigma^{(t)} &= 0.1\cdot\alpha^{(t)}\\
		\frac{q(\alpha^{(t)} \mid \alpha^{new})}{q(\alpha^{new}\mid\alpha^{(t)})} &= 
		\frac
		{
			\frac{1}{\sqrt{2\pi(\sigma^{new})^2}}{\rm exp}\left( -\frac{(\alpha^{(t)} - \alpha^{new})^2}{2\cdot{\sigma_{\alpha^{new}}}^2}\right)
		}
		{
			\frac{1}{\sqrt{2\pi(\sigma^{(t)})^2}}{\rm exp}\left( -\frac{(\alpha^{(new)} - \alpha^{(t)})^2}{2\cdot{\sigma_{\alpha^{(t)}}}^2}\right)
		}\\
		&= \frac{\alpha^{(t)}}{\alpha^{new}}{\rm exp}
		\left(
			\frac{(\alpha^{new} - \alpha^{(t)})^2}{2\cdot{\sigma_{\alpha^{(t)}}}^2}
			-\frac{(\alpha^{(t)} - \alpha^{new})^2}{2\cdot{\sigma_{\alpha^{new}}}^2}
		\right)
	\end{align}\
$$

$\sigma^{new} = \sigma^{(t)}$とすると正規分布が"symmetric"になり、上の比は常に$1$になります。

以上より採択確率は以下のようになります。

$$
	\begin{align}
		{\cal A}(\alpha^{(t)}, \alpha^{new}) = {\rm min}\left\{1, \frac{
			p(\boldsymbol t \mid \boldsymbol w, \alpha^{new})
		}{
			p(\boldsymbol t \mid \boldsymbol w, \alpha^{(t)})
		}
		\cdot
		\frac{\alpha^{(t)}}{\alpha^{new}}{\rm exp}
		\left(
			\frac{(\alpha^{new} - \alpha^{(t)})^2}{2\cdot{\sigma_{\alpha^{(t)}}}^2}
			-\frac{(\alpha^{(t)} - \alpha^{new})^2}{2\cdot{\sigma_{\alpha^{new}}}^2}
		\right)
		\right\}
	\end{align}\
$$

$p(\boldsymbol t \mid \boldsymbol w, \alpha)$の計算をどのように行えばよいかわからなかったので、訓練データからランダムに1行取り出し、現在の$\alpha^{(t)}$と提案分布から生成した$\alpha^{new}$それぞれで$p(\boldsymbol t \mid \boldsymbol w, \alpha)$を計算しました。

さらに論文では$\beta$もデータから推定すると書いてありますが、具体的にどういう手法で推定するかが全く書かれていないため、私は$\alpha$と同様のやり方で推定しました。

## BHMM2

ここまではハイパーパラメータが全て同じ値の対称なディリクレ分布を考えましたが、論文によると$\beta$は可能な品詞の数だけ用意するそうです。

したがって式(25)の$\beta$が$\beta_{t_i}$に置き換わり、式(19)の単語分布が

$$
	\begin{align}
		\boldsymbol \omega^{(t)} &\mid \boldsymbol \beta^{(t)} = (\underbrace{\beta_t,\beta_t,..,\beta_t,\beta_t}_{W_t}) \sim {\rm Dirichlet}(\boldsymbol \beta^{(t)})\\
	\end{align}\
$$

になります。

## 辞書知識の組み込み

誤解のないように書いておくと、式(19)は正確には

$$
	\begin{align}
		\boldsymbol \omega^{(t)} \mid \boldsymbol \beta^{(t)} = (\underbrace{\beta_t,\beta_t,..,\beta_t,\beta_t}_{W_t}) &\sim {\rm Dirichlet}(\boldsymbol \beta^{(t)}) = 
		\frac{\Gamma(\mid W_t \mid \beta_t)}{\Gamma(\beta_t)^{\mid W_t \mid}}
		\prod_{w \in W_t}{\omega_w}^{\beta_t-1}\\
		\boldsymbol \omega^{(t)} &= \{\omega_w \mid w \in W_t\}
	\end{align}\
$$

となっています。

（ここでは品詞$t$ごとに$\beta_t$を用意する場合を考えています）

今回のモデルでは文の位置$i$の品詞$t_i$が決まると、遷移確率$\boldsymbol \omega^{(t)}$に従って単語$w_i$が決定します。

$\boldsymbol \omega^{(t)}$は要素数が$\mid W_t \mid$のため、品詞$t$の単語としてありえないものは出現しないようになっています。

この$W_t$は辞書を用いて事前に決定する必要がありますが、$\boldsymbol \omega^{(t)}$は積分消去されて消えてしまうので、個数$\mid W_t \mid$だけ明らかでどの単語が品詞$t$としてありえるものなのかは分からなくなります。

このように品詞の事前知識を組み込むのは教師なし学習といえるか微妙ですが、すべての$W_t$を$W$と同じにする辞書フリーなやり方もできます。

## 実装

[https://github.com/musyoku/unsupervised-pos-tagging/tree/master/bayesian-hmm](https://github.com/musyoku/unsupervised-pos-tagging/tree/master/bayesian-hmm)

使い方についてはREADMEをお読みください。

## 実験

これ以降の実験ではすべて辞書フリーな状態で行いました。

つまり事前知識を一切使っていません。

また数字は全て##に置き換えています。

## Alice's adventures in wonderland

不思議の国のアリスの原作で実験を行いました。

### $K=7$の場合

まず品詞数$K=7$とした場合の推定結果です。

各単語の後ろの数字はその品詞タグとして出現した回数です。

**tag 0:**

\<bos\>/1634, \<eos\>/1634, 

**tag 1:**

	be/919, to/390, have/344, do/243, n't/217, it/196, '/193, go/178, not/145, you/130, say/122, get/113, will/111, know/107, S/103, think/97, see/96, s/94, on/94, begin/92, look/89, could/86, would/81, come/78, make/76, all/61, find/57, quite/55, hear/45, never/44, must/44, only/42, tell/42, just/42, try/40, seem/40, feel/37, speak/37, can/36, like/36, sit/35, should/32, grow/31, give/30, put/30, ca/28, run/28, might/28, nothing/26, keep/26, call/25, shall/25, soon/25, wo/24, mean/23, eat/22, wonder/21, ,/21, remember/20, they/19, walk/18, she/18, add/17, ever/17, take/15, want/15, stand/14, may/14, turn/14, suppose/14, ought/14, now/13, suddenly/13, always/13, set/12, manage/12, learn/12, hardly/12, please/12, swim/11, use/11, at/11, i/10, beg/9, 

**tag 2:**

	,/2315, ./1198, !/446, ?/190, as/107, out/89, up/72, when/40, wish/21, enough/18, sure/14, forget/13, afraid/12, else/12, whether/11, glad/11, help/10, believe/10, gardener/8, 

**tag 3:**

	she/535, i/534, it/394, alice/379, you/281, and/263, that/138, they/133, he/125, there/99, what/77, who/62, this/61, to/59, then/58, well/50, which/45, oh/43, how/41, why/39, we/34, all/29, course/25, if/24, do/24, so/22, however/20, but/20, perhaps/17, or/15, where/15, would/15, please/14, yes/13, as/13, though/11, when/9, 

**tag 4:**

	the/1632, a/626, her/177, his/96, very/79, this/68, no/66, your/62, my/58, its/57, an/57, their/52, some/42, one/37, any/33, another/20, too/17, these/14, every/12, two/10, beautiful/9, 

**tag 5:**

	little/127, them/88, herself/83, again/83, thing/80, queen/75, time/74, her/71, one/68, down/66, very/65, king/64, me/64, head/60, turtle/60, hatter/57, off/57, mock/55, gryphon/55, much/51, first/51, way/51, voice/51, rabbit/50, cat/50, here/49, more/49, mouse/48, other/47, him/43, duchess/42, now/42, tone/42, large/41, dormouse/40, back/40, great/39, good/39, eye/36, march/35, last/35, hand/35, once/34, reply/34, ask/33, long/33, day/33, minute/32, door/32, dear/32, right/32, hare/31, moment/31, word/31, talk/31, white/30, next/30, two/30, leave/29, foot/29, up/28, turn/28, three/28, out/28, ,/28, caterpillar/27, poor/27, well/27, change/26, remark/26, rather/25, away/25, no/24, same/24, half/23, yet/23, sort/23, jury/22, old/22, use/22, side/21, arm/21, question/21, happen/21, curious/21, write/21, child/21, wait/20, face/20, anything/20, hurry/20, repeat/20, house/19, before/19, end/19, come/19, fall/19, even/18, court/18, table/18, tea/18, 

**tag 6:**

	and/607, of/509, say/410, in/364, to/280, at/201, with/180, that/176, but/150, for/143, as/143, so/127, on/99, all/89, about/78, if/72, into/67, what/64, or/62, by/58, like/56, ,/55, after/43, take/42, such/41, then/36, over/35, from/35, down/35, when/30, round/28, how/27, without/26, upon/26, than/24, let/22, cry/21, till/21, think/20, thought/20, before/19, hold/17, finish/17, while/17, under/16, off/16, because/15, near/15, through/14, behind/13, among/12, join/12, shake/12, shout/11, be/11, either/10, open/10, sing/10, ./10, continue/9, 



これを見ると品詞1には動詞、品詞2は記号、品詞3は名詞、品詞4は冠詞や代名詞、品詞5は名詞、品詞6は前置詞が多く集まってきています。

次に予測された品詞タグに含まれる全単語における正解品詞の割合です。

![image](/images/post/2017-01-28/alice_7.png)

この図は正解品詞ごとに正規化されています。

たとえばVBやVHはほぼ全ての単語が品詞1に含まれていることを表しています。

### $K=11$の場合

次に品詞数$K=11$とした場合の結果です。

**tag 0:**

\<bos\>/1634, \<eos\>/1634, 

**tag 1:**

	it/258, alice/192, go/179, all/176, out/117, look/106, up/100, be/99, see/97, come/97, begin/92, her/91, on/89, them/88, you/84, herself/83, again/83, off/73, say/70, me/68, down/67, one/59, get/53, just/52, much/51, here/51, like/50, do/50, try/45, so/45, not/44, this/43, him/43, now/43, about/42, turn/42, that/41, two/40, back/40, such/37, speak/37, sit/36, well/36, put/34, reply/34, once/34, nothing/34, use/33, ask/33, grow/32, first/31, talk/31, think/30, more/30, leave/29, run/28, course/26, too/26, change/26, rather/25, then/25, yet/25, what/25, away/25, sure/24, no/24, in/24, mean/23, last/23, know/23, half/23, eat/22, old/22, round/22, happen/21, before/21, write/21, wait/20, repeat/20, anything/20, open/19, even/18, quite/18, find/18, soup/18, enough/18, something/18, over/18, seem/17, finish/17, hold/17, high/17, thing/17, which/16, make/16, hastily/16, bill/16, indeed/16, next/16, small/15, set/15, 

**tag 2:**

	and/537, say/462, but/156, that/128, if/96, when/79, as/75, so/71, then/69, what/62, for/51, oh/45, well/41, why/37, think/34, with/31, or/25, add/24, no/23, till/21, thought/20, while/20, however/20, how/18, before/17, perhaps/17, now/17, let/15, dear/15, where/15, because/15, cry/15, please/15, after/14, yes/13, call/11, whether/11, though/11, shout/11, twinkle/9, 

**tag 3:**

	time/68, thing/63, turtle/60, way/52, voice/48, tone/42, minute/32, hare/31, moment/31, door/30, hand/30, foot/29, day/26, word/23, sort/23, arm/21, question/21, side/21, face/20, house/19, end/19, table/18, child/18, court/18, remark/17, air/16, garden/16, bit/15, idea/15, tree/15, size/14, baby/14, mouth/14, sea/14, creature/14, dance/14, life/13, hurry/13, lobster/13, deal/12, tail/12, pig/12, fan/12, box/11, conversation/11, sister/11, hedgehog/10, bird/10, reason/10, pool/10, witness/10, piece/9, 

**tag 4:**

	to/585, of/508, in/343, at/212, and/205, with/150, ,/118, on/104, for/102, into/67, be/65, as/62, by/58, very/53, about/52, that/51, or/48, ./47, like/42, so/35, down/34, have/34, from/29, after/29, !/27, without/26, upon/26, take/26, over/22, round/19, than/16, near/16, under/16, through/14, among/12, join/12, against/9, 

**tag 5:**

	the/1628, a/623, her/157, his/96, this/67, your/62, my/58, its/57, an/57, their/49, no/43, some/41, one/37, very/37, any/29, that/24, another/22, every/12, those/10, these/8, 

**tag 6:**

	,/2247, ./389, !/309, ?/127, as/126, and/31, than/8, 

**tag 7:**

	be/777, have/310, n't/217, do/190, '/185, to/113, will/111, S/103, not/101, s/94, could/86, know/84, would/73, make/60, get/60, think/53, ,/45, must/44, find/39, only/38, never/38, quite/37, can/34, tell/34, should/32, hear/32, take/31, feel/29, might/28, ca/28, give/26, keep/26, seem/23, wish/21, very/21, shall/20, remember/20, wonder/17, walk/17, ever/16, soon/16, want/15, call/14, may/14, ought/14, always/13, hardly/12, wo/11, suppose/10, beg/9, 

**tag 8:**

	./772, !/111, ?/74, 

**tag 9:**

	little/127, queen/75, king/64, head/60, hatter/57, mock/55, gryphon/55, rabbit/50, other/47, cat/47, mouse/45, duchess/42, large/40, dormouse/40, great/39, good/39, eye/36, march/35, very/33, white/30, three/28, caterpillar/27, same/24, jury/22, long/22, curious/21, first/20, more/19, right/19, tea/18, low/17, poor/17, dear/17, footman/15, answer/14, game/14, next/14, book/13, queer/13, whole/13, dodo/13, many/12, serpent/12, slate/12, name/12, pigeon/12, majesty/12, last/12, place/11, grin/11, glove/11, offend/11, rest/11, ear/11, trial/11, soldier/11, cook/11, glass/10, bottle/10, own/10, different/9, 

**tag 10:**

	she/552, i/541, it/337, you/319, alice/204, they/152, he/125, and/97, there/92, that/71, who/55, what/54, how/40, we/34, do/27, which/26, this/24, would/19, to/16, '/15, wo/13, one/9, 

![image](/images/post/2017-01-28/alice_11.png)

## 英語Wikipediaデータセット

英語のWikipediaから10万文のデータセットを構築し学習させました。

また品詞数は30としました。

以下が結果です。


**tag 0:**

	<bos>/200000, <eos>/200000, ,/26, and/7, 

**tag 1:**

	he/10690, it/7052, and/6083, which/3820, but/2960, they/2866, she/2447, there/2304, this/2209, who/1705, however/1676, as/1460, "/1369, although/940, when/852, after/717, while/696, accord/676, @card@/608, then/529, where/501, one/485, though/474, i/473, if/454, some/439, most/423, so/420, also/395, these/365, many/342, or/337, even/336, because/335, what/319, thus/309, later/307, we/306, order/287, such/284, due/265, male/212, since/208, you/189, all/187, today/180, those/175, despite/174, therefore/169, whom/163, 5/162, 1/160, originally/158, instead/156, other/156, people/152, john/142, 6/140, 3/139, finally/139, S/139, before/137, rather/133, eventually/131, )/131, 2/126, 8/123, once/119, 0/113, shortly/111, both/109, 7/109, soon/108, peter/106, 4/104, sometimes/103, 9/102, prior/101, now/99, each/96, much/96, Israel/94, (/93, usually/91, first/89, student/87, along/85, additionally/84, player/84, yet/84, initially/83, zia/83, often/82, team/81, thomas/79, how/78, meanwhile/78, unfortunately/77, together/76, man/76, currently/75, indeed/73, nevertheless/72, Taylor/70, harrer/70, bear/70, peirce/69, perhaps/68, more/67, that/66, just/64, Nintendo/63, leeds/62, whether/61, com/61, danny/60, here/59, jones/59, sora/58, recently/57, William/57, mike/55, russia/55, thereby/54, upon/53, billy/53, follow/53, note/51, afterwards/51, George/50, Michael/50, thrust/50, Craig/50, hence/50, arthas/49, try/49, only/49, -/49, subsequently/48, capablanca/48, say/47, still/46, charles/46, none/46, start/46, apart/46, james/45, unlike/45, not/44, historically/43, especially/43, little/42, child/42, user/42, vegetto/42, depend/42, tv/41, jack/40, consequently/40, emacs/40, jackson/39, louis/39, furthermore/39, wyoming/39, spain/39, possibly/39, found/39, addition/39, Paul/38, o/38, Finland/38, smith/38, similar/38, king/37, Johnson/37, jimmy/37, hope/37, europe/37, don/36, india/36, generally/36, another/36, jat/36, otherwise/36, iran/36, back/36, east/35, bush/35, traditionally/35, karmichael/35, pollifax/35, sam/35, formerly/35, al/34, pearce/34, bell/34, metropolis/34, begin/34, them/34, construction/33, Diaz/33, 

**tag 2:**

	skipton/1, 

**tag 3:**

	;/9, 

**tag 4:**

	orionis/2, 

**tag 5:**

	in/1195, of/611, at/294, by/218, ,/179, '/173, from/128, for/125, with/123, to/118, as/87, on/81, into/29, when/20, and/20, since/18, during/17, about/16, after/16, against/15, than/15, over/15, through/14, until/13, like/12, where/12, ;/11, between/11, among/10, that/8, 

**tag 6:**

	-/5009, ##/4407, first/3434, "/2459, new/2382, unite/1510, other/1447, high/1407, most/1334, large/1310, ##th/1298, American/1206, same/1145, national/1145, second/1133, two/1096, state/998, early/965, small/962, own/906, great/891, old/761, good/747, British/718, last/715, more/680, main/671, major/662, only/650, original/643, few/627, local/618, three/615, late/613, long/600, world/600, international/551, public/540, former/534, young/520, single/508, median/506, third/505, political/500, next/484, short/473, final/470, military/469, very/462, non/458, football/451, general/449, royal/433, English/429, york/426, different/423, low/403, air/395, central/392, north/385, top/385, current/382, popular/382, war/380, western/375, four/375, black/375, full/373, European/369, Canadian/360, total/358, white/352, soviet/352, german/350, television/339, Japanese/337, common/336, south/332, female/330, radio/330, important/326, northern/326, special/324, red/320, lead/320, southern/319, west/317, big/314, modern/311, Australian/306, social/305, famous/304, civil/303, Arab/301, official/300, French/300, average/291, follow/291, human/285, entire/285, Russian/284, film/282, grand/280, real/278, east/274, music/271, professional/269, strong/264, natural/263, Indian/262, previous/260, eastern/259, free/254, art/251, independent/248, home/245, live/243, per/240, league/239, roman/238, blue/237, sport/236, middle/233, video/231, federal/231, catholic/230, Christian/224, rock/223, year/222, wide/221, time/220, economic/218, Spanish/218, Jewish/217, Italian/216, Olympic/215, dark/214, school/212, standard/211, religious/211, physical/208, personal/207, term/205, Chinese/203, successful/203, right/202, primary/202, private/200, ancient/200, significant/198, tv/196, powerful/196, water/195, medical/195, five/194, game/194, us/191, mid/191, county/190, democratic/189, foreign/187, particular/186, police/185, traditional/185, $/183, commercial/183, little/181, naval/181, poverty/181, computer/180, comic/180, scientific/179, fourth/179, fictional/178, husband/176, open/173, similar/173, present/173, security/173, upper/173, ##st/172, power/172, legal/172, six/171, prime/171, regular/170, anti/170, musical/169, city/169, complete/168, one/168, business/167, gold/167, secret/167, light/166, annual/165, family/163, recent/163, Greek/162, metal/162, close/160, 

**tag 7:**

	syphilis/2, 

**tag 8:**

	john/115, Michael/76, la/70, david/68, William/54, Robert/38, steve/38, frank/37, Richard/36, mark/33, peter/33, de/30, henry/29, tom/29, mike/29, jim/28, Paul/28, jean/28, thomas/26, bobby/25, bob/23, chris/23, brian/23, joe/22, lee/22, jack/22, jacques/22, van/21, dan/20, alex/20, ben/19, arthur/19, stephen/19, robert/18, charles/18, james/18, Chris/17, tim/17, sir/17, alexander/16, bill/16, el/16, eric/16, le/16, richard/15, Jeff/14, Charles/14, Thomas/14, Christian/14, te/14, weather/14, dave/14, jimmy/14, george/13, Scott/13, joseph/13, ron/13, Pierre/13, adam/13, rick/12, sarah/12, matt/12, James/12, Jonathan/12, nick/12, harry/12, edward/12, prince/12, Santa/12, na/12, drum/12, aka/12, steven/11, uss/11, andrew/11, billy/11, sean/11, martin/11, kuala/11, near/11, lord/11, Puerto/11, walang/10, mga/10, Vladimir/10, terry/10, christopher/10, maurice/10, leon/10, dennis/10, charlie/10, max/10, gary/10, Jan/9, 

**tag 9:**

	capture/1, 

**tag 10:**

	,/109262, of/71505, and/46078, in/42651, -/16882, (/15259, for/13470, '/12103, on/10100, to/9822, with/9536, from/7477, at/7021, be/6849, as/6136, by/5620, or/4355, )/3590, that/3113, "/3010, ;/2957, during/2066, after/2019, when/1981, between/1658, //1492, where/1390, into/1370, include/1325, under/1108, until/998, against/985, over/946, than/933, before/913, since/852, about/795, through/786, like/574, but/561, within/556, if/534, because/519, while/518, around/497, ##/465, follow/417, call/409, near/395, de/385, among/378, have/327, throughout/322, feature/278, which/260, use/249, million/241, contain/241, across/236, along/235, upon/235, alone/212, without/195, towards/188, versus/187, via/183, win/172, behind/160, despite/157, female/152, above/150, represent/146, per/143, ?/142, outside/137, become/131, show/129, name/127, provide/127, see/118, allow/118, who/118, hold/116, off/115, john/113, %/107, whose/104, involve/104, ii/100, year/93, route/90, unlike/88, receive/87, cause/83, star/77, cover/74, make/70, inside/70, onto/70, toward/66, produce/65, surround/65, act/64, beyond/62, give/61, del/61, entitle/60, amongst/59, comprise/59, below/59, create/58, offer/58, enter/58, although/57, series/55, richard/55, david/55, old/54, von/53, take/53, play/51, champion/51, whether/50, why/50, except/50, require/50, S/48, unless/48, reach/48, concern/48, company/47, nor/45, defeat/45, cross/45, di/44, regard/44, channel/43, alongside/43, later/41, van/41, o/41, mark/40, mean/39, till/38, join/38, Kong/38, score/38, band/38, lose/38, base/37, cup/37, due/37, james/37, guard/37, host/37, sir/37, ask/36, da/36, des/36, meet/36, billion/36, besides/36, aboard/36, span/36, tell/35, attack/35, kong/35, force/34, general/34, mary/34, ibn/34, gain/34, control/34, du/33, km/33, george/33, kill/33, rider/32, whom/32, captain/32, i/31, day/31, once/31, charles/31, record/31, bear/31, face/31, so/30, state/30, et/30, visit/30, introduce/30, declare/29, bin/29, remain/29, m/29, suffer/29, leave/29, der/28, exceed/28, lead/27, 

**tag 11:**

	nakhkhunte/6, 

**tag 12:**

	aham/1, 

**tag 13:**

	)/4202, "/3916, time/3156, year/3028, state/2553, game/2135, school/2097, city/2069, area/1805, name/1793, member/1601, team/1560, ##/1503, number/1497, system/1434, ##s/1393, group/1388, war/1368, university/1363, @card@/1271, town/1252, world/1249, man/1234, line/1232, work/1230, day/1215, series/1209, part/1197, company/1168, country/1105, book/1097, family/1086, population/1079, end/1074, service/1054, people/1047, house/1045, album/1027, song/1019, party/997, force/995, life/990, player/990, band/988, point/973, government/965, way/948, film/919, century/901, character/897, station/891, form/891, island/819, season/810, club/808, one/806, use/796, church/794, result/781, district/778, version/769, power/767, river/759, law/743, side/736, program/733, show/723, income/718, age/718, place/707, term/703, base/702, case/679, court/677, father/669, event/661, building/658, college/658, history/658, death/657, title/655, language/630, village/621, region/620, record/618, child/611, army/605, order/603, son/601, body/595, role/594, period/592, site/582, field/581, king/580, unit/578, community/577, center/570, race/564, league/564, level/557, set/552, word/551, ship/544, office/542, student/541, music/536, election/530, right/525, episode/524, woman/523, battle/512, development/510, division/510, award/509, other/509, model/508, position/505, park/505, effect/505, council/500, home/497, story/493, career/491, act/491, star/487, person/476, problem/474, road/469, attack/468, head/465, leader/464, department/463, type/462, project/462, friend/462, u/459, art/458, hand/457, release/457, design/453, process/449, example/449, study/445, land/442, organization/441, range/441, society/438, style/432, class/428, movie/423, water/422, theory/421, action/419, list/418, nation/416, month/414, president/414, association/413, brother/412, minister/412, union/411, fact/401, success/400, mother/397, championship/392, household/389, ground/386, car/384, movement/383, production/382, degree/379, province/377, track/374, wife/372, rule/372, location/371, interest/370, structure/369, network/365, novel/365, appearance/364, product/364, committee/364, board/364, county/362, centre/360, change/360, ability/360, source/358, census/358, bridge/356, article/355, card/354, performance/353, issue/352, method/351, activity/351, street/347, director/346, boy/344, 

**tag 14:**

	ashur/1, 

**tag 15:**

	##/11090, them/469, S/396, france/285, one/283, well/279, california/258, japan/247, st/243, and/239, him/236, australia/230, england/230, old/228, india/222, individual/217, i/211, europe/194, female/189, e/184, such/176, china/174, canada/174, age/172, germany/168, etc/164, london/159, it/157, along/152, d/148, Canada/148, u/146, all/146, paris/135, italy/134, Israel/128, England/126, especially/123, life/122, science/121, texas/120, russia/116, education/116, other/112, however/110, locate/110, pennsylvania/108, more/108, ohio/106, america/105, b/102, c/98, mathematics/95, p/95, scotland/95, massachusetts/94, each/94, London/93, illinois/92, parliament/91, or/90, rome/85, found/85, base/85, Iraq/84, inc/83, Virginia/83, Dr/82, spain/82, chicago/82, egypt/81, law/81, god/81, fame/80, history/80, a/80, washington/79, music/79, representative/79, Ontario/78, Finland/77, Georgia/77, ireland/77, technology/75, earth/75, particularly/75, iran/74, release/73, director/72, some/71, art/70, up/70, Co/69, general/69, play/69, Usa/68, her/68, Switzerland/67, Maryland/67, Oxford/67, Sweden/66, boston/66, michigan/65, Mr/64, publish/64, English/63, britain/63, wisconsin/62, berlin/62, power/62, austria/61, those/60, most/60, many/60, j/59, wale/59, build/59, singapore/58, physic/58, Florida/57, respectively/57, norway/56, death/56, common/56, that/56, greece/55, mexico/55, water/55, vol/54, light/52, woman/52, philosophy/51, follow/50, Washington/49, California/49, agriculture/49, philadelphia/49, florida/48, graduate/48, brazil/47, turkey/47, r/47, re/47, s/47, serve/47, t/46, self/46, Vietnam/46, f/46, athens/46, jerusalem/45, return/45, search/45, except/45, part/45, Serbia/44, h/44, Asia/44, complete/44, then/44, tennessee/43, economics/43, g/43, religion/43, montreal/43, French/43, space/42, child/42, write/42, connecticut/41, romania/41, k/41, himself/41, possible/41, land/41, founder/41, Sydney/40, man/40, length/40, both/40, daughter/40, o/39, dc/39, afghanistan/39, medicine/39, communication/39, poland/39, maine/39, independence/39, justice/39, Chicago/38, ken/38, melbourne/38, alaska/38, design/38, sign/38, his/38, //38, thailand/37, energy/37, trade/37, 

**tag 16:**

	%/1126, )/369, people/201, year/197, housing/163, km/26, graduate/25, die/23, damme/22, mile/20, percent/17, l/17, Rico/17, minute/14, point/13, goal/12, m/12, ii/12, consist/12, woxing/11, acre/11, Kong/11, Co/10, vegas/10, billion/10, yard/10, parker/10, compete/10, collins/9, 

**tag 17:**

	./98530, )/128, ,/83, "/51, ?/17, and/6, 

**tag 18:**

	barasti/1, 

**tag 19:**

	the/151920, a/46202, his/11016, an/8733, S/6110, s/6013, their/4924, this/4434, its/4180, in/3974, her/2756, "/2234, ##/2022, some/1833, many/1788, other/1774, by/1681, all/1669, these/1570, two/1549, for/1458, one/1389, and/1362, that/1320, no/1296, several/1230, $/1212, with/1191, any/1169, on/1064, as/1031, each/961, another/796, new/739, three/738, of/673, both/649, more/547, to/512, various/487, every/480, at/441, four/440, high/421, most/390, include/386, or/374, my/342, from/342, such/332, five/327, into/321, those/319, our/313, world/309, your/304, which/295, different/281, provide/262, over/255, great/251, numerous/249, good/238, whose/235, certain/229, under/224, very/220, six/215, only/210, modern/207, human/199, like/191, British/191, local/191, team/184, American/183, public/183, see/182, much/178, little/175, large/174, international/172, @card@/170, major/168, multiple/167, former/166, make/162, special/157, old/156, small/155, early/155, -/152, seven/143, long/142, base/141, either/140, national/139, about/133, recent/132, ten/128, produce/114, between/113, further/112, least/111, eight/107, use/105, appoint/104, after/104, heavy/103, through/103, traditional/102, approximately/100, united/99, up/99, increase/98, strong/98, private/98, take/95, Arab/95, full/95, natural/94, play/91, nine/90, low/89, chief/88, free/87, lord/87, general/86, create/86, similar/86, Canadian/86, gain/86, even/85, black/85, foreign/85, among/85, additional/84, (/84, less/82, individual/79, future/78, german/78, but/78, ancient/76, sri/76, government/75, late/75, real/75, cobra/75, super/74, off/74, water/72, study/71, considerable/71, Israeli/71, when/71, computer/69, prime/69, west/68, allow/68, military/68, classical/68, while/68, call/68, white/67, open/67, social/67, reduce/66, extensive/66, receive/65, subsequent/65, than/64, political/63, common/63, how/61, dark/61, avoid/61, what/60, without/60, red/59, separate/59, direct/59, poor/59, official/58, almost/58, Christian/57, specific/57, significant/57, quantum/57, time/57, Japanese/56, serious/56, during/56, develop/55, where/55, north/54, promote/54, twenty/54, regular/53, legal/53, limited/53, 

**tag 20:**

	##/29451, )/6517, "/1223, bear/911, new/705, September/677, march/622, south/600, may/583, October/570, December/566, august/554, north/548, year/526, November/520, january/515, April/509, john/474, county/462, july/446, february/429, June/393, san/355, S/348, %/344, age/341, over/332, (/310, west/307, km/307, york/304, s/302, addition/290, ##th/285, king/272, st/265, William/259, live/251, al/248, east/245, late/242, about/235, america/233, james/223, m/219, every/215, university/213, example/213, season/211, northern/207, henry/205, sir/204, charles/201, mile/201, early/200, general/192, george/190, thomas/185, president/182, include/182, saint/177, los/177, a/171, household/170, foot/164, africa/162, family/160, angeles/154, david/146, i/145, peter/145, college/145, both/144, die/143, reside/143, long/141, least/138, ireland/132, lake/132, la/132, ii/132, francisco/131, prince/129, carolina/129, yard/128, street/128, goal/126, then/122, professor/121, central/119, robert/118, metre/118, mary/116, london/114, see/114, day/113, bc/113, de/113, australia/109, smith/108, wale/107, old/107, lord/107, captain/107, bill/106, june/105, v/104, ##st/103, ?/101, louis/97, paul/96, iii/96, and/95, western/94, b/93, c/92, island/92, fort/91, europe/90, edward/90, richard/90, January/89, port/88, mount/88, student/88, road/87, now/86, around/86, minute/85, route/84, india/82, which/82, mexico/82, mm./82, people/82, baron/81, hour/81, hong/80, e/79, martin/79, square/79, jersey/79, queen/78, fact/78, just/78, southern/78, female/78, n/77, kill/77, approximately/76, Michael/76, ft/76, point/75, brown/75, kong/75, order/74, bay/74, governor/73, hill/73, pope/72, p/71, jean/71, x/71, episode/71, h/70, lady/70, diego/70, meter/69, star/69, joseph/69, earl/69, Asia/69, washington/68, say/68, colonel/68, frank/68, alexander/67, window/67, eastern/67, Kong/67, man/66, class/66, score/66, only/66, one/65, d/65, lieutenant/65, round/65, w/65, jones/65, mark/64, don/64, r/63, f/63, those/63, southeast/63, ad/62, jackson/62, us/61, Daniel/61, defeat/61, 

**tag 21:**

	one/1506, out/877, him/768, up/713, part/700, know/539, them/521, it/471, use/409, her/348, some/269, place/264, work/260, base/248, well/224, locate/220, child/220, play/216, live/198, appear/194, much/193, all/188, re/187, many/186, most/183, release/182, living/177, available/171, away/158, down/152, back/149, himself/149, involve/148, bear/145, find/143, exist/143, again/135, home/129, because/126, today/121, this/118, popular/117, control/113, off/111, responsible/110, build/107, themselves/103, kill/100, participate/100, president/98, present/98, common/96, die/94, compete/92, together/90, result/90, more/89, successful/87, here/86, charge/84, look/81, occur/81, so/80, those/79, there/78, perform/77, associate/75, fight/73, destroy/72, consist/70, active/69, true/69, derive/69, any/68, write/66, unknown/66, me/65, famous/65, change/64, note/64, mention/63, happen/61, possible/61, earth/60, bury/59, serve/59, engage/58, fast/58, capable/57, england/57, different/57, good/57, free/56, see/55, situate/55, nominate/54, stay/53, important/53, interested/52, dead/52, short/52, educate/52, member/51, credit/50, small/50, defeat/50, stand/50, advantage/49, remove/48, record/47, think/46, notable/46, something/46, sell/45, confuse/45, aware/45, clear/45, contact/45, induct/45, large/45, deal/45, alive/44, disappear/44, far/44, close/44, replace/43, accuse/43, qualify/43, chairman/43, divide/43, settle/42, herself/42, two/42, speak/41, arrest/41, say/41, money/41, high/41, fire/41, study/41, convict/40, visible/40, escape/40, develop/40, europe/40, publish/40, useful/39, Islam/39, his/39, low/39, sleep/39, hold/38, else/38, apart/38, instrumental/37, talk/37, strong/37, instead/37, compose/37, select/37, wait/37, enough/37, hear/37, Co/36, difficult/36, evolve/36, display/36, fit/36, list/36, self/35, support/35, alone/35, outside/35, death/35, account/34, along/34, care/34, install/34, name/34, imprison/33, separate/33, concern/33, long/33, survive/33, land/33, withdraw/32, capture/32, twice/32, france/32, catch/31, Christianity/31, characterize/31, operate/31, north/31, right/31, rely/31, focus/31, apply/30, necessary/30, read/30, execute/30, 

**tag 22:**

	nong/2, 

**tag 23:**

	tao/5, 

**tag 24:**

	csc/1, 

**tag 25:**

	./166, liviakis/5, 

**tag 26:**

	al/9, 

**tag 27:**

	)/101, acid/11, sawyer/10, anghel/10, prefecture/10, miller/8, 

**tag 28:**

	be/68328, to/35830, have/15933, as/8834, that/8729, in/8491, by/6994, ,/6605, and/5920, also/5025, not/4779, on/3721, with/3650, "/3382, make/3180, use/3178, it/3038, for/2989, do/2939, from/2849, can/2714, he/2679, become/2600, which/2502, at/2286, who/2277, take/2268, would/2261, only/1749, will/1728, know/1609, they/1595, up/1495, find/1473, give/1431, go/1395, may/1359, of/1353, play/1339, but/1327, him/1315, well/1301, see/1287, into/1263, more/1233, call/1210, begin/1203, come/1192, could/1175, out/1161, such/1159, then/1147, win/1144, serve/1138, include/1102, often/1100, leave/1068, later/1040, or/1031, all/1010, lead/985, write/962, than/951, get/947, after/946, now/935, create/910, still/901, hold/896, so/890, say/884, them/847, over/834, work/829, appear/827, run/817, move/803, consider/802, continue/800, release/799, about/798, return/769, start/722, form/713, there/713, show/710, name/699, receive/696, build/673, allow/668, n't/662, she/659, remain/642, join/638, back/637, never/611, through/610, refer/608, produce/606, due/602, develop/596, set/595, even/594, both/580, help/580, provide/575, bring/571, believe/563, you/557, turn/552, just/548, should/541, lose/537, very/534, first/532, must/530, before/524, tell/519, require/518, i/517, down/516, what/514, describe/511, follow/511, locate/511, usually/507, try/502, change/498, establish/498, currently/496, publish/496, claim/495, kill/491, when/487, cause/478, if/475, like/470, mean/466, able/453, send/445, while/444, again/443, open/442, support/438, put/436, die/434, meet/433, perform/428, reach/427, live/425, )/424, design/421, force/417, carry/417, feature/417, off/417, her/415, much/414, decide/413, grow/412, place/409, keep/408, himself/394, generally/393, enter/389, replace/388, offer/388, break/386, sell/385, operate/381, represent/380, once/377, record/375, complete/370, around/370, want/369, almost/367, how/365, under/365, we/364, think/363, seem/361, discover/358, need/355, choose/354, always/353, eventually/352, fall/352, pass/350, no/350, without/347, marry/345, add/343, originally/342, elect/338, found/335, soon/332, one/330, close/330, too/327, during/327, 

**tag 29:**

	traahi/1, 


興味深いことに品詞8には人名、品詞15には国名、品詞16には単位、品詞20には月を表す単語が集まってきています。

## 青空文庫

青空文庫のテキストデータを用いて10万文のデータセットを構築しました。

また品詞数は30としました。

以下が結果です。


**tag 0:**

	<eos>/200000, <bos>/200000, ｜/1, 
	
**tag 1:**

	
	
**tag 2:**

	て/107575, ながら/4208, ず/3250, たり/2237, たら/1517, ば/1511, なく/621, られ/537, つつ/412, って/342, に/337, たく/303, ざる/303, た/248, り/200, ら/181, で/165, ない/120, ん/97, つ/89, てる/87, たる/83, ぬ/79, せ/75, そう/74, だり/66, させ/58, と/55, ッ/53, る/53, 出さ/52, 得/52, んで/46, さ/40, しめ/39, う/38, たって/36, 合い/35, とも/33, 込み/30, ち/30, さえ/29, え/27, がち/25, れ/25, たい/25, こみ/24, がたく/23, 込ま/22, ちゃ/21, な/20, だって/19, やすく/19, っ/19, じ/19, たけれ/19, 合わ/18, ざま/18, きれ/17, しも/15, づ/15, む/15, えて/15, や/15, だら/14, とう/14, あわ/14, テ/14, かつ/14, っと/14, おり/13, 次第/13, うて/13, 来り/12, ます/12, むると/11, 合わさ/11, 共/11, ざら/11, ゃるか/11, げ/11, 奉り/10, こま/10, ざり/10, わ/10, 做/9, 
	
**tag 3:**

	あろ/1438, しよ/534, 行こ/171, やろ/119, せよ/96, 見よ/95, なろ/94, みよ/67, れよ/53, なかろ/48, 出よ/44, よかろ/35, いよ/34, 来よ/31, 帰ろ/31, 言お/28, 出そ/27, 隠そ/26, 知ろ/25, 殺そ/25, 去ろ/24, しまお/24, もらお/23, 書こ/22, 取ろ/21, 得よ/19, 迎えよ/19, してやろ/16, あげよ/16, かけよ/16, 立てよ/16, はいろ/16, いお/16, 入れよ/15, 逃げよ/15, 置こ/15, 語ろ/15, 見せよ/15, 立と/14, 聞こ/14, かかろ/14, 話そ/14, できよ/14, 求めよ/13, ゆこ/12, 死の/12, つこ/12, 示そ/11, 受けよ/11, 開こ/11, 送ろ/11, 救お/11, 避けよ/10, 上ろ/10, 考えよ/10, 買お/10, 離れよ/10, 動かそ/10, 試みよ/10, 止めよ/10, 待と/9, 
	
**tag 4:**

	
	
**tag 5:**

	##/31257, その/16485, この/8135, お/3646, そして/3332, また/3091, しかし/2688, 『/1514, 大/1496, あの/1486, 御/1310, ただ/1305, もう/1268, ｜/1223, そういう/1054, それから/1028, こんな/891, と/875, そんな/861, まだ/828, 第/814, 大きな/781, やがて/747, 小/718, 諸/710, そこで/692, という/684, 年/657, もし/637, ある/618, あるいは/600, こういう/597, 高/579, しかも/558, 日/549, 若い/544, 小さな/539, 同じ/537, なお/502, 又/493, 右/490, すぐ/490, 長い/462, ちょうど/438, あらゆる/432, 新/423, 張/410, それでも/410, すると/408, 今/399, 白い/398, すなわち/398, 古い/397, けれども/393, すでに/390, 実に/383, 玄/383, やはり/381, 新しい/368, 最も/362, ところが/358, 両/355, まず/353, 深い/350, 呂/349, どんな/349, 美しい/345, キャラコ/344, わが/339, 半/335, 木曾/334, 城/329, 時/327, それで/316, さて/316, だが/315, かの/310, 吉/309, いつも/296, さらに/292, もはや/286, 道/284, 高い/283, 吾/277, すべて/277, だから/275, ほとんど/275, むしろ/274, 決して/270, 劉/270, 日本/266, 寿/254, ご/251, そうして/250, 小さい/249, それに/248, ふと/248, 金/248, …/247, 源/243, よく/243, 関/243, 当時/240, 暗い/237, あたかも/235, つまり/234, まったく/233, 董/231, 老/230, いかにも/229, 全/229, おそらく/228, 藤/227, いよいよ/227, まるで/224, 同時に/223, もっとも/223, いい/222, ついに/220, いかに/219, かつて/218, とにかく/217, （/212, いかなる/212, よい/209, けれど/209, みな/207, どの/207, もっと/204, または/204, 明治/201, 黒い/201, いわゆる/201, うし/199, 宿/196, 無/195, 全く/194, 広い/193, 孫/192, 清/191, いや/189, 旧/186, そう/186, あまり/185, ごとき/185, どうして/184, 古/183, 南/182, かかる/178, 各/178, ことに/177, とうとう/177, 遠い/174, 人/174, 平田/172, こうして/171, ようやく/169, ごとく/169, 不/168, 夜/167, もちろん/166, 黒/166, 王/166, すっかり/165, 少し/164, 実際/163, ああ/163, かえって/162, 恐ろしい/162, ちょっと/161, 宗/160, いま/160, 朝/160, 陳/160, 外国/160, どうも/160, 強い/158, およそ/157, 皆/157, 本/156, 実は/156, 馬/156, 伏見/154, 妻/152, 中津/152, 間もなく/151, しかるに/150, しばらく/150, これから/149, ごく/149, 女/148, 大きい/148, 徳川/148, 特に/148, 俊/147, 』/146, なぜなら/146, 突然/146, 赤い/145, 悪い/143, 香/143, 
	
**tag 6:**

	おいで/47, お尋ね/4, 
	
**tag 7:**

	だろ/1408, でしょ/826, たろ/267, ましょ/147, られよ/13, させよ/12, 返そ/4, 
	
**tag 8:**

	
	
**tag 9:**

	込み入っ/2, 
	
**tag 10:**

	た/108259, ない/10944, ます/3747, ぬ/2695, ん/2181, てる/1849, だ/1795, たい/1379, ず/1343, う/1243, られる/973, なけれ/846, たる/635, れる/626, ざる/595, る/334, 得る/232, ね/229, 候/227, まい/227, うる/213, そう/151, たり/146, がたい/143, い/121, せる/115, り/111, させる/107, 難い/97, し/93, こん/92, つ/89, べし/87, やすい/87, 込ん/84, むる/71, たら/68, です/60, なさい/58, ざれ/57, ゆる/52, 出す/51, ゆ/48, える/46, 居る/45, にくい/45, たれ/44, かねる/44, こむ/40, す/38, 込む/37, よう/37, しめる/36, がち/36, でる/36, ずる/35, よ/33, づ/33, しむ/30, なり/30, 合う/30, ねえ/29, 乍/29, べき/28, き/27, 易い/27, たく/26, つる/26, すぎる/25, まし/25, うれ/24, はじめる/22, ていう/22, ずん/22, 〉/21, だす/21, るる/21, 得/21, 去る/19, つづける/19, ッ/19, なさる/18, 始める/18, まわす/18, 給え/17, ませ/17, 給う/16, 下さい/16, まする/16, 候え/16, きり/16, 置く/15, 難き/15, 申す/15, という/15, 奉る/14, ス/13, ゃる/12, 上る/12, しった/12, せり/11, ける/11, いも/11, かえる/10, しゃ/10, 過ぎる/10, あう/9, 
	
**tag 11:**

	に/119132, を/105484, で/55429, が/55082, は/40335, と/23549, も/18936, へ/11912, から/10643, まで/3364, として/2173, より/1806, でも/1207, か/1174, において/924, について/922, や/854, によって/762, など/677, にとって/502, に対して/467, さえ/461, なら/436, ばかり/424, さ/403, と共に/327, だけ/316, とも/314, ！/303, らしく/302, なく/290, こそ/289, ほど/288, の/283, とか/282, かも/248, しか/239, ｜/227, にたいして/210, し/199, あり/196, とともに/185, ？/181, だの/176, じゃ/171, すら/169, 以来/164, を以て/147, く/144, にて/140, 々/133, にかけて/129, ずつ/128, ながら/123, やら/121, のみ/97, しく/94, により/92, 一つ/82, 以上/80, って/77, ヲ/73, だって/71, ）/71, を通して/70, なり/70, なんか/69, っと/68, 近く/68, 化/66, ニ/66, ぐらい/64, 故/63, たち/59, り/58, つて/58, に従って/58, ［/58, 深く/56, ごろ/54, よ/49, 同時に/49, う/48, とても/47, 自身/46, ち/44, にわたって/43, 迄/41, よく/40, 多く/40, ご/39, 後/39, 相/38, 上/38, みえ/38, たら/37, に関して/37, なぞ/37, だり/36, 早く/36, せら/34, め/34, きり/33, がら/32, ひとつ/32, ゝ/32, 以下/32, だに/31, に対し/31, ゆえ/30, たり/29, ら/29, 自ら/28, らく/28, 余り/28, っ/28, にあたって/28, 高く/27, 程/27, さん/27, につれて/26, 頃/26, 故に/26, くし/25, を通じて/25, ども/25, これ/24, えと/24, ほか/24, 同様/24, につき/23, 視/23, じゅう/23, また/23, 等/22, における/22, 曰く/22, 位/21, をもって/21, しも/21, その他/21, かた/21, 前/21, に従い/21, いわく/21, びく/20, じ/20, 以後/20, 中/19, 蛇/19, づく/19, どおり/18, 輩/18, 通り/18, ぞ/18, 御/17, 気/17, 　/17, 者/17, わるく/16, 三つ/16, なんぞ/16, づたいに/16, 来訪/16, に際して/16, 言う/16, どころか/16, づか/15, 半/15, をめぐって/15, はじめ/15, はね/15, たで/15, ごとき/15, くば/14, ひとり/14, 甚だ/14, 立/14, あまり/14, つき/14, にあたり/14, さま/14, けど/14, ッ/13, 立ち/13, んで/13, 性/13, なんて/13, さらに/13, え/13, にかけ/12, いよいよ/12, ざめて/12, いと/12, いらい/12, 申/11, 遊ばさ/11, っぽく/11, ぶ/11, 入り/11, 歩き/11, 
	
**tag 12:**

	し/30282, い/19248, れ/10694, あっ/9853, なっ/6753, 来/5856, だっ/4675, 見/4004, せ/3296, 出/2760, 行っ/2709, なら/2458, しまっ/2254, いっ/2145, 言っ/1976, あり/1797, 思っ/1733, き/1651, 見え/1541, でき/1518, もっ/1495, 考え/1469, 持っ/1273, でし/1235, かけ/1214, 感じ/1151, 出し/1092, 立っ/1059, つけ/1041, 知ら/987, つい/939, 知っ/931, やっ/924, ござい/888, 得/884, くれ/838, 帰っ/829, 聞い/813, み/805, 書い/804, なかっ/793, 知れ/774, 出来/736, なり/716, 云っ/700, なく/673, かかっ/659, 置い/659, 入っ/647, 見せ/631, 入れ/628, 忘れ/587, 歩い/552, あげ/542, 立て/524, 居/519, 似/496, 受け/469, おい/468, 与え/462, あら/454, 離れ/453, 待っ/449, わから/437, 残っ/436, 取っ/431, 落ち/417, 思い/412, 向っ/400, はいっ/396, ながめ/392, 生き/389, 過ぎ/386, 居り/386, 話し/374, しれ/361, 覚え/350, わかっ/350, すぎ/349, 言い/343, 通っ/339, とっ/335, 思わ/334, 信じ/330, 認め/327, べから/327, 送っ/326, がっ/320, 上げ/318, 答え/311, 開い/310, 上っ/309, 違い/309, あろ/308, 向け/307, 眺め/303, 降り/300, 笑っ/299, ッ/295, なし/292, 申し/285, しまい/285, 切っ/282, 連れ/273, 求め/273, 始め/272, 出かけ/271, え/261, 続い/261, つづけ/258, 現われ/257, 戻っ/257, 黙っ/256, つか/252, 示し/252, 迎え/251, 尋ね/251, なれ/249, 聞え/248, 着/247, うけ/247, なけれ/246, 教え/245, 告げ/243, 近づい/242, 去っ/242, 述べ/240, 買っ/239, 残し/238, やって来/236, 思い出し/233, さし/232, 集まっ/228, 隠れ/227, 起っ/223, なつ/222, 起こっ/222, きい/221, 乗っ/221, 流れ/220, 向かっ/219, おり/219, 逃げ/219, 驚い/219, 寝/218, 用い/216, 伝え/213, 失っ/212, あけ/211, 眠っ/210, 消え/210, 愛し/209, 見つけ/205, いけ/203, 分ら/202, やり/202, しめ/200, いわ/200, 変っ/198, 行き/197, け/197, 捨て/197, 加え/197, 聞こえ/197, たて/195, 坐っ/194, いい/194, 抱い/191, つれ/191, 作っ/189, 走っ/189, め/189, 向い/187, いたし/186, 思え/185, なくなっ/185, 続け/184, らしかっ/183, 集め/182, 殺し/180, 違っ/177, 着い/177, いえ/176, 言わ/176, あらわれ/176, つづい/175, 至っ/175, 行か/173, 生まれ/172, 見つめ/168, 上がっ/168, 寄っ/168, 泣い/167, 働い/166, 変え/164, 打っ/163, もらっ/163, すわっ/162, 語っ/161, 恐れ/161, 生れ/159, 
	
**tag 13:**

	、/189341, は/6169, ##/4113, も/3832, その/2332, 　/1095, お/957, この/934, する/875, （/550, また/520, 御/439, なく/408, 大/348, まだ/280, から/270, …/240, 『/225, ば/212, もう/197, あの/197, 大きな/190, いる/187, ただ/187, 皆/176, 最も/156, 必ず/156, 云う/150, 近い/148, すでに/147, 同じ/142, 実に/136, 又/124, 全く/105, ）/100, 小/97, ほとんど/97, ふ/96, 住む/95, わが/93, すぐ/92, るる/92, ずっと/91, 見える/85, よく/85, 再び/84, 第/83, なお/81, 新しい/81, まず/80, いっそう/78, れる/77, “/72, 決して/70, かなり/69, 大いに/69, むしろ/69, 不/68, つて/68, たった/66, すら/66, どんな/66, 深い/65, すなわち/64, 新/64, かかる/64, 別に/63, 少し/63, 同じく/62, 行く/62, 約/62, こんな/60, 常に/59, 自注/59, 共/58, こそ/56, やはり/55, 一層/53, 早く/52, まで/52, 来る/51, 長い/51, 極めて/51, 今/50, 同/49, し/49, 白い/49, もっと/47, ようやく/46, ども/46, 初めて/45, いつも/45, 小さな/45, ことごとく/44, 一番/44, 林/44, さらに/43, 無/43, 美しい/43, 駕/43, あまり/42, とも/42, 諸/42, 半/41, 大変/40, 多く/40, 黒/40, 水/40, ど/40, 絶えず/39, いかにも/38, きっと/38, およそ/38, 持つ/38, 随分/37, 神/37, やや/36, すこぶる/36, たちまち/35, かえって/35, ごく/35, 直ちに/35, 蛇/35, 特に/34, そのまま/34, 明治/34, 各/34, 薄/34, 米/34, 立つ/33, 殆ど/32, 深く/32, 事務/32, 何だか/31, きわめて/31, 大きい/31, もはや/30, ちょっと/30, 同時に/30, 却って/30, 源/30, 赤い/30, 明るい/30, 実は/29, よほど/29, 強い/29, 平民/29, なかなか/28, たしかに/28, いろいろ/28, はなはだ/28, 長/28, 西洋/28, 金/28, 両/28, 更に/27, 毎日/27, 甚だ/27, それぞれ/27, いちばん/27, 思う/27, ある/27, 宿/27, まっ/27, 勘/27, 自ら/26, はじめて/26, ちょうど/26, じっと/26, 〈/26, つづく/26, 激しい/26, 角/26, 出る/26, 暗い/26, 城/26, 成る/25, そっと/25, しきりに/25, 突然/25, 大分/25, 益/25, はじめ/25, 小さい/25, 日本/25, そういう/25, ついてる/24, 次第に/24, 徳川/24, ゆく/24, めぐる/24, 通ずる/24, なき/24, 呼ぶ/24, 鉄/24, 竜/24, 青い/24, ほんの/23, いわゆる/23, 朝/23, 真/23, 
	
**tag 14:**

	
	
**tag 15:**

	なかっ/6726, まし/5225, られ/3571, ませ/1264, 出し/274, だし/219, させ/219, 得/219, 始め/187, ましょ/180, っ/174, はじめ/149, たかっ/148, きっ/102, 合っ/93, たろ/86, しめ/86, で/76, れ/75, きれ/72, らし/70, かね/63, すっ/55, ら/54, つづけ/43, こん/42, かかっ/42, なく/42, まわし/41, 去っ/35, 初め/34, 込ん/34, え/33, かけ/33, やし/32, たく/31, 直し/31, ま/23, 続け/21, な/21, あたわ/21, られよ/20, ます/19, すぎ/19, 寄っ/19, なかろ/18, 能わ/17, きら/17, まもっ/16, ぬい/16, 申し/15, 上っ/15, っし/13, きたっ/13, ざら/12, ざり/12, 来っ/12, て/12, がたかっ/11, わたっ/11, 切ら/11, 果て/11, 奉り/10, 上げ/10, つくし/9, 
	
**tag 16:**

	は/16871, も/13276, さ/3045, と/2078, なく/1969, あり/1493, よく/1471, また/1220, そう/1142, 思わ/956, なり/941, お/751, こう/607, まだ/599, 少し/572, もう/525, どう/506, でも/458, さえ/435, 言わ/412, 言い/371, まで/359, いい/358, すっかり/350, しか/321, せら/312, なお/303, 一つ/289, 見/288, 多く/283, じっと/282, 想像/279, 深く/266, はっきり/264, 相/262, 結婚/258, すぐ/256, 思い/255, 皆/248, のみ/245, いわ/244, 今/244, おり/243, 早く/236, なんと/231, 引き/230, 決して/225, あまり/224, 大きく/224, 立ち/221, 駈け/214, 許さ/213, 行き/207, 高く/206, ただ/205, しばらく/204, 打ち/202, 取り/201, ひどく/196, 遠く/196, 行わ/192, みな/191, すこし/191, ちょっと/188, ばかり/186, 発見/183, 安心/183, 実に/173, 知ら/171, ほとんど/170, 長く/169, 書き/168, 呼び/165, 心配/164, 注意/162, 逃げ/162, 云い/161, いつも/160, すでに/158, みずから/158, 聞き/157, つき/157, なかなか/157, 持ち/155, 通り/155, 押し/154, 怖/154, 云わ/154, 呼ば/151, 大いに/149, 理解/149, やや/148, 全く/147, いかに/146, かえって/145, ずっと/143, びっくり/143, 書か/143, 満足/143, 打た/140, 大変/140, 現/140, まったく/139, 吹き/137, 聞か/135, さして/134, ごとく/133, すら/131, いろいろ/131, もっと/131, 飛び/130, 読み/129, 殺さ/129, 最も/129, 説明/129, 又/129, 初めて/128, 悪く/127, 振り/127, うち/127, ##/126, 自ら/124, 再び/123, 取ら/122, しばしば/121, 強く/119, さし/119, ゆき/119, か/119, 斬り/118, とり/117, 馳/116, ああ/115, なし/115, ふと/112, 行なわ/112, 出発/112, 出さ/111, なさ/111, 持た/110, ますます/110, 共に/110, そっと/109, ぼんやり/109, ふり/109, 案内/108, 追い/108, どんなに/107, どうか/107, 一寸/106, 入り/106, 泣き/106, さらに/106, かつて/106, 知り/106, はじめ/106, 同時に/105, いよいよ/105, 決心/105, 歩き/105, 走り/104, 跳び/104, 白く/104, そのまま/103, 引/103, どうして/102, 語り/101, 支配/99, どうしても/99, 申/99, 存在/98, 近く/98, 笑い/98, 同じく/98, あまりに/98, だんだん/97, かなり/97, 利用/96, 置き/96, ッ/96, もち/95, やり/94, 用意/94, 仰/94, しろ/94, やはり/93, 本当に/93, 置か/92, 軽く/92, 考え/92, 振/92, 突き/91, やがて/87, ようやく/87, 次第に/86, 承知/85, 赤く/85, ひき/85, 開か/84, 激しく/84, かく/84, 
	
**tag 17:**

	ある/17742, いる/14737, ない/7699, する/7558, なる/2658, いう/2176, 見る/2108, 来る/1738, れる/1581, 思う/1502, 行く/1491, いい/877, ゆく/858, 居る/842, くる/796, 何/758, せる/704, す/656, みる/616, 言う/606, しまう/576, 見える/575, よい/539, いえ/508, な/508, 同じ/498, 出る/490, すれ/487, 云う/474, やる/444, できる/438, るる/387, あれ/374, くれる/369, 聞く/364, 多い/356, 知る/311, 考える/307, どう/298, 見れ/288, あり/286, 出す/268, 出来る/265, つく/256, なれ/239, ね/224, くれ/223, 帰る/222, 立つ/215, なす/195, なけれ/195, かかる/191, わかる/189, 入る/188, …/186, よる/184, おく/183, 至る/183, 書く/179, 通る/170, 面白い/161, 悪い/159, いらっしゃる/158, 得る/153, 与える/152, 置く/150, かける/143, 見せる/142, 死ぬ/140, 呼ぶ/139, 下さい/138, る/137, どこ/132, 示す/132, 感ずる/130, 近い/127, 私/127, 感じる/126, 取る/125, 持つ/123, 言え/119, 死ん/119, 食う/116, 歩く/114, 待つ/114, 申す/113, 去る/112, 無い/112, 語る/111, よ/110, そんな/110, なり/107, きく/107, から/104, 入れる/104, とる/97, 読む/97, うれしい/93, 愛する/93, つける/93, そう/92, こんな/92, 受ける/91, 分る/90, 違う/88, 当然/88, 上る/87, や/87, よれ/86, みれ/85, あげる/85, 近づく/85, 驚く/85, 有する/83, 及ん/82, 終る/81, もらう/79, 走る/79, 迎える/79, 下さる/78, 住む/78, 動く/78, こん/78, ほしい/76, あ/76, 笑う/75, 欲する/73, 立てる/73, 殺す/72, 会う/72, 込ん/72, 早い/71, いく/70, たまらない/70, 及ぶ/70, 問う/68, ッ/68, かえる/67, 上げる/67, もつ/67, だ/67, 話す/66, 行け/66, よろしい/66, 過ぎる/65, どんな/65, 求める/63, 選ん/63, 泣く/62, 出かける/62, 苦しい/62, 大きい/62, 信ずる/62, 起る/62, せよ/62, く/62, 候/61, 答える/61, やって来る/61, 思え/61, ん/61, 長い/60, 読ん/60, いえる/60, 望む/59, 少ない/59, わるい/59, かく/59, 深い/59, 送る/58, たつ/58, 明/58, 困る/57, 働く/57, 忘れる/57, いや/57, 誰/57, 見つける/56, 開く/56, 聞える/56, 眠る/55, 打つ/55, 頼む/55, 当る/55, 思い出す/55, 動かす/54, 悲しい/54, みえる/54, 呼ん/53, という/53, 着く/53, あたる/53, 啼く/52, 飲む/52, 作る/52, どれ/52, 生きる/51, 用いる/51, 失う/51, ちがう/51, 珍しい/51, 
	
**tag 18:**

	
	
**tag 19:**

	団/6, 
	
**tag 20:**

	の/190329, は/34269, な/21843, が/10963, と/6381, という/5738, から/4464, や/4019, も/3379, ｜/2653, ##/1850, ・/1448, なる/1309, （/927, か/869, たる/754, 之/705, らしい/690, ノ/581, 的/555, でも/486, として/427, る/373, に対する/338, ほど/324, より/323, べき/297, とも/271, 』/267, い/251, だ/248, 、/244, する/233, 州/207, における/191, によって/170, および/163, に関する/162, なき/157, ッ/155, まで/151, ある/144, にたいする/141, しい/136, ）/118, く/117, こそ/117, 々/116, ながら/115, だの/112, に対して/109, ぐる/106, による/105, ずつ/103, とともに/102, と共に/100, 羅/93, ヶ/92, といった/89, とか/88, 学/86, にあたる/79, やら/77, 子/76, し/72, を以て/71, 志/70, なら/70, 張り/69, 居/63, っ/63, または/63, 経/62, ばかり/62, において/62, 史/59, 郷/59, とかいう/59, じ/54, ら/54, 以来/53, でる/52, ちの/52, にて/51, 記/50, 乃/50, ない/49, 女/49, 中/49, よく/48, 生/46, しく/44, …/44, 巻/43, 光/43, 公使/42, 集/42, とても/42, 深い/41, っぽい/39, 庵/39, き/38, たち/38, だって/38, 籠/37, 謂/36, もしくは/36, 没/36, 諸/36, 馬/36, ハ/36, ぶ/35, 主義/34, につき/34, 甲/33, 侯/32, 掘り/32, 及び/32, 第/32, すなわち/32, ヲ/32, にわたる/31, ち/31, しも/31, ん/30, 前/30, じゅう/29, 師/28, 修道/28, ならびに/28, 吉/28, ば/28, 高/27, 伝/27, なり/27, きわまる/26, らし/26, 院/26, 殿/26, 論/26, 部/26, 函/25, ガ/25, 照らす/25, 法/25, 家/25, お/23, 秀/23, 典/23, エル/23, 守/23, 程/23, 〔/23, 等/23, っと/23, 談/22, 県/22, 即ち/22, 掛/21, ぼ/21, 兼/21, ちゃ/21, 島/21, 武/21, 頃/21, を通して/21, やすい/20, 目/20, せる/20, 臭い/20, たらし/20, 役人/20, 金/20, 迄/20, 録/19, 義/19, 鑑/19, つ/19, ごろ/19, はじめ/19, のみ/19, かしい/18, 恋/18, 教/18, 考/18, 提督/18, 問屋/18, って/18, 六月/18, くさい/17, に/17, がたい/17, らしき/17, 式/17, 華/17, 注/17, 市/17, 物語/17, みな/17, 行/17, をもって/17, 
	
**tag 21:**

	嘘/4, 
	
**tag 22:**

	命/495, 神/368, 王/277, 姫/189, 彦/137, 國/126, 天皇/126, 御子/112, 宮/104, 祖先/88, 次に/88, 君/84, 臣/83, 等/79, 女/72, 天/71, ）/69, 郎女/65, 子/62, 囲炉裏/61, 大和/60, またの名/46, 松平/40, 弟/39, 妹/34, 主/34, その他/33, 別/33, 国/33, 大/31, 造/30, 大神/30, イザ/30, 一方/28, 兄/28, 大塔/27, 皇后/27, 連/26, 様/25, 近江/24, 天下/24, □/24, 宿禰/23, ナギ/23, 守/23, 樣/22, 木/20, 若/20, シ/19, 原/19, 御陵/18, 武/18, 大國/18, 問屋/17, 出雲/17, スサノヲ/17, 部/17, 尊/17, 伊勢/16, 山/16, 伊賀/15, など/15, 葦原/15, チョイ/14, 炉/14, および/14, 紀州/14, 本陣/14, 葛城/14, ギリシア/14, ヲ/14, オホハツセ/14, 越/14, 穗/14, サホ/14, 石/14, アドルフ/14, 島/14, 尾張/13, 丹波/13, 河内/13, 野/13, 花/13, 名/13, 良/13, 筑紫/12, 廃止/12, 吉備/12, 院/12, 中心/12, 柳生/12, 庄/12, 類/12, 信濃/12, 庄屋/11, 正/11, 春日/11, 高木/11, 山城/11, マルタン/11, 師/11, ごとき/11, 美濃/11, 伯耆/10, 直/10, 大納言/10, ヴェルガ/10, 御名/10, 播磨/10, タケシウチ/10, ヤマトタケル/10, ミ/10, 橘/10, ヤマト/10, 池/10, 小/10, つて/10, 鹿/9, 
	
**tag 23:**

	見知らぬ/1, 
	
**tag 24:**

	神/6, 
	
**tag 25:**

	
	
**tag 26:**

	。/100000, う/8139, なけれ/3, 
	
**tag 27:**

	人/11882, 彼/10009, それ/8507, 私/7325, よう/6827, 自分/5749, 中/5002, もの/4810, こと/4691, これ/4182, 方/3888, 日/3433, 家/3321, 上/3144, 彼女/2840, 心/2680, ##/2679, 前/2660, そこ/2626, 手/2594, 顔/2389, 葉子/2337, 者/2295, 何/2208, 時/2202, 眼/2140, うち/2072, 女/1987, 目/1938, ため/1926, 年/1923, 男/1888, 声/1885, さん/1825, 間/1821, 半蔵/1790, 氏/1735, 今/1725, さ/1699, 的/1637, 気/1628, ここ/1624, クリストフ/1597, 身/1595, 下/1457, 一つ/1433, ところ/1428, 名/1423, 口/1334, 頭/1311, 言葉/1290, 話/1268, 姿/1253, 馬/1241, 人間/1231, 事/1209, 子供/1184, 道/1119, 彼ら/1119, 後/1118, 武蔵/1103, 町/1097, 夜/1076, 生活/1039, 父/1018, 物/1001, 所/971, 村/965, 力/944, 』/939, 胸/936, 音/934, どこ/931, 先/928, 人々/904, 手紙/904, 山/903, 他/893, 水/891, 風/872, 屋/869, 母/850, 子/834, 本/831, わたし/823, 門/813, 足/810, 娘/808, あなた/807, 耳/806, 次/777, 地/773, そう/768, ほう/767, 国/750, 部屋/744, あと/740, 外/734, 今日/734, 仕事/708, 日本/706, 度/706, 竜/700, 時代/685, 倉地/678, 助/673, そば/666, 先生/665, 花/664, 色/657, それら/657, 兵/647, 神/641, 船/629, ほか/627, すべて/619, 金/617, 多く/616, 室/608, 非常/592, 例/584, 面/579, 体/579, 敵/572, 光/568, 形/566, 世界/564, 石/564, 蛇/561, 火/551, 歳/549, 君/544, 空/536, 朝/533, 夫人/532, 誰/529, 内/527, 曹操/527, 酒/515, 影/513, 時間/511, 老人/509, だれ/508, 妻/505, 不思議/499, 様子/497, 意味/497, 二つ/488, 頃/487, なか/485, 涙/485, 窓/484, 壁/484, 夢/483, 地方/482, あたり/481, 場/481, 雨/479, 街道/476, 通/475, 血/472, 々/471, いろいろ/470, 民/469, 少し/466, 軍/462, 首/461, 静か/459, 店/456, 草/456, 急/449, 自然/443, 僕/443, 木/442, 相手/441, 庭/440, 死/438, 徳/436, 考え/433, 旅/432, 藩/431, 感情/429, 階/428, 奥/427, 客/422, 社会/422, 腕/420, 布/417, いずれ/416, なん/414, 場合/411, など/411, 海/411, 昔/409, 帝/409, 晩/408, 川/407, 江戸/404, 太郎/403, 問題/402, 様/401, 
	
**tag 28:**

	玄米/1, 
	
**tag 29:**

	と/26794, こと/12630, だ/9762, か/9127, よう/7053, もの/6598, ば/4787, ので/3805, です/3684, から/3629, たち/2882, 時/2560, だけ/2445, ところ/2273, ほど/2082, など/2058, という/1942, 者/1825, まで/1746, ばかり/1675, ら/1621, し/1586, 事/1508, ため/1305, さ/1177, べき/1154, そう/1108, で/1042, ）/987, らしい/919, として/855, ね/824, ん/747, まま/745, 的/744, なり/727, のに/704, とき/702, わけ/700, ども/679, なぞ/603, かも/600, のみ/565, なら/547, うち/542, けれども/530, より/527, まい/499, よ/481, ころ/460, ！/445, 自身/439, はず/437, へ/427, くらい/425, つて/384, ？/376, …/374, 以上/373, 所/368, 』/360, や/350, 人/330, 物/327, 後/313, 間/309, つもり/304, 様/300, 様子/288, ごと/285, なし/280, ぐらい/277, 気/270, 頃/267, 々/254, とか/245, らしく/240, 達/233, 位/232, みたい/220, ぞ/217, とも/215, 風/213, 由/212, 以外/211, ふう/211, ものの/209, 通り/208, ぶり/204, 等/203, あり/197, 方/194, さま/192, 前/190, 場合/189, 力/188, べし/187, ごろ/187, 話/181, ”/175, けれど/169, がた/167, とおり/166, 必要/166, つき/165, 限り/157, きり/155, 処/155, 日/154, す/153, について/147, 筈/147, たび/146, 心持/145, たる/142, かしら/140, ごとく/140, 故/136, って/134, ど/132, 点/129, べく/127, 人々/127, 性/124, 時代/122, 程/121, な/119, じゅう/119, そのもの/118, ほか/116, 男/115, あたり/114, 次第/108, 否/107, 声/107, あと/107, やら/106, 座敷/104, め/99, 衆/99, 気持/96, 共/94, 訳/94, に/92, 道/91, 夫婦/89, 音/89, 〉/87, ま/87, において/85, 思い/85, ゆえ/84, 機会/84, 内/83, かぎり/82, 如く/82, 付/81, なる/80, 言葉/80, 法/79, 羅/76, せい/75, 仲間/73, 状態/73, 上/72, がら/69, 例/69, げ/69, にとって/68, たり/68, なき/68, 丈/68, 姿/68, もん/67, じゃ/66, さん/66, れ/65, 側/65, ずつ/65, 顔/64, 場所/64, 郷/63, 程度/63, が/62, につれて/62, 奴/62, 時分/61, 介/61, 感/60, 全体/60, 越し/60, 迄/59, 際/59, ぼ/59, 感じ/58, 顔つき/58, 心/58, 人間/57, け/56, かた/56, 証拠/56, 
