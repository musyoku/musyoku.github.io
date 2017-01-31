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

ここではコインにバイアスがかかっている事後確率$P(t=1 \mid \boldsymbol w, \theta)$を求めます。

表が出た回数を$n_H$とすると最尤推定では$\theta=\frac{n_H}{10}$としますが、ベイズ的なアプローチでは$\theta$を積分消去した確率$P(t=1 \mid \boldsymbol w)$を考えることで、$\theta$によらないロバストな推定を行うことができます。

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

$K$種類の単語があったとき、それぞれの単語が出現する確率を、パラメータ$\boldsymbol \theta=(\theta_1,...,\theta_K)$の多項分布で表します。

この場合、単語$w_k$が出現する確率は$\theta_k$になります。

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

全ての$\theta_k$を$1/K$とする一様分布を事前分布にしても良いのですが、ディリクレ分布を事前分布に用いると、データを観測する以前の状態で、こういう観測結果が得られるだろうという仮想的な観測結果をモデルに組み込むことができます。

たとえば単語$w_k$は出やすいから事前に5回くらい観測したことにしておこうといった感じです。

また多項分布の共役事前分布はディリクレ分布であるため、観測結果$\boldsymbol x$を得たもとでの事後分布は$p(\boldsymbol \theta \mid \boldsymbol x) = {\rm Dir}(\alpha_1 + n_1, ..., \alpha_K + n_K)$とディリクレ分布になるため扱いやすいのが特徴です。

論文では簡単のためにsymmetricなディレクレ分布を使うとありますが、これはおそらく$\alpha_1,...,\alpha_K$の値が全て同じことを意味しています。

これらを踏まえて観測したデータのクラス確率を考えます。

クラス数を$K$とし、$i-1$個のデータ$\boldsymbol x=x_1,...,x_{i-1}$を観測したとします。

また各クラスの出現回数を$n_1,...,n_K$で表します。

クラス変数はパラメータ$\boldsymbol \theta=(\theta_1,...,\theta_K)$の多項分布に従っており、クラス$k$の出現確率は$\theta_k$で表されます。

また$\boldsymbol \theta$の事前分布としてパラメータ$\beta$の対称なディリクレ分布

$$
	\begin{align}
		P(\boldsymbol \theta) = {\rm Dir}(\underbrace{\beta,..,\beta}_{K})=\frac{\Gamma(K\beta)}{\Gamma(\beta)\cdot\cdot\cdot\Gamma(\beta)}{\theta_1}^{\beta-1}\cdot\cdot\cdot{\theta_K}^{\beta-1}\\
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

次に$\boldsymbol \tau^{(t,t')}$と$\boldsymbol \omega^{(t)}$を積分消去するのですが、これは式(13)の変数を変えるだけで求められます。

$$
	\begin{align}
		P(t_i \mid \boldsymbol t_{-i}, \alpha) &= \frac{n_{(t_{i-2}, t_{i-1}, t_i)} + \alpha}{n_{(t_{i-2}, t_{i-1})} + \mid T \mid\alpha}\\
		P(w_i \mid t_i, \boldsymbol t_{-i}, \boldsymbol w_{-i}, \alpha) &= \frac{n_{(t_i, w_i)} + \beta}{n_{t_i} + \mid W_t \mid\beta}\\
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
		\frac{n_{(t_{i-2}, t_{i-1}, t_i)} + \alpha}{n_{(t_{i-2}, t_{i-1})} + \mid T \mid\alpha}\cdot
		\frac{n_{(t_{i-1}, t_{i}, t_{i+1})} + I(t_{i-2} = t_{i-1} = t_i = t_{i+1}) + \alpha}{n_{(t_{i-1}, t_{i})} + I(t_{i-2} = t_{i-1} = t_i) + \mid T \mid\alpha}\cdot\nonumber\\
		\frac{n_{(t_{i}, t_{i+1}, t_{i+2})} + I(t_{i-2} = t_i = t_{i+2} , t_{i-1} = t_{i+1}) + I(t_{i-1} = t_i = t_{i+1} = t_{i+2}) + \alpha}{n_{(t_{i}, t_{i+1})} + I(t_{i-2} = t_i , t_{i-1} = t_{i+1}) + I(t_{i-1} = t_i = t_{i+1}) + T\alpha}
	\end{align}\
$$

$I(\cdot)$は引数が真のときに1を返し、それ以外の場合は0を返す関数であり、トライグラムカウントや品詞-単語ペアのカウントを補正する項になっています。

$t_i$をギブスサンプリングする際、まずnグラムカウントから$t_i$を削除する必要があります。

その際にカウントを1下げるテーブルは$n_{(t_{i-2}, t_{i-1}, t_i)}, n_{(t_{i-1}, t_{i}, t_{i+1})}, n_{(t_{i}, t_{i+1}, t_{i+2})}$の3つです。

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
		\alpha^{(t+1)} \gets
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

したがって式(25)の$\beta$が$\beta_{t_i}$に置き換わり、式(19)の単語遷移確率が

$$
	\begin{align}
		\boldsymbol \omega^{(t)} &\mid \boldsymbol \beta^{(t)} = (\underbrace{\beta_t,\beta_t,..,\beta_t,\beta_t}_{\mid W_t \mid}) \sim {\rm Dirichlet}(\boldsymbol \beta^{(t)})\\
	\end{align}\
$$

になります。

## 辞書知識の組み込み

誤解のないように書いておくと、式(19)は正確には

$$
	\begin{align}
		\boldsymbol \omega^{(t)} \mid \boldsymbol \beta^{(t)} = (\underbrace{\beta_t,\beta_t,..,\beta_t,\beta_t}_{\mid W_t \mid}) &\sim {\rm Dirichlet}(\boldsymbol \beta^{(t)}) = 
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

	<bos>/1634, <eos>/1634, 

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

<img src="/images/post/2017-01-28/alice_7.png" height="800px">

この図は正解品詞ごとに正規化されています。

たとえばVBやVHはほぼ全ての単語が品詞1に含まれていることを表しています。

### $K=11$の場合

次に品詞数$K=11$とした場合の結果です。

**tag 0:**

	<bos>/1634, <eos>/1634, 

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

<img src="/images/post/2017-01-28/alice_11.png" height="800px">

## 英語Wikipediaデータセット

英語のWikipediaから10万文のデータセットを構築し学習させました。

また品詞数は30としました。

以下が結果です。

**tag 0:**

	<bos>/200000, <eos>/200000, and/4, 

**tag 1:**

	he/9414, it/6487, and/5570, which/3663, but/2722, they/2530, she/2187, this/2033, there/1805, who/1607, as/1555, however/1532, "/1323, although/744, accord/633, when/612, after/587, @card@/579, while/548, then/472, also/460, so/422, one/407, i/399, though/393, or/379, if/360, some/358, what/336, these/336, most/332, even/323, later/291, order/287, we/285, where/284, thus/281, due/273, many/272, because/264, )/226, you/210, male/205, since/184, those/165, 5/164, therefore/159, whom/155, 1/154, people/154, today/144, now/142, originally/140, 6/140, S/140, john/138, rather/136, before/135, 3/134, eventually/131, finally/130, all/129, other/127, instead/125, 2/124, despite/123, 8/122, 0/112, 7/108, sometimes/107, shortly/106, soon/103, prior/100, student/100, usually/100, 9/100, 4/100, just/93, man/92, both/89, peter/89, once/87, (/87, much/87, how/86, team/85, try/83, player/82, bear/81, currently/79, additionally/78, first/78, Israel/77, thomas/77, yet/76, them/76, meanwhile/75, unfortunately/71, each/71, indeed/71, initially/70, zia/70, harrer/65, along/65, together/62, nevertheless/61, similar/61, upon/60, more/60, Nintendo/59, whether/58, perhaps/57, note/56, here/56, com/56, peirce/56, Taylor/55, William/54, start/54, Michael/52, leeds/51, George/50, addition/50, attempt/50, mike/49, russia/49, billy/49, sora/48, danny/48, thereby/48, subsequently/47, charles/47, thrust/47, Craig/47, -/47, recently/45, capablanca/45, jones/45, arthas/45, generally/44, child/43, example/43, louis/42, hope/41, king/40, afterwards/40, vegetto/40, apart/40, still/39, furthermore/39, hence/39, back/39, again/38, found/38, jack/37, unlike/37, historically/37, apparently/36, o/36, Johnson/36, bell/36, little/35, don/35, previously/35, user/35, consequently/35, jimmy/35, anyone/35, iran/35, god/35, only/35, james/34, such/34, construction/34, let/34, pearce/34, response/34, east/33, Diaz/33, pollifax/33, Finland/33, emacs/33, immediately/33, al/32, karmichael/32, metropolis/32, lady/32, possibly/32, land/32, Paul/31, critic/31, moreover/31, wyoming/31, none/31, smith/31, light/31, another/30, pi/30, specifically/30, jat/30, visitor/30, 

**tag 2:**

	./75, uncertain/3, 

**tag 3:**

	./12, Goldwyn/4, 

**tag 4:**

	,/19, and/3, 

**tag 5:**

	in/7651, on/1481, by/1055, of/774, for/733, at/701, from/568, ,/382, to/382, with/342, as/303, '/220, (/162, between/159, since/158, about/154, until/147, after/137, over/110, during/84, than/83, around/74, when/64, into/60, approximately/57, "/56, sir/47, and/46, route/44, only/44, like/36, before/34, john/34, under/32, feature/31, against/30, through/29, san/29, near/25, within/25, @card@/24, ;/23, where/23, charles/23, be/20, saint/20, number/20, premiership/20, )/19, upon/18, prince/18, follow/18, bear/18, eastern/17, include/16, among/16, reach/16, that/15, William/15, -/14, October/14, captain/14, professor/13, general/13, name/12, fill/12, robert/12, become/11, south/11, km/11, just/11, across/10, henry/10, walter/10, nearly/10, unlike/10, James/10, locate/10, throughout/10, while/10, article/10, march/10, st/10, james/10, if/9, 

**tag 6:**

	##/4073, -/3925, first/3376, "/2483, new/2243, unite/1491, other/1341, high/1309, large/1282, most/1272, ##th/1251, state/1206, American/1174, same/1141, second/1112, national/1104, two/1058, early/914, own/897, small/886, great/841, good/719, old/706, last/704, main/669, world/667, British/654, original/644, more/641, only/631, few/630, major/626, late/584, local/568, three/562, long/546, single/522, young/514, international/509, third/509, median/506, former/502, final/483, next/479, public/479, very/469, short/458, football/458, political/441, royal/436, military/434, general/434, york/423, English/418, different/394, war/390, air/389, low/384, central/381, popular/374, television/374, current/372, top/371, north/370, four/369, soviet/355, non/354, black/351, Canadian/350, total/349, western/346, full/346, European/343, german/339, radio/329, music/325, white/324, female/322, important/321, south/320, southern/318, common/318, lead/317, northern/314, special/311, art/308, west/306, follow/298, famous/298, official/297, big/296, Japanese/296, league/291, home/291, civil/290, red/289, Australian/288, entire/285, east/284, French/284, modern/279, film/274, grand/273, Arab/267, Russian/266, social/264, previous/259, real/257, strong/255, eastern/250, video/250, Indian/242, middle/242, sport/242, independent/242, rock/239, human/239, professional/236, per/236, blue/232, natural/229, catholic/228, roman/227, economic/226, Olympic/225, live/224, free/217, Italian/214, term/212, wide/212, standard/211, county/211, Spanish/210, successful/209, federal/208, five/204, dark/202, Christian/201, water/199, primary/197, significant/195, right/194, Jewish/194, democratic/193, tv/192, private/190, research/188, mid/187, powerful/187, present/186, comic/184, scientific/184, personal/183, us/182, poverty/182, police/180, ancient/180, fourth/180, medical/179, traditional/178, little/177, summer/176, particular/176, physical/176, religious/176, fictional/176, Chinese/175, security/175, railway/174, husband/174, school/174, ##st/172, computer/172, complete/172, commercial/172, energy/168, metal/168, close/165, year/165, time/164, open/162, prime/161, regular/161, annual/160, secret/159, naval/159, upper/159, similar/158, remain/158, light/157, foreign/157, golden/157, $/156, recent/156, gold/155, six/154, Greek/153, dutch/153, business/152, game/151, musical/151, 

**tag 7:**

	syphilis/2, 

**tag 8:**

	john/114, Michael/75, david/72, la/69, William/65, jean/39, frank/38, Robert/36, Richard/36, peter/36, steve/36, mark/34, de/34, jim/29, mike/29, Paul/28, thomas/28, henry/27, tom/27, bobby/25, bob/25, chris/25, brian/24, jack/23, lee/22, joe/21, van/21, el/20, jacques/20, dan/19, robert/19, ben/19, alex/19, charles/19, stephen/19, Chris/18, bill/18, tim/17, arthur/17, eric/16, james/16, sir/16, le/16, jimmy/16, alexander/15, Christian/15, ;/15, George/14, Charles/14, harry/14, Thomas/14, te/14, weather/14, adam/14, Jeff/13, rick/13, Scott/13, dave/13, ron/13, richard/13, Pierre/13, martin/13, bear/13, sarah/12, matt/12, Jonathan/12, george/12, edward/12, Santa/12, drum/12, aka/12, near/12, lord/12, James/11, steven/11, joseph/11, uss/11, nick/11, prince/11, billy/11, na/11, charlie/11, gary/11, kuala/11, Puerto/11, walang/10, mga/10, Jan/10, Vladimir/10, maurice/10, du/10, matthew/10, leon/10, dennis/10, max/10, roger/10, ren/10, marie/10, don/10, tommy/9, 

**tag 9:**

	se/1, 

**tag 10:**

	,/111183, of/71857, and/48530, in/39847, -/17940, (/15237, for/13384, '/12082, to/10382, with/10069, on/9636, be/8366, from/7593, at/7146, as/6701, by/5543, or/4618, )/3748, ;/2973, that/2757, "/2708, when/2251, after/2104, during/2048, between/1631, where/1594, //1503, into/1403, include/1150, under/1144, until/1018, against/1011, over/952, than/933, before/906, since/815, through/790, about/768, if/636, while/633, within/568, because/562, like/536, but/533, around/470, follow/418, near/416, call/401, use/397, ##/393, among/391, de/369, throughout/328, along/250, feature/246, million/235, across/235, contain/233, upon/227, alone/221, although/220, despite/216, win/203, which/202, without/201, towards/191, versus/187, female/186, via/184, behind/163, above/155, have/154, outside/148, see/146, represent/146, hold/143, ?/141, per/139, provide/134, show/133, name/127, off/122, whose/121, involve/121, receive/117, allow/115, john/104, ii/104, cause/102, %/93, unlike/89, cover/86, year/83, become/79, enter/76, produce/75, reach/75, onto/75, offer/73, inside/73, require/71, star/68, below/64, defeat/63, route/63, beyond/63, surround/63, del/62, amongst/61, lose/61, toward/60, except/59, whether/57, act/56, entitle/56, von/55, unless/55, regard/54, concern/54, who/52, david/52, channel/50, bear/49, gain/49, champion/48, comprise/48, old/47, van/47, company/46, richard/46, meet/46, series/45, nor/45, alongside/45, mark/44, besides/44, o/43, host/43, whereas/42, score/41, band/41, di/40, face/39, base/38, operate/38, aboard/38, general/36, why/36, attack/36, take/36, cross/36, kong/36, wear/36, control/36, day/35, des/35, guard/35, till/35, indicate/35, join/35, S/34, cup/34, give/34, da/34, ibn/34, house/34, introduce/34, ask/33, whom/33, replace/33, affect/33, span/33, james/32, declare/32, visit/32, george/32, make/31, bin/31, hms/31, resemble/31, due/30, captain/30, Kong/30, peter/29, rider/29, issue/29, play/28, state/28, et/28, du/28, der/28, once/28, ago/28, enforcement/28, joseph/28, connect/28, opera/28, approve/28, acquire/28, della/27, lie/27, 

**tag 11:**

	nakhkhunte/8, 

**tag 12:**

	edited/1, 

**tag 13:**

	)/4122, "/3921, time/3213, year/3105, state/2320, game/2175, school/2130, city/2091, area/1799, name/1764, member/1590, team/1542, number/1502, ##/1493, system/1425, ##s/1408, university/1399, group/1371, war/1366, @card@/1272, town/1260, work/1248, line/1229, day/1228, man/1222, series/1216, part/1198, company/1175, world/1148, family/1131, country/1117, book/1113, population/1077, service/1076, end/1065, album/1038, house/1028, song/1018, people/1015, party/1009, band/1004, force/990, point/985, government/984, player/979, life/968, way/946, century/929, film/918, character/900, station/895, form/876, island/829, season/828, club/823, church/810, power/797, district/789, river/781, result/777, version/765, base/755, law/744, use/743, one/741, side/736, program/724, income/719, show/716, age/707, place/705, term/692, college/691, case/685, court/681, event/671, father/670, building/668, language/666, title/665, history/648, death/646, village/626, army/625, region/618, record/617, field/608, son/603, community/602, body/601, child/598, role/594, period/593, order/584, level/582, site/579, unit/578, king/572, set/570, center/565, race/558, word/558, ship/557, election/551, development/542, office/541, student/538, right/531, episode/530, division/525, award/517, park/514, league/514, battle/513, story/510, model/509, act/505, council/504, effect/502, star/499, position/497, woman/497, career/497, person/482, type/479, attack/476, leader/475, road/474, department/470, other/467, hand/465, home/464, problem/464, project/458, land/458, music/456, release/454, u/454, head/453, friend/449, organization/448, society/447, process/445, design/443, style/440, range/439, action/438, study/435, example/432, brother/428, class/425, census/425, union/423, list/422, month/419, theory/419, nation/419, movie/418, minister/412, association/412, mother/403, championship/403, car/398, success/397, fact/394, president/394, ground/393, household/388, art/385, track/385, movement/385, degree/382, production/381, province/381, product/379, rule/377, card/376, board/376, network/375, performance/374, wife/374, street/374, location/373, interest/372, boy/372, novel/365, structure/363, appearance/362, centre/362, committee/362, ability/360, source/358, issue/358, article/357, bridge/355, activity/353, change/351, method/351, county/347, magazine/347, water/343, 

**tag 14:**

	nadhiyaan/1, 

**tag 15:**

	##/14400, S/498, them/481, one/436, such/427, and/407, well/343, france/293, california/278, japan/253, st/249, australia/247, england/246, india/245, him/231, age/231, i/229, old/227, individual/222, europe/220, e/194, female/194, all/194, over/193, london/188, however/184, china/183, canada/183, along/181, germany/177, life/167, it/163, etc/160, d/154, u/153, Canada/149, italy/146, paris/142, especially/141, locate/140, Israel/139, science/138, america/129, England/127, more/127, texas/126, or/126, education/124, other/119, ohio/117, russia/116, most/113, parliament/111, some/111, b/110, pennsylvania/109, each/107, spain/106, god/105, law/103, scotland/101, c/100, p/98, music/98, earth/98, many/98, mathematics/97, general/96, director/96, massachusetts/95, illinois/95, ireland/95, London/94, art/94, Dr/93, water/93, both/90, history/89, washington/89, technology/89, rome/89, release/89, fame/88, up/88, a/87, Iraq/86, particularly/86, egypt/85, follow/85, inc/84, Finland/83, Georgia/83, English/83, chicago/83, representative/83, Co/82, Virginia/82, those/82, iran/80, Ontario/79, power/79, s/79, found/77, then/77, Mr/76, britain/76, wale/75, base/75, play/73, bear/73, mexico/72, Oxford/71, berlin/71, Sweden/69, Usa/68, Maryland/68, her/68, publish/68, Switzerland/67, michigan/67, physic/67, build/67, common/66, death/65, wisconsin/64, serve/64, boston/63, respectively/62, singapore/62, j/61, austria/61, that/61, two/61, Florida/59, greece/59, return/58, norway/57, philosophy/57, light/57, h/56, graduate/56, r/55, search/55, daughter/55, Washington/54, self/54, California/53, vol/53, non/52, business/52, Asia/52, florida/52, montreal/51, justice/51, woman/51, white/51, except/51, agriculture/50, g/50, French/50, turkey/49, Vietnam/49, f/49, philadelphia/49, complete/49, brazil/48, religion/48, communication/48, another/48, jerusalem/48, child/48, part/48, t/47, himself/47, space/47, write/47, die/47, athens/46, medicine/46, k/46, africa/46, length/46, much/46, founder/45, o/44, tennessee/44, Serbia/44, poland/44, romania/44, re/44, food/44, land/44, duke/44, thousand/44, king/44, independence/43, freedom/43, design/43, sign/43, high/43, march/43, 

**tag 16:**

	%/1147, )/443, people/216, year/200, housing/162, graduate/57, die/54, km/29, compete/29, million/27, damme/22, mile/21, l/19, percent/18, thereafter/17, Rico/17, minute/16, "/15, point/15, consist/15, township/14, many/14, participate/14, goal/13, m/13, airway/13, woxing/12, most/12, mph/11, billion/11, Kong/11, ii/11, today/11, Co/10, copy/10, collins/10, langit/10, month/10, agerskov/10, vegas/10, fm/10, unite/10, lakan/10, gate/9, 

**tag 17:**

	./95423, )/137, "/45, ?/18, as/12, infantry/5, 

**tag 18:**

	Kala/1, 

**tag 19:**

	the/151920, a/46206, his/11018, an/8733, S/6124, s/6035, their/4925, this/4427, its/4180, in/3187, her/2752, "/2430, ##/2272, other/1880, some/1835, many/1799, all/1683, two/1590, these/1566, one/1419, and/1382, that/1329, no/1308, for/1303, by/1271, $/1242, several/1229, any/1171, with/1093, each/958, as/919, new/868, another/801, three/778, both/644, of/580, more/578, high/497, various/497, every/480, most/456, four/455, on/454, or/404, to/380, include/372, such/350, my/347, world/344, five/328, those/325, our/318, different/313, your/306, at/305, into/298, great/295, which/290, from/280, good/278, provide/261, British/256, numerous/252, public/252, certain/239, local/238, six/234, human/234, modern/231, very/228, only/219, old/216, international/216, whose/216, under/214, early/212, American/204, over/203, like/202, small/201, base/200, former/198, major/197, large/196, team/183, national/179, little/179, long/179, see/178, average/174, make/173, @card@/173, much/172, special/169, multiple/169, -/155, seven/151, either/146, recent/139, ten/130, use/129, Arab/127, natural/125, political/120, united/117, produce/117, military/115, foreign/115, heavy/115, general/114, low/114, further/114, full/114, play/113, free/113, up/112, increase/109, eight/109, social/108, traditional/108, strong/106, black/103, private/103, late/101, Canadian/99, through/99, water/98, appoint/98, take/97, real/97, least/96, between/95, similar/94, white/93, german/91, nine/90, Japanese/89, ancient/89, gain/89, even/88, create/87, chief/86, among/86, future/85, individual/85, additional/85, computer/83, Israeli/83, after/83, common/82, government/81, study/81, prime/81, call/81, but/80, open/79, west/79, red/79, less/79, Christian/79, sri/79, lord/78, classical/78, while/78, super/77, cobra/76, western/75, approximately/75, legal/74, off/74, dark/73, know/73, considerable/72, air/71, personal/71, direct/71, business/70, popular/70, physical/70, extensive/70, north/69, reduce/69, separate/69, religious/67, specific/67, subsequent/67, about/67, Australian/66, Jewish/66, federal/66, official/65, almost/65, regular/64, poor/64, (/64, allow/63, English/63, when/63, receive/62, develop/62, how/62, 

**tag 20:**

	##/26019, )/6394, "/1148, new/703, bear/700, September/651, march/573, December/547, south/546, may/544, October/540, august/524, january/507, November/499, north/496, year/493, April/488, county/456, john/443, july/430, february/422, June/366, %/334, san/327, km/313, york/297, age/294, addition/282, west/276, ##th/250, st/244, S/238, s/229, about/228, al/222, (/222, every/217, king/216, america/215, live/212, east/211, northern/210, William/209, m/207, mile/205, season/205, example/198, james/196, late/196, include/192, university/186, charles/186, george/185, henry/183, general/182, sir/180, los/175, household/171, early/167, foot/161, angeles/158, a/156, family/156, thomas/152, africa/151, president/150, saint/149, david/143, college/142, reside/142, i/135, least/133, yard/132, carolina/132, francisco/130, la/129, ii/129, street/127, long/124, lake/124, goal/123, peter/122, de/122, die/119, metre/119, over/116, see/116, central/111, bc/110, ireland/109, robert/108, lord/106, june/104, smith/103, london/101, day/101, mary/101, australia/100, old/100, both/100, prince/100, europe/99, v/97, captain/96, western/94, iii/92, ?/91, c/90, wale/89, edward/89, fact/88, bill/88, louis/88, road/88, island/88, paul/86, richard/86, fort/86, ##st/84, student/83, January/80, around/80, square/80, mount/80, hour/79, jersey/79, then/78, port/78, mm./78, minute/77, martin/76, southern/76, hong/76, baron/76, Kong/76, kong/76, b/75, ft/75, bay/73, order/72, route/72, hill/72, diego/71, india/70, meter/70, point/69, mexico/68, jones/68, window/67, e/66, p/66, just/66, joseph/66, become/66, professor/65, brown/65, Michael/64, people/64, pope/63, lady/63, frank/63, Robinson/63, avenue/63, episode/62, steve/62, d/61, alexander/61, ad/60, n/59, x/59, admiral/59, game/59, Asia/59, f/58, colonel/58, white/58, now/58, approximately/58, earl/58, w/58, cm/58, win/58, state/57, page/57, eastern/57, jackson/57, h/56, man/56, isbn/56, round/56, emperor/56, only/56, don/56, sq/56, germany/55, et/55, r/54, queen/54, class/54, green/54, southeast/54, 

**tag 21:**

	one/1519, out/894, him/814, up/760, part/720, it/583, know/575, them/550, use/482, her/363, work/308, place/289, some/275, ##/275, base/266, well/265, play/256, bear/247, locate/238, child/225, live/224, appear/214, all/202, release/201, much/199, most/194, re/190, find/183, many/182, living/177, available/176, down/173, away/167, back/166, involve/162, himself/160, again/159, exist/154, today/133, die/132, because/132, this/131, off/131, president/129, kill/128, home/123, build/122, control/121, popular/115, result/115, responsible/114, together/112, themselves/110, so/109, more/106, there/105, present/104, common/103, occur/100, )/100, here/99, participate/95, consist/95, compete/93, look/89, write/89, perform/88, fight/86, charge/84, say/83, see/83, successful/82, name/80, those/79, change/78, non/78, destroy/76, member/76, associate/75, active/71, derive/71, record/71, over/71, famous/70, small/68, note/68, me/67, any/67, bury/63, happen/63, true/62, mention/62, important/61, engage/60, fast/60, high/60, publish/59, stay/58, capable/58, remove/58, earth/58, two/58, nominate/58, short/57, different/57, free/57, situate/57, possible/56, develop/56, england/56, defeat/56, good/56, think/55, hold/55, serve/55, stand/55, interested/54, unknown/54, educate/54, you/54, sell/53, far/53, north/53, study/53, credit/51, dead/51, long/51, replace/50, large/50, clear/49, deal/49, establish/48, settle/48, something/48, disappear/47, advantage/47, command/47, close/47, aware/46, arrest/46, open/46, alive/45, confuse/45, notable/45, contact/45, induct/45, chairman/45, accuse/44, Co/44, support/44, operate/44, display/44, divide/44, speak/43, apart/43, compose/43, qualify/43, create/43, hear/43, vote/43, money/43, death/43, instead/42, complete/42, herself/42, low/42, life/42, star/42, list/42, capture/41, his/41, escape/41, care/41, second/41, power/41, fire/41, itself/41, apply/40, useful/40, visible/40, account/40, along/40, light/40, lose/40, land/40, convict/39, else/39, twice/39, difficult/39, outside/39, grow/39, instrumental/38, self/38, read/38, accept/38, professor/38, withdraw/37, construct/37, each/37, fit/37, enough/37, 

**tag 22:**

	du/3, 

**tag 23:**

	etc/9, 

**tag 24:**

	milee/1, 

**tag 25:**

	./3127, liviakis/5, 

**tag 26:**

	al/8, 

**tag 27:**

	)/83, ./60, sawyer/11, anghel/10, prefecture/10, acid/10, miller/8, 

**tag 28:**

	be/66780, to/35133, have/16102, that/9111, as/8062, by/6647, in/5636, also/4955, not/4818, ,/4726, he/3952, and/3824, "/3499, it/3485, on/3409, make/3220, with/2997, use/2990, do/2944, which/2761, can/2714, for/2627, become/2623, who/2437, from/2356, take/2290, would/2263, they/1931, at/1891, only/1751, will/1734, know/1595, but/1584, give/1454, up/1454, find/1436, go/1404, may/1373, play/1327, include/1276, him/1274, see/1255, begin/1250, well/1240, into/1223, then/1214, more/1213, come/1208, call/1207, there/1195, such/1184, could/1177, often/1149, out/1134, serve/1123, win/1108, leave/1090, later/1084, all/1014, lead/1009, get/951, write/945, than/931, of/931, create/925, she/919, still/911, now/905, say/894, after/893, so/885, hold/845, run/821, appear/817, work/810, consider/810, move/803, continue/797, return/783, release/783, over/777, them/776, start/763, about/759, form/723, or/717, show/704, name/691, allow/680, receive/669, n't/661, build/656, remain/644, join/639, back/624, even/613, never/611, refer/608, set/607, due/603, produce/602, both/600, through/597, develop/586, bring/579, i/575, help/574, provide/570, believe/566, turn/548, should/548, first/547, tell/541, must/530, just/520, very/520, describe/515, before/510, you/508, what/507, claim/503, locate/501, usually/499, require/499, down/499, change/497, lose/494, like/494, currently/493, kill/492, establish/490, mean/485, follow/480, try/468, cause/467, while/467, if/463, publish/459, able/454, this/452, send/448, support/439, decide/439, put/434, live/432, open/432, force/431, design/425, when/424, meet/418, again/417, perform/415, carry/415, once/414, keep/412, much/411, feature/410, her/407, grow/399, reach/398, place/394, break/387, himself/387, off/387, we/386, generally/385, die/385, record/381, almost/377, think/374, sell/373, represent/371, replace/369, want/369, enter/367, offer/367, seem/364, operate/362, originally/358, always/356, choose/356, how/355, need/353, eventually/352, pass/351, marry/347, complete/346, without/345, discover/343, fall/343, one/340, soon/340, found/336, close/335, around/335, no/334, though/333, add/332, elect/330, too/329, manage/329, 

**tag 29:**

	obsolete/2, 


興味深いことに品詞8には人名、品詞15には国名、品詞16には単位、品詞20には月を表す単語が集まってきています。

一部の品詞には単語がほとんど割り当てられていない現象が起きていますが、論文では品詞数を17にしているのに対し、今回は2倍近い品詞数$K=30$で実験をしているため状態数が過剰であった可能性があります。

もちろんバグの可能性もありますが、正しい動作が何なのか分からないのでプログラムを見直すべきかどうかが分かりません。

ヒートマップは以下のようになりました。

![image](/images/post/2017-01-28/wiki.png)

## 青空文庫

青空文庫のテキストデータを用いて10万文のデータセットを構築しました。

また品詞数$K=30$としました。

以下が結果です。

**tag 0:**

	<eos>/200000, <bos>/200000, 

**tag 1:**

	カズ/1, 

**tag 2:**

	て/107577, ながら/4214, ず/2563, たり/2259, ば/1548, たら/1532, なく/636, られ/523, つつ/413, って/344, ん/333, たく/285, に/281, ざる/256, り/199, た/194, ら/189, で/149, つ/85, せ/71, そう/66, だり/65, ない/63, てる/57, させ/56, ッ/54, 出さ/53, んで/48, 得/48, る/45, と/45, ぬ/42, しめ/40, さ/40, 合い/38, とも/37, たって/36, う/31, れ/29, さえ/29, 込み/29, ち/28, たる/27, ちゃ/24, え/24, がたく/23, っ/23, こみ/22, がち/22, 込ま/21, たけれ/20, だって/19, やすく/19, ざま/19, 合わ/18, だら/18, じ/18, しも/17, や/17, な/16, む/16, づ/15, とう/15, かつ/15, むべ/15, えて/15, たい/14, あわ/14, テ/14, うて/14, っと/14, 来り/13, ゃるか/13, ど/12, 合わさ/12, 次第/12, ざり/12, く/12, むると/11, おり/11, きれ/11, ざら/11, または/11, ども/11, げ/11, 共/10, こま/10, かね/10, ぎ/10, 做/9, 

**tag 3:**

	あろ/1438, しよ/534, 行こ/171, やろ/119, せよ/96, 見よ/95, なろ/94, みよ/67, れよ/53, なかろ/48, 出よ/45, よかろ/35, いよ/34, 来よ/31, 帰ろ/31, 言お/28, 出そ/27, 隠そ/26, 知ろ/25, 殺そ/25, 去ろ/24, しまお/24, もらお/23, 書こ/22, 取ろ/21, 得よ/19, 迎えよ/19, してやろ/16, あげよ/16, かけよ/16, 立てよ/16, はいろ/16, いお/16, 入れよ/15, 逃げよ/15, 置こ/15, 語ろ/15, 見せよ/15, 立と/14, 聞こ/14, かかろ/14, 話そ/14, できよ/14, 求めよ/13, ゆこ/12, 死の/12, つこ/12, 示そ/11, 受けよ/11, 開こ/11, 送ろ/11, 救お/11, 避けよ/10, 上ろ/10, 考えよ/10, 買お/10, 離れよ/10, 動かそ/10, 試みよ/10, 止めよ/10, 待と/9, 

**tag 4:**

	

**tag 5:**

	##/30593, その/16258, この/8052, お/3625, そして/3330, また/3083, しかし/2677, 『/1507, 大/1496, あの/1455, 御/1330, ただ/1320, もう/1316, ある/1243, ｜/1209, と/1181, という/1042, それから/1024, そういう/1022, まだ/891, そんな/880, こんな/876, 第/815, 大きな/760, やがて/749, 小/695, そこで/692, 諸/684, もし/635, 年/634, あるいは/602, 同じ/581, こういう/570, 高/566, しかも/545, 若い/536, 小さな/527, 日/519, なお/514, 又/508, すぐ/489, ない/488, 右/471, 長い/466, ちょうど/426, 新/425, すでに/419, あらゆる/416, けれども/411, それでも/411, すると/408, すなわち/405, 実に/403, 今/399, 張/398, やはり/393, 古い/383, 最も/381, 玄/381, 白い/371, 新しい/366, どんな/362, ところが/358, わが/347, 深い/345, まず/344, 美しい/343, 呂/339, キャラコ/334, 木曾/333, 半/327, 両/322, 城/320, さて/318, それで/317, だが/314, 時/307, かの/302, 吉/297, 決して/297, もはや/297, いい/296, いつも/294, さらに/283, むしろ/283, 道/281, ほとんど/279, すべて/275, だから/274, 高い/274, 吾/272, …/268, よく/268, 劉/267, よい/257, 日本/253, そうして/249, それに/248, ご/247, 金/246, 小さい/242, 寿/241, （/240, 源/240, 暗い/240, ふと/236, 関/233, つまり/232, 当時/231, まったく/230, もっとも/230, いよいよ/227, 董/227, あたかも/227, いかにも/226, おそらく/225, 老/224, かつて/224, まるで/224, 藤/222, 同時に/221, ついに/220, 全/220, いかなる/218, もっと/216, 全く/214, とにかく/212, いかに/209, どの/209, けれど/208, または/206, 明治/202, いわゆる/200, みな/199, そう/193, 黒い/191, 宿/191, かかる/190, あまり/190, 広い/190, 無/188, 孫/187, 清/186, いや/185, 不/182, 各/182, とうとう/180, ごとき/178, 旧/177, どうして/177, 古/176, 少し/176, 南/175, うし/174, ことに/174, 人/174, 遠い/172, こうして/170, ようやく/168, もちろん/167, 悪い/166, でも/165, 皆/165, 王/165, ちょっと/164, およそ/163, かえって/163, 再び/163, 実は/162, どうも/162, 実際/161, 強い/159, 黒/158, 恐ろしい/158, 平田/157, 陳/157, 夜/157, 朝/156, しばらく/155, ああ/155, いま/154, 特に/154, ごとく/154, 本/153, 外国/153, 別に/153, 』/152, 馬/152, すっかり/151, 赤い/150, しかるに/150, 宗/149, これから/149, ごく/149, 女/148, 妻/148, 徳川/147, ずっと/147, 伏見/147, なぜなら/145, 大きい/145, 

**tag 6:**

	おいで/52, お尋ね/4, 

**tag 7:**

	だろ/1417, でしょ/826, たろ/267, ましょ/147, られよ/13, させよ/12, 。/5, 

**tag 8:**

	

**tag 9:**

	太っ/2, 

**tag 10:**

	た/108308, ない/11027, ます/3748, ぬ/2736, ず/2027, ん/1949, てる/1878, だ/1795, たい/1390, う/1247, られる/973, なけれ/848, たる/692, ざる/641, れる/634, る/337, 得る/232, まい/230, ね/229, 候/228, うる/213, そう/159, がたい/148, たり/124, い/123, せる/115, させる/108, り/107, 難い/100, し/94, つ/92, こん/91, べし/89, やすい/89, 込ん/84, むる/75, なさい/61, です/60, ざれ/57, ゆる/55, たら/55, 出す/52, ゆ/51, える/48, たれ/45, 居る/45, にくい/45, かねる/44, たく/44, こむ/39, がち/38, す/38, しめる/37, 込む/36, でる/36, よう/35, ずる/35, よ/33, づ/33, しむ/31, 合う/31, き/31, なり/29, ねえ/29, 乍/29, べき/28, つる/28, 易い/27, まし/25, ていう/24, すぎる/24, うれ/24, 〉/23, ずん/23, はじめる/22, 得/22, だす/21, るる/21, 去る/20, つづける/20, なさる/18, 始める/18, 給え/18, ませ/17, まわす/17, 給う/16, 下さい/16, まする/16, 難き/16, 候え/16, きり/16, 置く/15, 申す/15, ッ/15, 奉る/14, 上る/13, ス/13, という/13, ゃる/12, ける/12, かた/12, せり/11, いも/11, しった/11, 過ぎる/11, あう/10, かえる/10, …/9, 

**tag 11:**

	に/119197, を/105482, で/55662, が/49848, は/36963, と/22656, も/16714, へ/11855, から/8961, まで/3509, として/1947, より/1664, でも/1055, か/978, について/927, や/860, かも/840, において/800, など/693, によって/640, にとって/521, さえ/456, に対して/407, さ/400, ばかり/378, なら/354, だけ/326, なく/312, ！/299, ｜/298, と共に/298, とも/279, こそ/269, ほど/268, らしく/259, とか/248, しか/238, し/234, にたいして/210, あり/208, とともに/204, すら/172, じゃ/166, の/162, だの/161, 以来/140, を以て/131, にかけて/131, にて/129, く/129, 々/121, ずつ/119, やら/111, ながら/107, つて/103, により/93, しく/83, ニ/81, 一つ/80, ぐらい/78, のみ/75, なんか/75, 以上/74, ヲ/70, 近く/70, ）/68, 故/67, 化/66, って/63, だって/63, を通して/62, っと/61, たち/61, ［/58, 深く/55, なり/54, り/53, ごろ/52, う/51, 同時に/49, 後/46, とても/45, 多く/41, 自身/41, 相/39, ご/38, よく/37, たら/37, に関して/37, に従って/37, ち/37, 早く/36, ゝ/36, よ/36, 迄/35, せら/35, め/35, 上/34, に対し/34, ゆえ/34, がら/33, にわたって/33, ひとつ/32, なぞ/32, 自ら/31, ？/31, 余り/30, 高く/29, らく/28, ら/28, につれて/28, における/28, 以下/28, っ/27, くし/26, だに/26, さん/26, 程/25, ども/25, 頃/25, 曰く/25, 故に/25, きり/24, にあたって/24, えと/24, みえ/24, 視/22, たり/22, じゅう/22, その他/22, かた/22, を通じて/22, に従い/22, につき/21, 中/21, これ/21, しも/21, 以後/21, いわく/21, また/21, びく/20, 半/20, ほか/20, につれ/20, をもって/19, づく/19, ぞ/19, 者/19, けど/19, 輩/18, 等/18, 蛇/18, 通り/18, さま/18, どころか/18, じ/17, つき/17, ぎり/17, わるく/16, だり/16, 位/16, づたいに/16, 来訪/16, 前/16, に際して/16, はね/16, 気/16, ごとき/16, 三つ/15, 御/15, どおり/15, 言う/15, くば/14, ひとり/14, 甚だ/14, なんぞ/14, あまり/14, はじめ/14, にあたり/14, なんて/14, ぎ/13, をめぐって/13, ッ/12, にかけ/12, ぶ/12, ト/12, 歩き/12, ざめて/12, 立ち/12, いと/12, 皆/12, もっとも/12, んで/12, いらい/12, いよいよ/11, ことごとく/11, っぽく/11, 入り/11, 立/11, じゃあ/11, ちかく/11, 

**tag 12:**

	し/29890, い/19231, れ/10688, あっ/9853, なっ/6753, 来/5837, だっ/4675, 見/3943, せ/3304, 出/2732, 行っ/2709, なら/2478, しまっ/2254, いっ/2145, 言っ/1976, あり/1878, 思っ/1733, なかっ/1651, き/1632, もっ/1495, でき/1492, 見え/1478, 考え/1436, 持っ/1273, でし/1235, かけ/1166, 出し/1126, 感じ/1098, 立っ/1059, 知ら/996, つけ/987, つい/936, 知っ/931, やっ/924, ござい/888, 得/876, なり/836, 帰っ/829, くれ/822, 聞い/813, 書い/804, み/799, 知れ/770, 出来/725, 云っ/700, かかっ/668, 置い/659, 入っ/647, 見せ/622, 入れ/605, 忘れ/582, 歩い/552, なく/524, あげ/523, 居/520, 立て/508, 似/492, おい/467, あら/454, 待っ/449, 受け/441, わから/440, 与え/436, 残っ/436, 離れ/432, 取っ/431, 落ち/407, 思い/402, 向っ/400, はいっ/396, ながめ/385, 生き/384, 居り/379, 過ぎ/374, 話し/367, しれ/361, わかっ/350, すぎ/348, 思わ/340, 通っ/339, 言い/338, とっ/335, 覚え/332, べから/327, 送っ/326, 信じ/324, がっ/320, 認め/316, 答え/312, 開い/310, 上っ/309, あろ/308, 上げ/307, 違い/303, 笑っ/299, ッ/296, 始め/294, 降り/294, 眺め/293, なし/287, 向け/286, 申し/283, しまい/282, 切っ/279, え/278, 連れ/272, 求め/264, 続い/261, 戻っ/257, 黙っ/256, つづけ/254, 尋ね/253, なれ/252, つか/251, 去っ/251, なけれ/251, 現われ/250, 出かけ/250, 迎え/245, 着/244, 近づい/242, 買っ/239, 教え/239, 示し/237, 聞え/235, やって来/232, 述べ/232, 残し/229, 集まっ/228, うけ/228, 告げ/224, おり/224, 起っ/223, なつ/222, さし/222, 起こっ/222, きい/221, 乗っ/221, 寝/220, 向かっ/219, 驚い/219, 隠れ/219, 流れ/216, 思い出し/215, 逃げ/215, 失っ/212, 消え/212, 眠っ/210, あけ/206, 行き/205, いけ/203, 分ら/203, 伝え/203, しめ/202, け/200, やり/200, 用い/199, いわ/199, 変っ/198, 愛し/198, 聞こえ/196, 坐っ/194, 続け/194, 抱い/191, 見つけ/191, 作っ/189, 走っ/189, いい/189, 捨て/188, つれ/187, たて/187, め/187, 加え/186, 言わ/186, なくなっ/185, 思え/184, らしかっ/183, はじめ/178, 行か/177, 違っ/177, 着い/177, 殺し/176, つづい/175, 向い/175, いたし/175, 至っ/175, 生まれ/172, 寄っ/172, 上がっ/171, 集め/170, 泣い/167, あらわれ/167, 働い/166, 打っ/163, もらっ/163, 語っ/162, すわっ/162, 見つめ/162, 恐れ/161, 返っ/160, 生れ/157, 

**tag 13:**

	、/189356, は/5698, ##/4420, も/3285, その/2557, する/1250, 　/1098, お/1050, この/1014, （/504, 御/439, また/421, 大/370, から/360, なく/355, 云う/279, …/252, 同じ/246, 『/232, あの/227, 大きな/212, ば/205, まだ/197, ただ/166, 近い/162, 必ず/161, 皆/158, もう/145, いる/144, れる/139, こんな/136, 最も/135, 小/121, 見える/115, るる/115, すでに/114, ふ/105, 住む/101, 実に/100, どんな/100, ）/99, まず/87, 又/87, わが/85, 新しい/83, 白い/79, かかる/75, 第/74, いっそう/73, 深い/72, 諸/70, すぐ/69, “/68, 持つ/68, つて/67, ほとんど/66, 新/66, 同じく/65, すら/64, こそ/63, 全く/63, 共/63, たった/62, なお/62, 長い/61, 小さな/60, 立つ/60, むしろ/59, 自注/59, 両/58, そういう/57, すなわち/55, よく/55, 不/54, 一層/54, ずっと/53, 美しい/52, 大いに/50, 無/50, かなり/49, 常に/49, 再び/49, とも/49, まで/49, ども/48, 別に/47, 半/47, 駕/47, 黒/46, せる/46, 今/45, 決して/44, さらに/44, 約/44, 同/44, ようやく/43, 神/42, 極めて/41, 事務/40, 林/40, 日本/40, 古い/40, 少し/39, きっと/39, 呼ぶ/39, 早く/38, 強い/38, 大きい/38, うし/38, ちょうど/37, 蛇/37, 行く/36, 金/36, そんな/36, 同時に/35, ？/35, ある/35, 小さい/35, 水/35, 若い/35, 大変/34, やはり/34, 来る/34, 却って/34, それぞれ/34, 明治/34, 平田/34, 一番/33, ごく/33, 称する/33, こういう/33, 米/33, なき/32, 源/32, 暗い/32, 城/32, もっと/31, 初めて/31, 殆ど/31, すこぶる/31, およそ/31, 宿/31, 黒い/31, 各/30, 通う/30, 有する/30, 薄/30, 働く/30, 鉄/30, 求める/30, ことごとく/29, 直ちに/29, しかも/29, 激しい/29, 角/29, まっ/29, 平民/29, 遠い/29, 愛する/29, 入る/29, 実は/28, いつも/28, 〈/28, 朝/28, つづく/28, 東京/28, 江戸/28, 低い/28, 吾/28, あらゆる/28, 右/28, 高/28, かえって/27, 多く/27, よほど/27, はなはだ/27, そのまま/27, いちばん/27, 竜/27, 高い/27, 随分/26, 成る/26, 更に/26, たちまち/26, 等しく/26, いかにも/26, 主として/26, 益/26, 正/26, 流れる/26, 妻/26, 赤い/26, 外国/26, 木曾/26, キャラコ/26, 寿/26, 張/26, はじめて/25, ！/25, なかなか/25, 絶えず/25, 

**tag 14:**

	

**tag 15:**

	なかっ/5868, まし/5217, られ/3585, ませ/1264, させ/221, 得/200, 出し/199, だし/185, ましょ/180, っ/173, 始め/152, たかっ/148, はじめ/113, きっ/97, たろ/86, 合っ/83, しめ/77, で/72, らし/67, かね/66, きれ/64, れ/58, すっ/54, なく/46, ら/45, こん/42, まわし/41, つづけ/39, かかっ/33, やし/32, 初め/31, たく/29, 込ん/27, 去っ/26, あたわ/26, 直し/24, な/22, かけ/21, られよ/20, ます/19, え/19, ぬい/19, ま/18, すぎ/16, きら/16, まもっ/15, 能わ/15, 寄っ/15, 上っ/15, 申し/14, っし/13, て/13, ざら/12, きたっ/12, 続け/12, 奉り/11, がたかっ/11, 来っ/11, 果て/11, なかろ/10, し/10, ざり/9, 

**tag 16:**

	は/17725, も/13937, さ/3064, なく/2131, と/2068, よく/1470, あり/1433, また/1324, そう/1156, 思わ/950, なり/814, お/694, まだ/618, こう/612, 少し/586, もう/528, どう/512, し/474, でも/453, さえ/438, 言わ/403, いい/395, 言い/377, まで/370, すっかり/358, 見/352, しか/319, せら/311, なお/311, 多く/292, じっと/284, 一つ/282, すぐ/280, 想像/279, 深く/274, はっきり/266, 相/263, 皆/262, 結婚/256, 思い/251, のみ/246, いわ/245, 早く/243, おり/238, 今/237, 立ち/233, あまり/230, なんと/229, 引き/228, 大きく/226, 決して/221, 駈け/214, ただ/213, 許さ/210, 高く/206, みな/202, しばらく/201, 行き/200, ほとんど/197, 打ち/197, 取り/196, ちょっと/194, 遠く/193, 行わ/192, ひどく/192, すこし/190, 実に/188, ばかり/187, 安心/184, 発見/183, いつも/178, 長く/171, 全く/169, 書き/166, 逃げ/165, 大いに/164, ずっと/164, 呼び/164, すでに/163, 注意/163, 知ら/162, 心配/160, みずから/160, 持ち/158, 云い/156, やや/155, いかに/153, かえって/152, 押し/152, 怖/152, 通り/152, 云わ/152, なかなか/151, 聞き/150, 呼ば/150, 理解/148, 又/148, つき/147, 大変/146, か/146, びっくり/144, 書か/143, 満足/143, ごとく/142, まったく/140, 打た/140, 吹き/139, 現/139, 聞か/137, さして/135, いろいろ/134, 初めて/134, すら/133, 再び/133, もっと/133, 飛び/131, 最も/131, 説明/131, 自ら/131, 振り/131, 殺さ/129, 悪く/128, さし/128, 読み/127, なし/125, うち/124, 強く/122, しばしば/121, 考え/120, 取ら/119, ふと/119, とり/118, 斬り/117, 馳/116, そっと/116, ああ/115, さらに/115, どうして/114, 行なわ/114, 出発/114, ますます/113, ##/113, なさ/112, かなり/112, そのまま/111, どうか/111, どんなに/110, 一寸/110, ぼんやり/110, はじめ/110, 案内/109, 出さ/109, 歩き/109, ゆき/109, いよいよ/108, 泣き/108, ふり/108, 走り/107, 共に/107, 知り/107, 白く/107, 跳び/106, 決心/105, かつて/105, どうしても/104, 持た/104, つけ/104, 追い/103, 引/103, 同時に/102, やはり/102, だんだん/101, 支配/100, 語り/100, 笑い/100, 入り/100, 軽く/100, 存在/99, 近く/99, しろ/99, やり/98, 置き/98, 申/97, 次第に/96, 利用/96, あまりに/96, せよ/96, 本当に/95, もち/94, 同じく/94, 仰/94, 用意/93, 置か/93, ようやく/93, 振/93, ッ/93, 突き/91, ついに/91, 見え/91, 一番/88, かけ/88, 

**tag 17:**

	ある/17113, いる/14779, ない/7210, する/7190, なる/2658, いう/2163, 見る/2107, 来る/1743, 行く/1514, れる/1511, 思う/1509, ゆく/870, 居る/839, くる/798, いい/770, す/676, せる/673, みる/617, 言う/600, しまう/576, な/567, いえ/552, 何/540, 見える/526, よい/506, 出る/491, すれ/487, やる/444, できる/384, あれ/371, くれる/370, 聞く/366, るる/365, 同じ/349, 云う/344, 多い/313, 考える/307, 知る/307, どう/297, 見れ/288, 出す/270, 出来る/249, つく/243, くれ/240, なれ/237, 帰る/221, 呼ん/217, ね/215, なす/193, 読ん/189, あり/188, なけれ/188, わかる/185, おく/184, よる/184, 至る/181, 入る/177, …/175, 書く/171, 立つ/170, かかる/169, 通る/164, いらっしゃる/159, 面白い/157, 得る/153, 与える/152, 置く/150, る/145, 悪い/143, 見せる/141, 下さい/138, 死ぬ/137, かける/136, 感ずる/131, 取る/128, 感じる/125, 言え/122, 示す/122, 呼ぶ/121, 待つ/119, 申す/119, 食う/115, 踏ん/115, 歩く/114, 去る/111, 語る/111, なり/110, きく/109, 近い/107, 無い/103, 入れる/103, よ/100, 読む/99, つける/99, よん/98, とる/97, 沈ん/97, や/96, 死ん/95, 運ん/93, どこ/91, 当然/91, うれしい/90, 持つ/90, 分る/90, 飲ん/90, 上る/88, そう/88, よれ/86, みれ/85, 違う/85, こん/85, 近づく/83, 驚く/83, 及ん/82, 終る/81, 迎える/80, あ/80, もらう/79, 愛する/79, 下さる/78, 遊ん/78, 走る/76, ほしい/75, あげる/75, 有する/75, 私/75, つかん/74, 立てる/73, 受ける/73, 選ん/73, 込ん/73, 殺す/72, 会う/72, いく/71, 笑う/71, 早い/71, 及ぶ/71, 動く/71, 問う/71, 上げる/70, たまらない/69, 浮かん/69, 行け/68, 頼ん/68, 望ん/68, かえる/67, 話す/66, よろしい/66, 包ん/66, 過ぎる/65, だ/65, 出かける/64, 欲する/64, く/64, 候/63, 住む/62, 思え/62, 起る/62, いえる/62, 明/62, そんな/62, 苦しい/60, やって来る/60, 泣く/59, 望む/59, 答える/59, 大きい/59, 信ずる/59, 眠る/58, いれ/58, 忘れる/58, 開く/58, 悲しい/58, 困る/57, 少ない/57, たつ/57, せよ/57, 生きる/56, 見つける/56, 思い出す/56, みえる/56, どれ/55, いや/55, 着く/54, 動かす/54, 聞える/54, 飲む/53, わるい/53, ッ/53, のん/53, 飛ぶ/52, 深い/52, 含ん/52, から/51, 送る/51, 打つ/51, 切る/51, 失う/51, 求める/51, かく/50, 

**tag 18:**

	

**tag 19:**

	団/6, 

**tag 20:**

	の/190446, は/37265, な/21817, が/16217, と/8566, という/6393, から/5590, も/5399, や/3964, ｜/2597, ##/1996, か/1599, ・/1454, なる/1298, （/944, として/815, たる/747, らしい/709, 之/699, でも/607, ノ/582, 的/554, より/522, ほど/385, る/381, に対する/337, だ/315, によって/291, 』/282, とも/282, べき/280, い/256, 、/229, する/228, 州/207, において/205, における/186, に対して/175, および/163, に関する/161, なき/155, ッ/146, にたいする/144, ある/143, しい/138, ）/130, こそ/130, と共に/129, だの/127, とともに/127, く/124, ながら/123, 々/121, ずつ/117, ばかり/114, なら/114, し/112, による/109, ぐる/105, とか/105, といった/94, ヶ/92, 羅/92, やら/91, を以て/88, 学/85, まで/85, のみ/82, にあたる/79, 志/71, 子/70, 以来/70, っ/68, 張り/67, ば/62, にて/61, 居/60, 郷/60, 経/60, ら/60, 史/59, または/59, とかいう/59, よく/57, じ/56, しく/54, ちの/50, 記/49, 乃/49, ない/48, 中/47, でる/46, だって/46, 生/45, 集/45, とても/45, 女/44, …/44, 深い/43, 公使/43, 巻/42, 光/42, 庵/40, 第/40, たち/40, っぽい/39, き/39, なり/38, につき/37, 謂/36, 没/36, に/36, ハ/36, もしくは/35, 馬/35, ヲ/35, 諸/34, すなわち/34, ぶ/34, 主義/33, 掘り/33, 吉/33, 甲/32, 侯/32, 籠/32, ち/32, っと/32, 高/31, にわたる/31, 及び/31, じゅう/31, ん/30, 修道/29, しも/29, を通して/29, 前/29, 照らす/27, 院/27, 師/27, ならびに/27, ごろ/27, きわまる/26, ぼ/26, 殿/26, 函/25, ガ/25, らし/25, 論/25, 法/25, 家/25, 程/25, 等/25, 伝/24, 部/24, 頃/24, 迄/24, 秀/23, エル/23, よ/23, 談/22, 県/22, はじめ/22, 典/21, つ/21, 島/21, 即ち/21, 〔/21, 掛/20, 臭い/20, 兼/20, 金/20, って/20, 録/19, やすい/19, 目/19, 鑑/19, 教/19, 役人/19, 守/19, 市/19, 恋/18, 義/18, ちゃ/18, 式/18, 提督/18, 物語/18, 達/18, を通じて/18, 六月/18, 以上/18, に従って/18, くさい/17, がたい/17, たらし/17, 華/17, 注/17, 武/17, 約/17, つた/17, ど/17, せる/16, 

**tag 21:**

	嘘/4, 

**tag 22:**

	命/506, 神/375, 王/291, 姫/192, 彦/137, 國/129, 天皇/122, 御子/114, 宮/108, 君/87, 次に/87, 祖先/86, 等/83, 臣/82, 天/81, 女/76, ）/68, 郎女/65, 大和/63, 子/61, 囲炉裏/60, またの名/47, 弟/43, 松平/40, その他/37, 様/37, 大神/36, 国/36, 別/33, 主/33, 妹/31, 樣/31, 造/30, イザ/30, 皇后/29, 兄/28, 大塔/28, 一方/27, 天下/26, 連/25, 大/25, 守/25, 近江/24, 尊/24, 宿禰/23, ナギ/23, 原/23, 名/22, □/22, 問屋/21, シ/21, および/20, 伊勢/20, 木/19, 部/19, 若/19, 武/18, 大國/18, 院/18, 御陵/17, 出雲/17, 山城/17, 山/17, スサノヲ/17, ヲ/17, 良/17, など/16, 葛城/15, サホ/15, ごとき/15, 葦原/15, チョイ/14, 炉/14, 尾張/14, オホハツセ/14, 穗/14, 石/14, アドルフ/14, 鹿/13, 紀州/13, 春日/13, 丹波/13, 河内/13, ギリシア/13, 橘/13, 越/13, 信濃/13, 本陣/12, 伊賀/12, 直/12, 筑紫/12, 廃止/12, 吉備/12, 御名/12, 播磨/12, 花/12, 柳生/12, 庄/12, 島/12, 皇/12, 呉/12, 庄屋/11, 正/11, 高木/11, マルタン/11, 中心/11, 類/11, 松丸/11, 。/11, 伯耆/10, 檜/10, 城/10, ヴェルガ/10, 太子/10, タケシウチ/10, ヤマトタケル/10, ヤマト/10, 師/10, 小/10, 玉/10, つて/10, 祖/10, 城主/10, 美濃/10, 入/10, オホサザキ/9, 

**tag 23:**

	曹洪/2, 

**tag 24:**

	神/6, 

**tag 25:**

	。/48, ラバック/1, 

**tag 26:**

	。/99920, う/8139, なけれ/3, 

**tag 27:**

	人/12032, 彼/10008, よう/9224, それ/8508, こと/8125, 私/7380, もの/7212, 自分/5760, 中/5002, これ/4188, 方/3937, 日/3515, 家/3316, 上/3154, 時/3112, ##/2909, 彼女/2840, 者/2714, 心/2701, 前/2670, そこ/2627, 手/2614, 何/2406, 顔/2404, 葉子/2337, うち/2236, ところ/2150, 眼/2147, ため/2110, 女/1997, 年/1950, 目/1942, 男/1938, 声/1927, 間/1848, さん/1830, 半蔵/1812, さ/1759, 今/1739, 氏/1736, 気/1712, ここ/1622, 的/1621, 身/1604, クリストフ/1597, 事/1587, 下/1459, 一つ/1441, 名/1421, 話/1358, 口/1336, 言葉/1318, 頭/1314, 姿/1284, 人間/1268, 馬/1246, 後/1221, 子供/1191, 道/1145, 彼ら/1120, 武蔵/1104, 町/1102, 夜/1089, 所/1087, 物/1086, 生活/1041, 父/1027, どこ/971, 力/968, 村/965, 人々/961, 音/955, 』/949, 胸/939, 先/929, 手紙/917, 山/900, 水/896, 他/895, 風/892, 屋/866, 母/854, 子/851, そう/846, 本/828, わたし/824, あと/818, 足/814, 門/812, 耳/809, 娘/809, あなた/804, 次/777, 地/771, ほう/767, 国/751, 部屋/741, 外/735, 今日/734, 時代/730, 仕事/711, 日本/703, 竜/701, 度/696, 倉地/680, 助/674, そば/666, 花/666, 先生/666, 色/660, それら/658, ほか/656, 兵/650, 神/641, 船/629, すべて/619, 例/617, 金/614, 室/609, 多く/609, 非常/592, 形/589, 様子/587, 面/583, 体/580, 敵/573, 光/569, 蛇/565, 石/564, 世界/561, 火/551, 頃/550, 歳/548, 誰/545, 朝/541, 君/540, 空/533, 夫人/531, 内/530, 曹操/528, だれ/524, 酒/513, 影/512, 老人/509, 時間/509, 意味/509, 妻/503, 不思議/498, あたり/497, 二つ/488, なか/486, 壁/486, 地方/485, 涙/485, 窓/484, 場合/484, 雨/483, 夢/482, 場/482, 考え/479, まま/475, 街道/475, 通/474, 血/473, いろいろ/471, 少し/470, 民/465, 々/463, 軍/463, 静か/461, 首/460, 草/459, 店/454, 急/452, 様/449, 自然/448, 木/447, 相手/446, 僕/443, 庭/441, 徳/438, 点/436, 死/435, 客/434, 藩/432, 旅/430, 階/430, とき/429, 感情/429, 晩/429, など/428, 奥/426, 社会/423, 腕/420, いずれ/418, 布/417, 帝/416, ころ/415, なん/413, 昔/408, 海/408, 

**tag 28:**

	愛撫/1, 

**tag 29:**

	と/25185, だ/9699, こと/9194, か/8540, ば/4728, よう/4658, もの/4195, から/4052, ので/3784, です/3681, たち/2877, だけ/2246, など/2020, ほど/2013, 時/1675, まで/1661, ら/1609, ところ/1550, し/1475, ばかり/1448, 者/1401, べき/1171, 事/1130, ため/1120, さ/1105, そう/1011, ）/944, という/942, らしい/901, で/825, ね/825, 的/761, なり/740, として/692, ん/686, のに/683, ども/674, なぞ/598, まま/576, なら/552, とき/541, のみ/521, ？/519, けれども/508, わけ/507, よ/489, へ/489, まい/479, より/471, 自身/437, ！/428, や/393, うち/382, くらい/379, はず/372, …/352, つて/347, 以上/346, 』/338, ころ/300, 間/284, なし/283, らしく/282, ごと/280, ぐらい/273, 々/269, つもり/266, とか/263, 所/252, 物/246, 様/242, あり/238, 達/229, とも/226, みたい/219, 等/214, ぞ/212, 頃/210, ものの/209, 以外/207, 位/204, 後/203, ぶり/203, 様子/198, 由/193, 風/193, 気/187, ごろ/183, べし/181, 前/181, 人/180, ”/176, さま/167, けれど/167, 必要/165, つき/164, 通り/164, 力/164, がた/164, ふう/162, す/152, たる/148, たび/143, について/139, って/139, 方/138, かしら/135, きり/128, ど/128, 限り/128, とおり/127, ごとく/126, べく/124, 故/123, 処/122, 筈/121, じゅう/118, 場合/118, そのもの/117, 性/115, 程/108, やら/107, め/107, 座敷/105, 否/104, あたり/99, ほか/97, 日/95, 心持/94, 話/92, 衆/92, なる/92, 点/92, 次第/92, 共/90, も/90, な/85, 〉/80, 付/80, 夫婦/80, 内/80, ゆえ/79, 時代/78, 羅/77, なき/76, かぎり/75, 思い/75, 法/75, 機会/73, じゃ/72, 仲間/71, がら/70, 人々/70, 音/70, 訳/69, たり/67, において/67, 道/67, に/66, 声/64, 郷/64, 男/64, 以来/63, 上/63, につれて/63, れ/63, 全体/61, 側/61, 状態/61, 介/61, 如く/60, 越し/60, ぼ/60, ま/60, さん/59, 迄/59, ずつ/59, じ/58, 丈/58, 気持/57, げ/56, いろ/55, だらけ/55, わ/54, 感/54, もん/54, ながら/53, んで/53, け/53, 奴/52, 言葉/52, 顔/52, ある/51, ざし/51, 誉/51, 以下/50, かた/50, 流/50, 書/50, にとって/49, どおり/49, 


特徴的なのは品詞12と17に動詞が、品詞22と27に名刺が集まっていますが、英語の場合に比べるとあまり面白い結果が得られませんでした。

ヒートマップは以下のようになりました。

![image](/images/post/2017-01-28/aozora.png)

![image](/images/post/2017-01-28/aozora.major.png)

## 補正項について

式(25)に含まれる$I(\cdot)$ですが、何を意味しているかは次のとおりです。

- $I(t_{i-2} = t_{i-1} = t_i = t_{i+1})$のとき$n_{(t_{i-2}, t_{i-1}, t_i)} = n_{(t_{i-1}, t_{i}, t_{i+1})}$
- $I(t_{i-2} = t_{i-1} = t_i)$のとき$n_{(t_{i-2}, t_{i-1})} = n_{(t_{i-1}, t_{i})}$
- $I(t_{i-2} = t_i = t_{i+2} , t_{i-1} = t_{i+1})$のとき$n_{(t_{i-2}, t_{i-1}, t_i)} = n_{(t_{i}, t_{i+1}, t_{i+2})}$
- $I(t_{i-1} = t_i = t_{i+1} = t_{i+2})$のとき$n_{(t_{i-1}, t_{i}, t_{i+1})} = n_{(t_{i}, t_{i+1}, t_{i+2})}$
- $I(t_{i-2} = t_i , t_{i-1} = t_{i+1})$のとき$n_{(t_{i-2}, t_{i-1})} = n_{(t_{i}, t_{i+1})}$
- $I(t_{i-1} = t_i = t_{i+1})$のとき$n_{(t_{i-1}, t_{i})} = n_{(t_{i}, t_{i+1})}$

$t_i$をギブスサンプリングする際、まずnグラムカウントから$t_i$を削除しますが、その時影響を受けるカウントテーブルは$n_{(t_{i-2}, t_{i-1}, t_i)}, n_{(t_{i-1}, t_{i}, t_{i+1})}, n_{(t_{i}, t_{i+1}, t_{i+2})}$の3つです。

しかし上の条件が成り立つと、3つあるように見えるテーブルが実は全部（または2つが）同一のテーブルだったということになりますので、式(25)の計算に悪影響を及ぼします。

（たとえば$t_{i-2} = t_{i-1} = t_{i} = t_{i+1} = 2$だった場合、$n_{(t_{i-2}, t_{i-1}, t_i)} = n_{(t_{i-1}, t_{i}, t_{i+1})} = n_{(2, 2, 2)}$になります）

そもそも式(25)の書き方は誤解を生むため、正確に書くと以下のようになります。

$$
	\begin{align}
		P(t_i \mid \boldsymbol t_{-1}, \boldsymbol w, \alpha, \beta) \propto 
		\frac{n_{t_i, w_i} + \beta}{n_{t_i} + \mid W_t\mid\beta}\cdot
		\frac{n_{(t_{i-2}, t_{i-1}, t_i)}^{(i)} + \alpha}{n_{(t_{i-2}, t_{i-1})}^{(i)} + \mid T \mid\alpha}\cdot
		\frac{n_{(t_{i-1}, t_{i}, t_{i+1})}^{(i+1)} + I(t_{i-2} = t_{i-1} = t_i = t_{i+1}) + \alpha}{n_{(t_{i-1}, t_{i})}^{(i+1)} + I(t_{i-2} = t_{i-1} = t_i) + \mid T \mid\alpha}\cdot\nonumber\\
		\frac{n_{(t_{i}, t_{i+1}, t_{i+2})}^{(i+2)} + I(t_{i-2} = t_i = t_{i+2} , t_{i-1} = t_{i+1}) + I(t_{i-1} = t_i = t_{i+1} = t_{i+2}) + \alpha}{n_{(t_{i}, t_{i+1})}^{(i+2)} + I(t_{i-2} = t_i , t_{i-1} = t_{i+1}) + I(t_{i-1} = t_i = t_{i+1}) + T\alpha}
	\end{align}\
$$

この式は4つの項の掛け算になっていますが、それぞれの項に含まれるトライグラムカウントとバイグラムカウント$n_{(\cdot)}$は品詞の観測回数を考慮する必要があります。

まず、

$$
	\begin{align}
		\frac{n_{(t_{i-2}, t_{i-1}, t_i)}^{(i)} + \alpha}{n_{(t_{i-2}, t_{i-1})}^{(i)} + \mid T \mid\alpha}\\
	\end{align}\
$$

は$i$番目の品詞の確率を与えます。

$n_{(\cdot)}^{(i)}$は$i-1$番目までの観測結果を反映しているカウントになっています。

次に

$$
	\begin{align}
		\frac{n_{(t_{i-1}, t_{i}, t_{i+1})}^{(i+1)} + I(t_{i-2} = t_{i-1} = t_i = t_{i+1}) + \alpha}{n_{(t_{i-1}, t_{i})}^{(i+1)} + I(t_{i-2} = t_{i-1} = t_i) + \mid T \mid\alpha}\cdot\\
	\end{align}\
$$

は$i+1$番目の品詞の確率を与えます。

$n_{(\cdot)}^{(i+1)}$は$i$番目までの観測結果を反映しているカウントになっています。

つまり、式(37)の状態に比べて観測回数が1多くなっています。

従って、たとえば$t_{i-2} = t_{i-1} = t_i = t_{i+1} = 2$の場合、$n_{(2, 2, 2)}^{(i+1)} = n_{(2, 2, 2)}^{(i)} + 1$である必要があります。

式(25)の記号上は同じテーブル$n_{(2, 2, 2)}$だったとしても、どの項に含まれているかによって、上記のように観測回数をカウントに上乗せしなければなりません。

他の$n_{(\cdot)}$についても同様のことが言えます。

## おわりに

私はAppendixが充実している論文が好きなので、この論文は解読するのに苦労しました。

式(25)の導出ですが、私はおそらく誰かに指摘されなければ補正項$I(\cdot)$の必要性すら思いつかないので、もし自分でこの手法を開発したら補正項を付け忘れてしまうと思います。

こういった細かい部分まで気を配れるようになりたいです。

今回は状態数が固定のHMMでしたが、[The Infinite Hidden Markov Model](http://mlg.eng.cam.ac.uk/zoubin/papers/ihmm.pdf)のようにデータから状態数も学習できるモデルがあるため、今後実装したいと思います。