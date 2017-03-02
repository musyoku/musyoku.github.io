---
layout: post
title: Infinite Hidden Markov Modelによる教師なし品詞推定
category: 実装
tags:
- 自然言語処理
excerpt_separator: <!--more-->
---

## 概要

- [The Infinite Hidden Markov Model](http://mlg.eng.cam.ac.uk/zoubin/papers/ihmm.pdf)を読んだ
- C++で実装した

<!--more-->

## はじめに

Infinite Hidden Markov Model（以下IHMM）は、隠れマルコフモデル（HMM）の状態数をデータから決定することができるモデルです。

HMMにおける状態遷移確率をディリクレ過程によって生成することで無限の状態数を扱います。

それぞれの状態からの遷移確率に共通の（離散的な）基底分布を与えることで現在の状態数を共有し、さらにこの基底分布の基底分布として連続分布を与えることで新しい状態を無限に生成することができます。

## 参考文献

- [最近のベイズ理論の進展と応用 (III) ノンパラメトリックベイズ](http://chasen.org/~daiti-m/paper/ieice10npbayes.pdf)
	- IHMMによる「不思議の国のアリス」の学習例が載っています
- [ノンパラメトリックベイズ法](http://www.phontron.com/slides/110510-keihanna.pdf)
- [続・わかりやすいパターン認識―教師なし学習入門―](https://www.amazon.co.jp/dp/427421530X)
	- ディリクレ過程や隠れマルコフモデルの分かりやすい解説が載っています

## HMM

まずマルコフモデルについて簡単に説明します。

マルコフモデルは複数の**状態**を持ち、ある状態から別の状態へ一定の確率で遷移します。

この確率を**遷移確率**と呼び、状態の遷移後、その状態に依存した一定の確率で**出力記号**を出力します。

この確率を**出力確率**と呼びます。

図で表すと以下のような動作になります。

![image](/images/post/2017-02-27/markov_model.png)

まず初期状態を表す特殊な状態から$s_{t-1}$に遷移します。

遷移したら$p(y\mid s_{t-1})$から出力記号を生成し出力します。

次に$p(s_t\mid s_{t-1})$に従って遷移先を決定します。

遷移したら$p(y\mid s_{t})$から出力記号を生成し出力します。

マルコフモデルでは状態を$s$で表し、出力記号を$y$で表すのが慣例のようです。

今回は品詞推定を扱うので、状態が品詞に対応し、出力記号が単語に対応します。

ただし、通常のテキストデータでは単語列のみ入手可能で品詞は分かりません。

このように出力系列のみ観測可能で、状態系列が分からないモデルのことを隠れマルコフモデルといいます。

## IHMM

HMMでは遷移確率$A$と出力確率$B$は以下のような行列で表します。

$$
	\begin{align}
	A = \begin{pmatrix}
			0.1 &0.7 &0.2\\
			0.2 &0.1 &0.7\\
			0.7 &0.2 &0.1\\
		\end{pmatrix}\
	B = \begin{pmatrix}
			0.9 &0.1\\
			0.6 &0.4\\
			0.1 &0.9\\
		\end{pmatrix}\
	\end{align}\
$$

ここでは状態数を3、出力記号数を2としています。

（遷移確率があらかじめ与えられているのはHMMではなくただのマルコフモデルですので、ここでは推定した遷移確率と考えてください）

この行列は行が品詞を表しており、列が遷移先の品詞と出力記号に対応しています。

![image](/images/post/2017-02-27/markov_transition.png)

分かりやすくするため時刻$t$の添字を付けてありますが、実際は時刻に関係なく同じ行列を使います。

この図の枠内の確率をよく見ると総和が全て1になっていますが、マルコフモデルではこのような多項分布を用いて遷移確率や出力確率を表します。

IHMMではこの多項分布を無限次元に拡張することで無限の状態を扱います。

この無限次元の多項分布を作るために用いられるのがディリクレ過程と呼ばれる確率過程です。

詳しくは触れませんが、ディリクレ過程は元となる分布（基底測度、基底分布）に似た分布を生成することができます。

IHMMではそれぞれの遷移確率（上の行列での各行の多項分布）や出力確率が、共通の基底分布$G_0$を持つディリクレ過程に従っていると考えます。

図で表すと以下のようになります。

![image](/images/post/2017-02-27/hmm_dirichlet.png)

この$G_0$は論文では$\rm oracle$と呼ばれています。

さらにこの$G_0$は一様分布$H$を基底分布に持つディリクレ過程に従っています。

![image](/images/post/2017-02-27/ihmm_dirichlet.png)

出力確率は省略しましたが遷移確率と同様の生成過程を持っています。

次に、品詞$s_1\ldots s_t$を観測した状態で、$s_{t+1}$の事後予測確率を論文の記号を用いて表すと以下のようになります。

$$
	\begin{align}
		p(s_{t+1} = j \mid s_t=i, \beta, \gamma) &= \frac{n_{ij}}{\sum_{j'=1}^{K}n_{ij'} + \beta}
					+\frac{\beta}{\sum_{j'=1}^{K}n_{ij'} + \beta}G_0(s_{t+1}=j \mid \gamma)\\
		G_0(s_{t+1} = j \mid \gamma) &= \frac{n_{j}^o}{\sum_{j'=1}^{K}n_{j'}^o + \gamma}
					+\frac{\gamma}{\sum_{j'=1}^{K}n_{j'}^o + \gamma}H(s_{t+1}=j)\\
		H(s_{t+1}=j) &= \frac{1}{K+1}\\
	\end{align}\
$$

$n_{ij}$は品詞$i$に続いて品詞$j$が出現した回数、$\sum_{j'=1}^{K}n_{ij'}$は品詞$i$に続いて出現した全ての品詞の回数です。

$n_{j}^o$は品詞$j$が出現した回数、$\sum_{j'=1}^{K}n_{j'}^o$は全ての品詞の出現回数の総和です。

$K$は現在の総品詞数です。

式(2)と(3)の事後予測確率は経験分布と基底分布との混合分布の形になっています。

![image](/images/post/2017-02-27/ihmm_posterior.png)

そのため$n_{ij}=0$の場合でも第二項の基底分布を用いて補完することで0ではない値を返すことができます。

さらに、今まで観測したことがない新しい品詞$j=K+1$の場合、$n_{ij}=0$かつ$n_j^o=0$ですが、式(4)の一様分布があるため値を計算することできます。

その場合は補完の補完を経由しているため非常に小さな値になりますが、$j=K+1$に限らず$j>K$について、式(2)を用いることで何らかの確率を割り当てることができます。

このようにしてIHMMは無限の状態を扱っており、基底分布との混合分布になっているおかげで無限次元の遷移確率と出力確率を計算することができるようになっています。

## 二段階の生成過程について

$G_0$を経由せずいきなり$H$から$p(s_{t+1}\mid s_t)$を生成すれば良いのでは？という疑問があると思います。

式で書くと以下のようになります。

$$
	\begin{align}
		p(s_{t+1} = j \mid s_t=i, \beta, \gamma) &= \frac{n_{ij}}{\sum_{j'=1}^{K}n_{ij'} + \beta}
					+\frac{\beta}{\sum_{j'=1}^{K}n_{ij'} + \beta}H(s_{t+1}=j)\\
	\end{align}\
$$

これでも全ての状態$j$への遷移確率が計算できてしまうため問題はありませんが、不自然なモデルになってしまいます。

この式では$n_{ij}$が0だった場合常に一様分布を用いて補完することになりますが、一様分布による補完をするということは、品詞$i$の後ろには全ての品詞が平等に出現するという仮定を置くことに相当します。

しかし、品詞の出現頻度は均等ではなく、高い頻度で出現するものもあればほとんど出現しないものもあるのが一般的ですので、そういった性質を無視して一様分布で補完するのは不自然です。

したがって、この品詞の出現のしやすさを事前分布としてモデルに組み込むために$G_0$（oracle）が必要になってきます。

再掲しますが、IHMMではある品詞の出現回数を全ての品詞の出現回数の総和で割ることで、各品詞の事前分布を作っています。

$$
	\begin{align}
		G_0(s_{t+1} = j \mid \gamma) &= \frac{n_{j}^o}{\sum_{j'=1}^{K}n_{j'}^o + \gamma}
					+\frac{\gamma}{\sum_{j'=1}^{K}n_{j'}^o + \gamma}H(s_{t+1}=j)\nonumber\\
	\end{align}\
$$

また$G_0$を経由するもう一つの理由は、現在の品詞数$K$をすべての状態間で共有させるためです。

後述しますが、実際に品詞推定を行う場合は、現在存在しない新しい品詞を無限に考えるのではなく、今ある品詞$j \leq K$に加えて$j=K+1$の一つだけ考えます。

$G_0$があれば$n_j$の種類の数がそのまま$K$になります。

## 出力確率

状態$s_1 \ldots s_t$と出力$y_1\ldots y_t$が与えられた状態での$y_{t+1}$の予測確率は以下のようになります。

$$
	\begin{align}
		p(y_{t+1} = w \mid s_{t+1}=i, \beta^e, \gamma^e) &= \frac{m_{iq}}{\sum_{q'=1}^{K}m_{iq'} + \beta^e}
					+\frac{\beta^e}{\sum_{q'=1}^{K}m_{iq'} + \beta^e}G_0^e(y_{t+1}=w \mid \gamma^e)\\
		G_0^e(s_{t+1} = j \mid \gamma^e) &= \frac{m_{j}^o}{\sum_{j'=1}^{K}m_{j'}^o + \gamma^e}
					+\frac{\gamma^e}{\sum_{q'=1}^{K}m_{q'}^o + \gamma^e}H^e(y_{t+1}=w)\\
		H^e(y_{t+1}=w) &= \frac{1}{\mid W \mid}\\
	\end{align}\
$$

## ギブスサンプリングによる品詞の推定

品詞の推定にはギブスサンプリングを用います。

まずテキストデータを読み込んだ後、各単語にランダムに品詞を割り当て、$n_{ij}$、$n_j^o$、$m_{iq}$、$m_{j}^o$を適切にインクリメントします。

ギブスサンプリングでは現在の品詞列$s_1 \ldots s_t \ldots $について、品詞を一つづつサンプリングして更新していきます。

ここでは$s_t$をサンプリングして更新する場合を考えます。

まず品詞$s_t$が影響を与える確率は$p(y_t \mid s_t = j)$、$p(s_t = j \mid s_{t-1})$、$p(s_{t+1} \mid s_t = j)$の3つです。

![image](/images/post/2017-02-27/hmm_gibbs.png)

ちなみに$s_t$が影響を与える確率に含まれる周りの変数のことをマルコフブランケットと呼ぶそうです。

サンプリングするにはまず現在の$s_{t-1}$、$s_{t}$、$s_{t+1}$、$y_{t}$のカウントをモデルから削除します。

これは対応する$n_{ij}$、$n_j^o$、$m_{iq}$、$m_{j}^o$を1減らすとできます。

次に以下の確率を$j = 1 \ldots K+1$のそれぞれに付いて計算し、その値に比例した割合で$s_t^{new}$サンプリングします。

$$
	\begin{align}
		p(s_t = j \mid y_t) \propto p(y_t \mid s_t = j) \cdot p(s_t = j \mid s_{t-1}) \cdot p(s_{t+1} \mid s_t = j)
	\end{align}\
$$

最後に$s_{t-1}$、$s_{t}^{new}$、$s_{t+1}$、$y_{t}$のカウントをモデルに追加します。（対応する$n_{ij}$、$n_j^o$、$m_{iq}$、$m_{j}^o$を1増やします）

これを品詞列のすべての品詞について繰り返していくと正しい品詞列に収束するそうです。

## 中華料理店過程

ディリクレ過程を実装する際はそれと等価な中華料理店過程（CRP）を用います。

説明は省略しますが今回のIHMMでは以下のような構成になっています。

![image](/images/post/2017-02-27/hmm_crp_1.png)

この図は品詞列$s_1 \ldots s_{t+1}$が与えられた時にそれを客としてレストランに追加した後の状態を表しています。

テーブルの番号は品詞を、黒丸はそれぞれの客を表しています。

oracleは変数を用いてそれぞれの頻度をカウントしますが、遷移確率$p(s_{t+1}=j\mid s_t=i)$は$i=1 \ldots K$のそれぞれについてレストランを用いて客を管理します。

$n_{ij}$はレストラン$i$における、品詞$j$を提供するテーブルに居る客の総数を表します。

上の図では$n_{1,3}=2, n_{1,1}=2, n_{1,2}=1$となります。

この状態で、ギブスサンプリングによって$s_t$を更新する場合を考えます。

わかりやすく$s_{t-1} = 1$、$s_t=3$、$s_{t+1}=2$とします。

まず$s_t$をモデルから削除する必要がありますが、この時影響を受ける客は、

- レストラン$i=1$でテーブル3に座っている客
- レストラン$i=3$でテーブル2に座っている客

の2人です。

図の品詞3のように同じテーブルが複数ある場合は、それぞれのテーブルの客数に比例した確率でテーブルを選択し、そこから一人削除します。

例えばテーブルが3つあり、客数がそれぞれ5,6,7だとすると、$\frac{5}{18},\frac{6}{18},\frac{7}{18}$の多項分布からテーブルを確率的に決定します。

この時、客を消すことによってテーブルが空になった場合、oracleのカウントを1下げます。

![image](/images/post/2017-02-27/hmm_crp_2.png)

$n_j^o$はテーブルが空になった時のみカウントが減ります。

客を消すたびに1減らすわけではありません。

次に、ギブスサンプリングした結果$s_t$が4になったとします。

モデルに$s_t$を追加する必要がありますが、この時客を追加するテーブルは

- レストラン$i=1$のテーブル4
- レストラン$i=4$のテーブル2

の２つです。

客を追加する際、テーブルが1つも存在しなければ作成します。

テーブルを作成した場合はoracleを1増やします。

![image](/images/post/2017-02-27/hmm_crp_3.png)

それ以外の場合は以下の手順で客を追加します。

- 対象の品詞を提供しているテーブルに着席しているそれぞれの客数と$\beta$からなる多項分布からテーブルを決定する
- $\beta$が選ばれた場合はテーブルを作成し客を追加、さらにoracleを1増加
- 既存のテーブルが選ばれた場合は客を追加し、oracleには触らない

具体的には、例えば品詞2を提供するテーブルが２つあり、客数がそれぞれ4と5だとします。

また$\beta=1$とすると、$\frac{4}{10},\frac{5}{10},\frac{1}{10}$の多項分布から確率的にテーブルを選択します。

この$\beta$は式(2)に出てきます。

ここまでは遷移確率の実装について書きましたが、出力確率も同様にして実装します。

その場合は品詞ごとにレストランを用意し、単語のテーブルを管理します。

## 実装

C++で実装しましたがPythonのラッパーも作っています。

[https://github.com/musyoku/unsupervised-pos-tagging/tree/master/infinite-hmm](https://github.com/musyoku/unsupervised-pos-tagging/tree/master/infinite-hmm)

現在も開発途中であり、ハイパーパラメータのサンプリングなどは未実装です。

あとバグがある可能性もあります。

## 人口データ

バグがあるかどうかを調べるためのテストデータを作りました。

上のgithubのコードに含まれている`generate_test_sequence.py`を使います。

状態遷移確率と出力確率は以下のように設定しました。

$$
	\begin{align}
	A = \begin{pmatrix}
			0.4 &0.1 &0.4 &0.1\\
			0.2 &0.3 &0.2 &0.3\\
			0.3 &0.2 &0.1 & 0.4\\
			0.1 &0.4 &0.3 & 0.2\\
		\end{pmatrix}\
	B = \begin{pmatrix}
			0.3 &0.7 &0.0 &0.0 &0.0 &0.0\\
			0.0 &0.0 &1.0 &0.0 &0.0 &0.0\\
			0.0 &0.0 &0.0 &1.0 &0.0 &0.0\\
			0.0 &0.0 &0.0 &0.0 &0.7 &0.3\\
		\end{pmatrix}\
	\end{align}\
$$

単語0と単語1は品詞1から生成されます。

単語2は品詞2のみから生成されます。

単語3は品詞3のみから生成されます。

単語4と単語5は品詞4から生成されます。

（注：単語は0から開始しています）

具体的には以下のような出力が得られます。

0 3 5 2 3 5 2 2 1 1 1 3 1 1 3 5 2 2 2 1 2 4 5 3 2 2 2 1 5 3 2 2 2 3 1 0 2 2 3 2 0 1 3 2 4 1 0 1 3 0 3 3 5 3 3 2 4 2 1 1 1 1 1 0 3 4 4 2 2 4 2 4 2 2 1 3 1 4 4 1 5 3 2 3 4 2 2 4 2 4 3 3 2 4 2 2 0 1 1 0 3 3 3 4 2 3 2 2 3 4 3 1 2 3 4 2 4 1 2 3 4 4 2 4 4 2 4 3 2 5 3 0 2 2 2 3 1 4 5 3 3 1 1 4 3 1 3 5 2 1 3 2 1 2 5 2 1 0 1 1 3 3 3 4 2 4 2 2 2 4 0 2 0 1 3 2 4 3 4 4 2 4 2 2 3 1 0 1 3 5 2 4 4 3 4 2 2 2 3 4

ここから隠れた品詞列を推定しました。

以下が推定結果です。

```
tag 1:
	3/7789, 
tag 2:
	4/4148, 5/1753, 1/11, 3/2, 2/1, 
tag 3:
	1/6883, 0/2832, 2/15, 
tag 4:
	<eos>/200, 
tag 5:
	2/4478, 5/6, 1/1, 
tag 6:
	2/1093, 3/638, 5/63, 
tag 7:
	2/2222, 4/1465, 5/580, 0/7, 1/6, 
tag 8:
	2/2213, 3/1520, 4/1323, 5/589, 1/210, 0/152, 
```

#/#の表記は左側が単語、右側はその品詞に割当てられた回数を表しています。

tag 5までの結果を見ると、設定通り単語0と1が同じ品詞に割り当てられ、単語4と5も同じ品詞に割り当てられています。

また、単語2と3はそれぞれ違う品詞に割り当てられています。

tag 6以降は正しく推定できなかった部分だと考えられます。

## 不思議の国のアリス

前処理として出現頻度が1の単語を`<unk>`に置き換えて実験を行いました。

初期品詞数は2で学習を開始しました。

ギブスイテレーションを１万回行った結果が以下になります。

```
tag 1:
	<eos>/1397, ./1209, !/115, ?/65, too/2, angrily/2, contemptuously/2, sharply/1, 
tag 2:
	the/1632, a/622, her/156, his/96, this/71, your/62, its/57, an/57, their/47, my/46, some/37, one/35, no/33, any/29, very/28, that/25, another/20, two/11, these/9, all/9, which/8, 
tag 3:
	<unk>/162, queen/75, thing/66, king/64, head/60, hatter/57, gryphon/55, mock/55, way/53, time/51, voice/51, rabbit/49, mouse/48, duchess/42, tone/42, dormouse/40, cat/38, eye/36, minute/32, door/32, hand/31, 
tag 4:
	she/553, i/519, it/323, you/252, alice/155, they/141, he/123, and/110, there/89, that/72, who/56, what/48, we/34, how/29, which/26, to/16, this/16, without/14, half/7, soo/7, where/7, 
tag 5:
	,/2418, !/331, ?/111, as/104, '/55, and/50, that/34, turtle/33, when/28, or/17, up/12, but/12, till/11, more/10, <unk>/7, before/7, than/5, mind/5, howl/4, ootiful/4, choke/3, 
tag 6:
	and/609, say/372, but/158, as/116, S/103, that/99, if/96, s/94, so/91, what/67, when/51, for/50, or/39, then/35, how/33, thought/19, while/18, perhaps/17, cry/16, think/16, because/15, 
tag 7:
	be/310, go/179, <unk>/138, get/113, say/103, look/95, come/79, make/76, do/70, see/70, like/67, begin/64, find/52, take/45, hear/45, tell/42, sit/36, put/34, try/33, ask/33, grow/32, 
tag 8:
	in/330, to/247, at/212, with/174, on/101, all/74, about/71, into/67, by/58, for/46, after/42, from/35, down/34, such/33, out/31, of/26, upon/26, like/25, over/24, round/21, off/19, 
tag 9:
	alice/232, <unk>/110, then/59, now/54, oh/45, well/45, again/41, why/40, here/33, not/33, dear/28, turtle/27, two/26, all/25, before/25, yet/25, no/24, just/23, however/20, down/19, soup/18, 
tag 10:
	little/127, <unk>/105, great/39, very/35, march/35, good/30, white/30, other/28, large/26, same/24, first/21, right/19, poor/18, long/17, next/16, curious/16, last/13, whole/13, three/10, own/10, queer/10, 
tag 11:
	it/263, her/92, on/89, up/88, them/88, that/85, herself/83, me/66, <unk>/59, you/56, down/48, him/43, again/42, off/38, this/34, out/34, back/33, all/29, away/25, once/25, about/23, 
tag 12:
	to/345, n't/217, not/112, you/102, very/71, be/62, quite/46, so/44, no/33, all/31, have/31, just/29, i/25, too/24, never/24, well/22, only/21, as/18, <unk>/13, always/13, hardly/12, 
tag 13:
	be/563, have/290, do/183, will/111, would/96, could/86, must/44, seem/40, can/36, should/32, ca/28, might/28, begin/28, wo/24, never/24, shall/23, feel/22, only/18, want/15, ought/14, may/14, 
tag 14:
	<unk>/137, out/52, one/50, much/40, more/31, sort/23, use/22, time/21, end/19, tea/18, enough/18, change/16, long/16, pig/16, some/14, room/13, cat/12, fan/12, large/12, rest/12, ear/11, 
tag 15:
	'/143, know/83, think/79, say/42, see/27, wonder/22, wish/21, speak/17, suppose/14, glad/11, sure/11, mean/11, afraid/8, oop/7, hope/7, believe/6, find/5, dare/5, wait/4, feel/4, fond/4, 
tag 16:
	of/484, to/121, and/101, for/44, in/37, as/23, than/19, or/19, without/9, eat/6, along/6, their/5, !/4, hang/4, since/4, either/3, all/3, worse/2, time/2, around/2, legged/2, 
```

名詞、形容詞、動詞などが綺麗に分かれているように見えます。

これをプロットしたものが以下になります。

![image](/images/post/2017-02-27/pos_alice.png)

この図は行ごとに正規化を行っています。

例えば正解品詞MDは品詞13に割り当てられていることを表しています。

理想的には横も縦もピークが1箇所になれば良いのですが、現状1つの予測品詞に複数の正解品詞が含まれています。

また最初に挙げた参考文献（[最近のベイズ理論の進展と応用 (III) ノンパラメトリックベイズ](http://chasen.org/~daiti-m/paper/ieice10npbayes.pdf)）で同様の実験が行われていますが、その実験では品詞数は7で収束したと書かれています。

しかし私の実装では低頻度語の`<unk>`への置き換えがない場合に品詞数が爆発し、置き換えてもなお7では収束しませんでした。

文献の方はギブスサンプリングではなくビームサンプリングを用いている可能性が高いのですが、単純に私の実装に不具合がある可能性もあります。

またハイパーパラメータの推定部分の実装がまだ完了していないため、現状では何が正しいのか分かりません。

## Wikipedia

英語版wikipediaのデータから10万文を取り出して学習させました。

初期品詞数は10とし、2万回のギブスイテレーションを行ないました。

行った前処理は

- 出現回数が1以下の単語を全て`<unk>`に置き換える
- 単語が数字のみで構成されるものを##に置き換える

です。

以下が学習結果です。

**tag 1:**

`<eos>`/99553, 

**tag 2:**

the/148725, a/45597, his/10937, an/8233, their/4848, this/4505, its/3920, other/2818, her/2740, two/2059, many/1782, all/1748, some/1730, new/1630, no/1599, these/1597, "/1429, most/1342, more/1306, that/1296, any/1292, several/1166, three/1074, one/919, very/918, each/854, both/836, another/796, high/778, which/656, four/653, ##/624, every/596, various/580, north/542, local/536, different/464, my/445, such/372, southern/326, low/325, early/325, certain/322, our/312, five/297, British/287, modern/286, your/284, great/270, much/257, recent/251, public/242, eastern/234, social/233, numerous/229, traditional/219, former/218, good/216, six/213, original/212, human/210, little/210, los/210, western/209, natural/197, northern/196, least/195, American/194, those/186, south/186, united/179, international/177, extremely/175, whose/174, only/168, heavy/164, strong/164, relatively/162, multiple/156, hong/143, less/143, late/143, too/141, grand/141, central/140, fort/137, 3/132, either/131, political/129, Russian/128, seven/122, Australian/121, '/121, old/121, private/119, complete/118, by/118, national/118, special/115, major/114, French/114, 

**tag 3:**

of/76028, in/48675, for/16848, with/14285, by/13428, on/12437, as/11672, from/9748, at/9201, into/2823, after/2665, during/2423, under/1822, than/1702, between/1701, over/1699, through/1435, :/1387, about/1237, when/1148, against/1123, "/1115, include/1009, until/782, within/764, around/730, like/702, since/631, call/606, while/598, before/563, among/516, throughout/398, near/393, across/370, follow/358, without/335, feature/327, contain/315, off/300, along/295, towards/259, if/235, via/218, above/212, up/187, win/183, //181, down/172, behind/171, inside/167, toward/163, despite/138, give/137, provide/130, below/121, onto/117, name/112, reach/99, outside/98, beyond/92, cover/88, represent/81, cause/75, star/75, require/70, per/69, concern/66, back/65, wear/65, alongside/64, receive/62, hold/62, gain/57, cross/50, entitle/49, enable/47, surround/47, support/47, century/47, occupy/46, face/46, whose/44, enter/44, release/42, carry/41, aboard/41, past/40, opposite/39, mark/37, depict/35, whereby/34, comprise/33, possess/33, read/32, adopt/31, versus/30, champion/30, affect/29, exceed/27, prove/25, 

**tag 4:**

one/4042, work/1633, him/1346, two/1104, most/835, well/830, back/804, more/790, history/778, water/745, lead/576, fact/531, friend/527, love/478, return/476, it/463, artist/441, addition/433, boy/425, her/396, three/380, thing/374, technology/371, governor/366, similar/354, letter/352, earth/348, low/345, education/339, report/334, individual/327, light/318, some/317, law/311, india/311, method/304, tradition/295, remain/295, themselves/285, himself/282, england/282, representative/273, reference/261, strong/260, drug/258, weapon/257, china/257, all/247, talk/239, four/239, museum/228, window/226, california/222, singer/219, job/219, color/216, staff/212, distance/210, fight/209, politics/204, turn/203, west/202, those/201, britain/200, religion/193, writing/192, murder/187, control/187, justice/185, producer/184, another/184, code/181, schedule/179, peace/177, happen/175, relate/174, wheel/170, lot/169, prison/167, weight/163, shape/161, each/159, poetry/156, texas/155, depend/154, chairman/153, gas/149, estimate/149, germany/147, 19th/147, exchange/147, hope/147, fast/146, unknown/146, aim/146, ministry/145, independence/145, forward/144, italy/144, australia/142, human/142, 

**tag 5:**

`<unk>`/13659, first/3974, ##/3434, new/2402, world/1999, unite/1540, large/1417, population/1323, American/1306, second/1273, "/1219, same/1199, national/1136, small/1124, early/1077, old/1076, good/1060, own/1048, age/1028, way/1016, state/983, great/923, york/866, south/861, single/819, team/809, late/796, major/765, last/750, art/702, public/698, few/695, main/689, British/683, land/681, English/680, general/675, long/657, common/651, only/644, young/642, female/632, side/630, former/595, household/592, median/580, final/569, german/562, popular/557, story/546, international/539, top/538, third/536, important/529, family/502, Indian/496, political/494, official/470, short/465, mother/459, ground/446, current/446, use/444, military/440, Canadian/437, central/436, royal/436, property/416, special/412, west/412, home/388, car/382, musical/378, follow/373, sea/367, roman/366, full/365, summer/356, civil/346, famous/346, husband/340, red/320, western/320, social/318, independent/311, European/309, Japanese/307, rock/307, health/305, poverty/303, professional/303, and/301, Christian/300, primary/293, enemy/290, change/290, real/285, future/281, temple/280, federal/280, human/278, 

**tag 6:**

year/2742, state/2305, school/2240, name/2074, area/1895, group/1703, member/1574, number/1347, band/1345, album/1313, company/1303, party/1290, people/1229, system/1204, line/1203, man/1183, life/1183, song/1166, house/1153, series/1148, day/1108, service/1097, other/1078, family/1054, force/1029, end/1004, result/983, power/965, country/951, century/917, language/826, show/781, base/769, title/740, army/679, size/670, village/663, case/660, design/659, period/639, battle/635, right/635, program/634, record/632, role/612, career/593, event/592, club/592, region/589, society/572, office/566, college/552, development/544, president/542, king/530, position/529, model/525, ship/513, park/513, class/506, movement/500, student/474, river/471, type/467, hand/467, project/465, council/465, brother/464, production/462, present/461, attack/461, action/460, theory/453, problem/449, board/449, person/448, union/444, interest/432, range/421, sound/419, award/419, issue/414, department/410, network/407, source/405, fire/397, kingdom/390, structure/384, idea/383, star/376, store/376, success/375, home/373, stage/372, daughter/370, view/368, product/364, night/360, episode/357, section/357, month/352, 

**tag 7:**

as/6087, "/5227, not/4708, that/4392, also/3300, make/2660, out/2131, know/2079, when/1866, `<unk>`/1724, do/1433, so/1406, give/1378, if/1220, where/1169, often/1146, now/1076, because/1069, although/1067, well/1055, come/1037, even/993, more/954, later/943, release/930, still/926, say/904, though/776, much/774, consider/767, with/697, n't/695, kill/669, most/642, after/634, usually/618, follow/604, establish/604, help/601, bear/598, just/593, begin/590, start/571, publish/570, since/567, once/521, base/507, along/506, it/498, name/482, sometimes/480, like/475, originally/474, never/470, `<repdns`/437, replaced-dns/437, soon/427, for/423, think/418, what/397, almost/363, tell/357, receive/354, defeat/347, before/343, actually/336, her/335, spread/335, how/332, a/329, upon/328, elect/328, operate/325, finally/324, away/323, seem/305, introduce/295, those/294, far/285, alone/283, cover/283, grow/281, thus/271, yet/270, instead/258, whether/258, about/254, launch/246, simply/242, teach/242, probably/241, despite/234, draw/232, mainly/230, quickly/222, me/221, commonly/219, long/218, identify/215, you/214, act/213, 

**tag 8:**

and/9845, be/6774, but/4644, use/3898, have/2517, such/2005, include/1857, up/1814, play/1763, take/1693, only/1645, see/1594, become/1564, find/1464, go/1423, write/1287, all/1217, while/1149, call/1093, or/1078, serve/1058, hold/1016, appear/1014, create/1009, win/999, leave/942, produce/933, form/928, then/908, build/908, locate/901, him/897, run/873, lead/840, allow/804, work/800, change/732, which/725, many/720, get/711, move/698, provide/698, live/691, join/642, them/639, place/626, bring/611, both/606, develop/588, perform/578, continue/578, lose/564, die/556, mean/552, open/534, some/531, show/529, meet/516, found/513, sell/508, claim/504, return/502, send/500, describe/494, cause/493, keep/484, marry/482, down/476, feature/470, require/462, off/457, put/453, turn/447, support/440, reach/436, pass/435, replace/434, add/434, want/427, record/425, especially/417, remain/414, rather/413, believe/412, involve/408, on/408, ask/405, enter/399, complete/398, available/396, offer/393, occur/384, carry/383, set/376, represent/370, look/365, study/364, destroy/363, fight/361, always/354, design/352, 

**tag 9:**

##/58430, `<unk>`/7653, 1/1463, 2/996, 5/838, bear/827, 3/751, `|`/636, 4/617, over/567, 6/528, 7/521, 8/430, American/405, old/350, 9/330, mi²/310, :/303, text/297, japan/283, km²/262, france/251, b/233, c/211, london/206, individual/193, california/190, australia/189, e.g./182, paris/181, i.e./178, germany/171, Canada/167, St./166, ohio/160, f/154, india/144, family/142, us/142, Washington/139, n/126, Ontario/121, a/121, English/120, brazil/118, h/117, illinois/115, spain/112, etc./109, ireland/109, England/108, italy/108, m/108, Virginia/108, scotland/103, page/102, d/102, London/102, 0/101, texas/101, china/100, x/99, it/96, Sweden/95, now/94, 1st/94, Argentina/89, chicago/89, pennsylvania/84, massachusetts/84, austria/82, Oxford/81, norway/79, guitar/79, l/79, Florida/78, philadelphia/75, england/74, berlin/74, Quebec/73, Switzerland/73, russia/73, spring/72, Georgia/72, Judah/72, german/71, Japanese/70, poland/69, June/69, bass/68, ten/68, music/67, Arabic/66, ?/66, die/66, connecticut/65, r/65, Usa/64, Uk/62, Iowa/62, particular/62, 

**tag 10:**

time/3373, city/2417, part/1865, game/1673, war/1497, music/1368, town/1260, university/1207, film/1173, book/1160, high/957, player/956, government/950, race/945, point/865, child/858, death/851, island/847, character/835, average/797, term/783, station/772, church/760, place/749, son/735, community/730, form/727, league/707, version/693, building/687, woman/683, income/683, site/650, father/644, level/628, season/628, word/626, style/619, body/615, field/612, center/602, north/588, leader/579, east/577, court/568, district/566, original/555, law/550, study/543, process/507, order/507, road/505, election/498, total/484, research/483, head/471, movie/471, minister/459, organization/453, business/452, act/447, county/443, nation/439, air/437, next/431, effect/426, rock/425, standard/422, material/420, information/417, plan/411, operation/401, release/385, course/382, couple/379, girl/378, set/378, wife/376, rule/374, video/371, census/371, value/370, example/369, space/366, performance/366, bank/363, director/363, money/353, committee/347, novel/343, track/340, writer/335, density/328, list/327, computer/326, marriage/325, centre/325, appearance/324, province/324, cup/322, reason/319, 

**tag 11:**

)/30224, ,/9128, "/6014, %/5003, year/1170, km²/497, '/465, unit/457, c/451, million/450, people/434, household/426, km/421, ;/402, university/402, team/389, ii/387, together/368, mile/355, family/310, live/304, shift/302, mi²/299, m/288, championship/283, at/259, white/250, street/249, male/238, press/237, division/236, season/235, per/234, district/232, later/229, housing/224, foot/222, record/215, living/209, versus/192, county/188, game/170, edition/168, hall/167, iii/162, square/159, ?/156, itself/152, point/151, Asian/151, 3/149, begin/147, minute/145, percent/143, school/143, come/142, away/141, long/141, state/140, ago/139, day/138, billion/135, road/134, bc/129, inch/129, on/128, mm./125, yard/125, hour/123, acre/118, time/118, 's/115, station/114, :/113, 2/110, award/108, student/106, metre/105, cm/105, association/102, meter/101, dollar/101, college/99, system/97, orchestra/96, month/95, tend/94, vote/92, 5/92, page/89, version/89, w/89, pass/88, islander/86, d/85, degree/85, pacific/85, khan/84, player/82, again/82, iv/81, 

**tag 12:**

./85588, ;/6094, u.s./495, :/279, .../166, st./147, ##/119, s./86, page/81, etc./80, v./77, Mr./60, dr./54, Inc./46, No./43, Volume/39, Dr./37, mr./35, vs./33, !/29, no./29, d.c./29, bros./28, e.t./25, co./25, ltd./24, Mrs./22, S./15, mrs./15, f.c./15, ft./13, t.a.t.u./13, mass./13, a.d./13, blvd./12, h.r./12, Rev./12, p.m./11, inc./11, jr./11, ms./11, ave./11, volume/11, gen./11, rev./10, St./10, Ms./10, a.m./9, ?/8, c.j./8, u.n./7, hon./7, jan./7, j.j./6, vols./6, lbs./6, h.p./6, var./6, Dec./5, oct./5, Sept./5, sr./5, e.p./5, j.p./5, nos./5, b.c./5, capt./4, Prof./4, lt./4, cm./4, ch./4, Sgt./4, corp./4, c.e./4, approx./3, s.h.i.e.l.d./3, t.j./3, j.k./3, brig./3, h.w./3, ste./3, f.a./3, Pa./3, ore./3, m.b.a./3, c.s./3, p.s./3, j.r./3, incl./3, g.i./3, m.p./3, col./2, w.e.b./2, govt./2, j.s./2, g.f.j./2, m.b./2, Nov./2, samanta/2, ark./2, Sen./2, 

**tag 13:**

,/84543, be/67881, to/41463, and/41415, have/11914, (/8643, 's/7918, that/7818, or/3975, ;/2524, would/2256, can/1906, who/1572, will/1523, do/1477, also/1457, :/1231, become/1197, '/1137, which/1080, could/1011, may/876, -/759, by/676, take/601, where/598, make/539, begin/533, before/486, should/424, must/368, S/359, then/335, from/324, receive/297, //282, show/268, might/264, s/258, tell/238, contain/236, start/232, first/227, upon/210, living/208, --/191, without/188, remain/184, !/171, &/152, ever/151, get/146, because/136, grow/131, never/127, ?/123, eventually/123, state/120, break/113, leave/107, team/103, involve/102, whose/101, i/99, offer/97, .../87, once/87, regard/77, shall/77, into/72, ii/71, report/70, just/68, too/66, announce/66, themselves/64, order/58, attend/56, air/55, distinguish/55, declare/53, subsequently/52, exhibit/52, strike/52, figure/52, vary/51, insist/49, happen/48, constitute/47, gradually/47, nor/46, resemble/46, introduce/45, encompass/44, beneath/43, step/42, official/40, rarely/40, sport/39, against/38, frequently/38, 

**tag 14:**

"/5410, 's/3151, `<unk>`/2997, '/1991, or/891, radio/608, king/463, William/455, david/413, (/397, //360, big/341, S/326, television/301, black/290, entire/289, &/282, football/230, grand/222, Israel/219, red/213, van/208, martin/204, blue/202, college/200, frank/197, cd/194, joseph/193, basketball/190, queen/186, michigan/184, pop/183, film/182, de/177, corporation/177, Robert/174, green/174, white/172, b/168, French/162, fiction/161, steve/156, joe/154, drum/154, el/149, iron/147, edward/145, victoria/144, 1/142, online/141, douglas/140, super/134, Irish/132, chris/124, nuclear/123, n/120, lp/116, local/116, tv/115, paul/114, toronto/114, v/113, Williams/113, h/112, yellow/111, mare/109, g/109, tony/109, industrial/103, stone/100, twenty/99, 4/99, ken/98, river/98, dan/97, little/97, insurance/96, da/93, sport/93, rice/92, ski/91, Virginia/91, der/91, x/91, lewis/91, satellite/90, warren/90, mean/89, f-zero/88, married/88, coach/88, jim/88, commercial/87, audio/86, howard/86, class/86, rugby/86, korea/86, gay/85, entertainment/85, franklin/85, 

**tag 15:**

,/35917, (/20539, and/10058, in/4994, :/3926, to/2159, //2092, $/1437, from/1322, -/1210, or/1131, `|`/1001, ;/948, de/800, September/670, October/608, age/603, December/586, November/576, march/570, august/551, &/549, may/532, June/508, April/485, about/476, at/417, act/378, until/346, july/344, January/332, january/292, february/287, bear/262, when/253, fontsize/246, X/246, =/239, female/225, isbn/224, '/215, approximately/206, February/190, native/187, route/184, °/179, `<unk>`/167, exit/163, king/154, since/154, every/152, July/144, near/138, channel/128, on/126, include/119, von/118, african/114, la/113, around/110, et/108, year/108, del/107, £/102, between/99, where/99, record/98, road/92, chapter/90, highway/88, di/87, Fred/81, die/79, du/77, episode/75, +/72, le/72, see/69, ?/69, no/67, interstate/66, grade/65, volume/61, des/57, season/55, */55, side/55, abu/49, ad/48, west/48, model/47, number/47, summer/46, bwv/45, us/44, ×/43, class/43, baron/43, western/43, through/41, und/41, 

**tag 16:**

he/12398, it/9695, they/4409, which/3625, there/3373, she/3027, who/2849, this/2303, however/1889, to/1854, have/1232, i/1068, due/861, can/747, accord/741, refer/705, we/693, you/666, may/503, after/495, also/492, then/482, try/468, later/440, able/425, `<unk>`/413, these/374, decide/332, them/311, eventually/277, say/268, what/258, male/256, again/255, attempt/228, people/214, back/204, note/203, thus/195, sonic/191, will/191, today/188, currently/188, prior/185, other/181, Hispanic/174, recently/173, agree/172, should/171, move/166, home/162, here/158, continue/157, refuse/156, belong/151, fail/147, close/144, generally/141, believe/138, grant/137, unable/135, likely/132, sasuke/127, some/116, intend/116, initially/111, compare/110, must/105, student/103, hard/99, olajuwon/97, crohn/93, therefore/93, enough/92, Shakespeare/90, access/90, difficult/88, subsequently/87, her/86, everyone/86, additionally/85, unfortunately/84, rise/84, suppose/83, previously/83, attach/81, hippolyta/81, tail/81, instead/81, meanwhile/80, jai/80, Moore/78, kipling/77, typically/76, nevertheless/75, god/75, carthage/73, wish/72, attribute/70, assign/70, argue/68, 

**tag 17:**

john/1115, which/969, them/901, those/525, order/424, san/401, age/385, james/384, example/379, president/333, henry/313, george/289, charles/288, Michael/276, robert/265, la/261, general/237, lord/227, peter/225, god/219, what/212, thomas/208, richard/206, sir/205, science/201, lee/189, emperor/186, Paul/176, bob/175, all/175, whom/174, mark/165, tom/162, prince/162, saint/160, louis/158, christ/157, George/155, least/153, america/149, jesus/147, st/147, europe/147, jack/144, late/144, captain/140, bill/136, canada/129, jackson/125, washington/124, lake/121, brian/120, mike/120, arthur/118, harry/113, don/113, director/112, mount/110, star/110, ray/109, pope/105, al/105, judas/100, jean/100, Daniel/97, me/92, andrew/91, professor/91, jerusalem/90, albert/90, princess/90, simon/89, reality/87, both/87, Richard/87, muhammad/86, Taylor/86, grant/85, jimmy/85, health/82, max/80, then/79, many/79, alice/78, d/77, wood/77, ben/76, old/76, Christian/73, philip/73, dave/71, johnny/71, robin/71, metal/71, man/70, lawrence/69, santa/68, ivan/68, ophrys/67, gary/67, Elizabeth/66, 

**tag 18:**

`<unk>`/23571, "/1314, on/1008, (/949, !/467, />/447, county/430, i/414, bar/327, a/239, smith/238, ii/190, e/188, under/187, francisco/182, .../182, brown/169, davis/166, mary/155, j/151, hill/123, jones/123, go/119, r/115, k/111, t/106, die/104, park/91, ¨/89, baehr/88, parker/86, miller/86, Lloyd/85, johnson/85, walker/84, y/82, thomas/77, o/76, taylor/76, brother/73, writer/70, anne/68, island/67, harris/65, beach/65, tommy/63, */63, adams/58, stewart/58, carter/57, evans/57, u/56, eighteen/55, Davis/55, singh/55, show/54, stern/54, butler/52, me/52, station/51, diego/49, Moore/49, studio/48, olive/46, Kennedy/46, vi/46, confirmation/45, move/45, yes/45, Clinton/45, z/44, alexander/44, rise/44, wen/43, recording/43, ab/42, carol/42, vii/42, viii/42, house/41, jacques/41, gardner/41, morgan/40, edwards/40, te/40, sing/39, session/39, Christopher/39, oh/39, young/38, Baehr/38, funkadelic/38, murphy/38, terry/37, five/37, morris/37, producer/36, communication/35, tour/35, sly/34, Bennett/34, 

**tag 19:**

six/190, honor/136, calendar/66, sector/62, activist/62, m/50, cave/50, forty/46, medicine/45, personality/43, freedom/39, bearer/35, delhi/34, cds/32, jam/32, jedi/31, id/31, once/25, chuan/25, privately/21, voting/20, pond/15, magician/15, wuo/15, atp/15, slalom/12, illegal/11, bryant/10, ascend/9, hurl/8, galicia/8, moby/8, Rousseau/7, lacrosse/7, cfb/7, ejaculation/7, trix/6, spicy/6, iec/6, vibe/6, tindal/6, nautical/5, aground/5, pigeon/5, cage/5, post-normal/5, yankovic/5, closer/5, bruges/4, hepworth/4, isaacs/4, straus/4, oceania/4, gallo/4, mte/4, unclos/4, inconspicuous/4, reckoning/4, sg/4, mackinac/4, megapol/4, wanderer/4, Redford/4, diva/4, tbf/3, sombre/3, charlottenburger/3, americana/3, self-identity/3, sledge/3, magdalena/3, parisse/3, pocomo/3, wilkinson/3, margarita/3, kc/3, bloodshot/3, nab/3, orthodontist/3, hendry/3, greeley/3, shalivahana/3, hattersley/2, adolphe/2, ksktu/2, felsenheimer/2, ryn/2, warm-blooded/2, bethany/2, generalisation/2, fri/2, gerber/2, screeching/2, cordite/2, priiez/2, manchukuo/2, grapefruit/2, schroeder/2, alms/2, jazz-fusion/2, ashland/2, 

**tag 20:**

-/2188, son/18, my/12, dong/8, gang/8, build-off/6, creator/6, talking/5, kari/4, aghnutai/4, turbocharged/3, edin/3, purple/3, imöng/2, raibu/2, hemsell/2, plop/2, liberate/2, yearley/2, qiao/2, zaria/2, fish-eagle/2, mcneil/2, qc7/2, m.p./2, bertie/2, halbertsma/2, filipina/2, hoya/2, taping/2, gon/2, casaba/2, pietra/2, misty/2, yamashta/2, crag/1, backer/1, "/1, 

![image](/images/post/2017-02-27/pos_wiki.png)

## 関連

- [A Fully Bayesian Approach to Unsupervised Part-of-Speech Tagging](/2017/01/28/A-Fully-Bayesian-Approach-to-Unsupervised-Part-of-Speech-Tagging/)
	- 品詞数を固定した場合の教師なし品詞推定です

## 終わりに

ギブスサンプリングの代わりにビームサンプリングを用いる手法が提案されています。

[Beam Sampling for the Infinite Hidden Markov Model](http://mlg.eng.cam.ac.uk/pub/pdf/VanSaaTehGha08.pdf)

githubで公開している私のIHMMの実装には開発途中のビームサンプリングの実装が含まれていますが、現状正しく動作しません。
