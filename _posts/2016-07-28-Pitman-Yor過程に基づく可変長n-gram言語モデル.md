---
layout: post
title: Pitman-Yor過程に基づく可変長n-gram言語モデル（VPYLM）
category: Chainer
tags:
- HPYLM
- VPYLM
excerpt_separator: <!--more-->
---

## 概要

- [Pitman-Yor過程に基づく可変長n-gram言語モデル](http://chasen.org/~daiti-m/paper/nl178vpylm.pdf) を読んだ
- C++でVPYLMを実装した

<!--more-->

## はじめに

VPYLMは[HPYLM](/2016/07/26/A_Hierarchical_Bayesian_Language_Model_based_on_Pitman-Yor_Processes/)を拡張し、HPYLMでは固定だったn-gramオーダーを各単語ごとに推定できるようになったモデルです。

HPYLMと同様に、文脈$h=w_{t-n}...w_{t-1}$に続く単語$w$の確率は

$$
	\begin{align}
		P(w\mid h)=\frac{c(w\mid h) - d\cdot t_{hw}}{\theta + c(h)}+\frac{\theta + d\cdot t_h}{\theta + c(h)}P(w\mid h')
	\end{align}\
$$

となります。

ただし$h'=w_{t-n+1}...w_{t-1}$はn-gramのオーダーを一つ落とした文脈、$c(w\mid h)$はレストラン$h$でのテーブル$w$にいる客の総数、$c(h)=\sum_{w}c(w\mid h)$はその総和、$t_{hw}$は$w$を提供するテーブルの総数、$t_h=\sum_{w}t_{hw}$はその総和を表します。


## 停止確率

VPYLMでは文脈木の各レストラン$u$に、木をルートからたどるときにそこで止まる確率$q_u$があると考えます。

そしてこれらの停止確率は共通のベータ事前分布から生成されていると仮定します。

$$
	\begin{align}
		q_u \sim {\rm Beta}(\alpha, \beta)
	\end{align}\
$$

また$q_u$の期待値は

$$
	\begin{align}
		\double E[q_u] = \frac{\alpha}{\alpha+\beta}
	\end{align}\
$$

となります。

たとえば単語列$\ The\ quick\ brown\ fox\ jumps\ over\ the\ lazy\ dog\ $があり、$dog$を文脈木に追加する場合を考えます。

深さnのレストランで停止する確率を$q_{n}$とすると、文脈$h$のもとで客$dog$の停止深さが$l$となる確率は

$$
	\begin{align}
		P(n=l\mid h) = q_l\prod_{i=0}^{l-1}(1-q_i)
	\end{align}\
$$

となるので、客$dog$がルートのレストラン$\epsilon$に追加される確率は$q_{\epsilon}$、レストラン$lazy$に追加される確率は$(1-q_{\epsilon})q_{lazy}$、レストラン$the$に追加される確率は$(1-q_{\epsilon})(1-q_{lazy})q_{the}$、・・・のように計算されます。

![VPYLM](/images/post/2016-07-28/vpylm_stop_probs.png)

ちなみに1から引いているのはそこを通過する確率を計算するためです。

通常は深いノードに行くほど停止確率が小さくなりますが、文脈に沿った経路のそれぞれの$q_n$が小さければ深いn-gramとなり、HPYLMとは違って様々な深さのノードを許すモデルになっています。

## 可変長ベイズn-gram言語モデル

VPYLMでは、単語列$\boldsymbol w=w_1w_2...w_T$の確率を

$$
	\begin{align}
		P(\boldsymbol w)=\sum_{\boldsymbol n}\sum_{\boldsymbol \Theta}P(\boldsymbol w, \boldsymbol n, \boldsymbol \Theta)
	\end{align}\
$$

と表します。

$\boldsymbol \Theta$は文脈木の代理客を含めたすべての客の配置を表す隠れ変数、$\boldsymbol n=n_1n_2...n_T$は$\boldsymbol w$のそれぞれの単語が生成された隠れたn-gram長を表します。

HPYLMと同様に客の配置$\boldsymbol \Theta$は推定すべきパラメータであり、VPYLMでは$\boldsymbol \Theta$と$\boldsymbol n$の両方をギブスサンプリングによって推定します。

その際、単語列$\boldsymbol w$の位置$t$の単語$w_t$の隠れたn-gramオーダー$n_t$を、

$$
	\begin{align}
		n_t \sim P(n_t\mid \boldsymbol w, \boldsymbol n_{-t}, \boldsymbol \Theta_{-t})
	\end{align}\
$$

のようにギブスサンプリングする必要があるのですが、実際は$P(n_t\mid \boldsymbol w, \boldsymbol n_{-t}, \boldsymbol \Theta_{-t})$から直接サンプリングすることはできません。

そこでこの式をベイズの定理から

$$
	\begin{align}
		P(n_t\mid \boldsymbol w, \boldsymbol n_{-t}, \boldsymbol \Theta_{-t}) \propto  
		P(w_t\mid \boldsymbol w_{-t}, \boldsymbol n, \boldsymbol \Theta_{-t})P(n_t\mid\boldsymbol w_{-t}, \boldsymbol n_{-t}, \boldsymbol \Theta_{-t})
	\end{align}\
$$

と変形します。（$\propto$は比例を意味します）

$\boldsymbol n_{-t}$は、単語$w_t$を除いた単語列$\boldsymbol w_{-t} = w_1,w_2,...,w_{t-1},w_{t+1},...,w_T$に対応するn-gramオーダー$n_1,n_2,...,n_{t-1},n_{t+1},...,n_T$を表しています。

### $n_t$のサンプリング

ここでの$\boldsymbol n_{-t}$はただの整数列であり、この値がそのままモデルのパラメータにあるわけではありません。

そして$n_t$をギブスサンプリングするためには$\boldsymbol n_{-t} = n_1,n_2,...,n_{t-1},n_{t+1},...,n_T$をモデルの何らかのパラメータとして持っている必要があります。

そこでVPYLMでは、文脈木の各レストラン$u$で、全ての単語（客）の通過回数$a_u$と停止回数$b_u$を記録します。

そして式(3)の停止確率$q_u$の期待値をベータ事後分布の期待値として

$$
	\begin{align}
		\double E[q_u]=\frac{a_u+\alpha}{a_u+b_u+\alpha+\beta}
	\end{align}\
$$

と推定します。

こうすることで式(7)の右辺の第二項にある$\boldsymbol n_{-t}$の条件付き確率を

$$
	\begin{align}
		P(n_t=l\mid\boldsymbol w_{-t}, \boldsymbol n_{-t}, \boldsymbol \Theta_{-t})=\frac{a_l+\alpha}{a_l+b_l+\alpha+\beta}\prod_{i=0}^{l-1}\frac{b_i+\beta}{a_i+b_i+\alpha+\beta}
	\end{align}\
$$

のように表すことができ、$n_t$を$\boldsymbol n_{-t}$の条件付き確率からギブスサンプリングすることができるようになります。

また式(7)の右辺の第一項はn-gramオーダーが$n_t$と決まった時の$w_t$のn-gram確率で、式(1)により計算します。

### $\boldsymbol \Theta$のサンプリング

$\boldsymbol \Theta_{-t}$は単語$w_t$をレストランから除外したあとの客の配置を表します。

これはHPYLMと同様${\rm RemoveCustomer}$をすれば得られます。

その状態で式(7)により新たなn-gramオーダー$n_t$をサンプリングし、その深さ$n_t$に単語$w_t$を追加し直すことで$\boldsymbol \Theta$が自動的にギブスサンプリングされます。

（$\boldsymbol \Theta$は、代理客を含めた客の配置が決定した時点でサンプリングされたことになります。）


## 実装

C++での実装を[GitHub](https://github.com/musyoku/vpylm)にあげておきました。

ハイパーパラメータのサンプリングの実装に関しては[HPYLM](/2016/07/26/A_Hierarchical_Bayesian_Language_Model_based_on_Pitman-Yor_Processes/)の記事に注意点などを書いています。

### $n_t$のサンプリング

$w_t$に対応する$n_t$をサンプリングするには、深さ$l=0,1,2,...$の場合それぞれについて以下の値を求めます。

$$
	\begin{align}
		P(w_t\mid \boldsymbol w_{-t}, \boldsymbol n, \boldsymbol s_{-t})P(n_t=l\mid\boldsymbol w_{-t}, \boldsymbol n_{-t}, \boldsymbol \Theta_{-t})
	\end{align}\
$$

第一項は式(1)から求めますが、この時深さは$l$と決まっているため、オーダー$l$の文脈$h_l$に対応するレストランを用いて以下のように計算します。

$$
	\begin{align}
		P(w_t\mid h_l)=\frac{c(w_t\mid h_l) - d_{\mid h_l \mid}\cdot t_{h_{l}w_{t}}}{\theta_{\mid h_l \mid} + c(h_l)}+\frac{\theta_{\mid h_l \mid} + d_{\mid h_l \mid}\cdot t_{h_l}}{\theta_{\mid h_l \mid} + c(h_l)}P(w_t\mid h_l')
	\end{align}\
$$

[A Bayesian Interpretation of Interpolated Kneser-Ney](http://www.gatsby.ucl.ac.uk/~ywteh/research/compling/hpylm.pdf)の表記に従えば以下のようにも表されます。

$$
	\begin{align}
		P(w_t\mid h_l)=\frac{
			c_{h_{l}w_{t}\cdot} - d_{\mid h_l \mid}\cdot t_{h_{l}w_{t}}
		}{
			\theta_{\mid h_l \mid} + c_{h_{l}\cdot\cdot}
		}+\frac{
			\theta_{\mid h_l \mid} + d_{\mid h_l \mid}\cdot t_{h_l\cdot}
		}{
			\theta_{\mid h_l \mid} + c_{h_l\cdot\cdot}
		}P(w_t\mid \pi(h_l))
	\end{align}\
$$

式(10)の第二項$P(n_t=l\mid\boldsymbol w_{-t}, \boldsymbol n_{-t}, \boldsymbol \Theta_{-t})$は式(9)から求めますが、この時第二項が十分小さくなれば、それ以降の深さ$l$は見ずに計算を打ち切ります。

また式(9)はレストランが存在しない深さ（つまり通過回数$a_u$や停止回数$b_u$が存在しない）も考慮しなければならないのですが、私はレストランが無い部分では通過回数と停止回数は$0$とし、式(3)を用いて計算しました。

## 実験

英語版Wikipediaからランダムに取ってきた5万文に対し単語n-gramモデルを学習させました。

学習されたVPYLMから文章を生成させると以下のようになりました。

```
it was first published in ## . 
of their corresponding last digits . 
she has a number of notable uses . 
these were very influential constitutional conservatives , and take a liking to the first . 
otherwise it is revealed to be a part of the municipality of science fiction edited by kathryn cramer , david g . 
jackson , ms . 
the ## census it had a population of ## people . 
today , there is no longer be burned the u . 
and a cafeteria and hope that children including the major national morning of october rust to stop the german people , and the university of oxford , east of the county seat of the district of columbia . 
he served as a member of the ## census , the cdp population was ## , ## at the ## census . 
this film released in october ## . 
wong tai sin estate and made the finals . 
## , ## , and ## . 
it is currently used mostly for football matches . 
as a result of the ## census , the cdp population was ## , ## at the ## census . 
there are some of the most successful on the first day of his death in ## . 
it is part of the battle creek , michigan metropolitan statistical analysis is used as a new member , and is located in the province of venice , located in the southeastern ohio and was admitted to the bar in ## and is currently one of the biggest tributary of great antiquity here , the two organizations . 
he left his dream of having a flat square television , other changes , and while doing so , at the battle of britain , and placing him on stage , a high school in dayton , ohio , united states . 
in ## , the northern ridge , in which the teams have to draw was once a year later . 
during the night of october ## , ## , he joined the united states army air forces retained the office of prime minister nawaz sharif , the political and social services in the program . 
however , some people would consider the book of the summer olympics . 
ultimately , the film failed in the process of building its administration and teaching in france , where the teams are shown a sporting grounds . 
those who fell in the battle of britain , the original ## , ## people . 
three of the four grand slam singles titles at the ## census the population was ## , ## at the ## census . 
later , the mother of jesus christ , and even a small port of the colorado kid , who has been criticized for its annual strategy dying from starvation and civil strife constantly throughout the game , the mayor of brisbane from , and the two became a professor at his house . 
adams , who are in a white father , who was impressed with all of them , however , that the election was held in london , on their own right , but in ## the company is headquartered in santa fe county . 
the young people who do not share a similar way as the beginning of the game . 
the supreme court of ohio , united states . 
in the summer of ## , he was granted full of the new zealand house of representatives in ## and ## , he was chosen by king george iii . 
though the sultan s harem with the help of the election was held in london , england . 
, but was defeated after the war , moore never forget about the course of their own right , but eventually the other candidates in the uk . 
and in the u . 
from the university of michigan and was taken to prison island . 
the first episode of season . 
the result was that the actual number of the spratly islands , including last year they performed by the bolsheviks but later added during the second world war , he began his baseball and hockey with many of the members of the royal institution . 
he took over the project was completed in ## and ## . 
each year . 
the english and low liquidity . 
discourse , and is located in the province of venice , located in the southeastern asia is the only high school in ## . 
in ## , the northern territory , canada . 
according to the united states census bureau , the cdp has a total area of ## . 
the district is bounded by the western part of the park . 
the name of a station on the keisei main person in a rematch of the top ## of the ## grand slam singles and mixed doubles titles , and ## grand slam singles had to change their old money from his long railway line and expanded the scope of the duty of care which looks as if he had made the decision to lower the international treaties giving it a few days , in the united states . 
in ## , the northern terminus was moved to their traditions , but can be used to generate an image . 
the entirety . 
when it was still very young and took them to get out of the race , and the school was established by president ronald reagan sent on ## december ## , it had a population of ## people and the police and the local community . 
the result was that the general public . 
however , when he was ## years old . 
in ## , he published his first appearance of the internet . 
during the summer of ## . 
in the city the population was spread out with ## . 
on ## december ## , it had a population of ## people and the police and the local community . 
, with some common characteristics by their parents felt the need for a living electrical engineering programs have been built by the great belt bridge , the longest and most of his cabinet of prime minister also has the highest honour given by a few months later , it is not necessary for them on the bottom of the valley of the river thames , and can be used to generate false . 
the act of union was initially a colonel miller to open up the rifle and follows them to a better understanding of what is known about the cultural markers of italian opera . 
the socreds . 
whether a duty of care on their own right , but he did not have any of his art gallery in new york city , new york and now lives in the church and left the country . 
in the city the population was spread out with ## . 
it should be noted that neither does it had a population of ## people . 
due to the fact that they are not directly to the pressure hull , which actually is the same as the german people , and thus has a long tradition of a group of nuclear power stations and was elected mayor in the room . 
faraday was the leader of the opposition of the 13th century and early 20th century fox studios in japan . 
these were very difficult to find out a young man , he worked for a time , and the trio of terror . 
and so the north shore . 
```


## 関連

- [A Hierarchical Bayesian Language Model based on Pitman-Yor Processes](/2016/07/26/A_Hierarchical_Bayesian_Language_Model_based_on_Pitman-Yor_Processes/)
- [ベイズ階層言語モデルによる教師なし形態素解析](/2016/12/14/%E3%83%99%E3%82%A4%E3%82%BA%E9%9A%8E%E5%B1%A4%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AB%E3%82%88%E3%82%8B%E6%95%99%E5%B8%AB%E3%81%AA%E3%81%97%E5%BD%A2%E6%85%8B%E7%B4%A0%E8%A7%A3%E6%9E%90/)
- [Pitman-Yor言語モデルのハイパーパラメータの推定に関して](/2016/10/16/Pitman-Yor%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AE%E3%83%8F%E3%82%A4%E3%83%91%E3%83%BC%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E3%81%AE%E6%8E%A8%E5%AE%9A/)
