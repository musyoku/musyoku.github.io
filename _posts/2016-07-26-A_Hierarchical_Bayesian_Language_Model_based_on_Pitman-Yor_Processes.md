---
layout: post
title: A Hierarchical Bayesian Language Model based on Pitman-Yor Processes (HPYLM)
category: Chainer
tags:
- HPYLM
excerpt_separator: <!--more-->
---

## 概要

- [A Hierarchical Bayesian Language Model based on Pitman-Yor Processes](http://www.gatsby.ucl.ac.uk/~ywteh/research/compling/acl2006.pdf) を読んだ
- [A Bayesian Interpretation of Interpolated Kneser-Ney](http://www.gatsby.ucl.ac.uk/~ywteh/research/compling/hpylm.pdf) を読んだ
- C++でHPYLMを実装した

<!--more-->

## はじめに

HPYLMはPitan-Yor過程によるスムージングを行うベイズ階層n-gram言語モデルの一種です。

後で記事にしますが可変長ベイズ階層n-gram言語モデルであるVPYLMとは違いHPYLMはn-gramのオーダーを固定します。

## スムージングとHPYLM

テキストデータが以下の3文とします。

```
she will sing
she will like
he will call
```

この時、たとえば単語列she willに続いてlikeが来る確率$P(like\mid she\ will)$は、she willで始まる文が2つあり、そのうちの1つがshe willに続いてlikeが来ているので、$1$割る$2$で$0.5$となります。

しかしhe willにlikeが続くデータはないため、$P(like\mid he\ will)=0$となります。

このようにデータに出てこないものは全て確率が$0$となってしまうのですが、スムージングと呼ばれる方法を用いると$0$ではない適切な確率を計算できるようになります。

ここでは3-gramなモデルで説明を行います。

つまり、ある単語が生成される確率は、以下のように後ろの2単語のみで決まると仮定したモデルです。

$$
	\begin{align}
		P(dog\mid The\ quick\ brown\ fox\ jumps\ over\ the\ lazy) &= P(dog\mid the\ lazy)\nonumber
	\end{align}\
$$

HPYLMは以下の様な文脈木を考え、単語のことを客、木のノードをレストランと呼びます。

![HPYLM](/images/post/2016-07-27/hpylm.png)

HPYLMではこの文脈木を用いて単語の数をカウントします。

たとえばshe willという単語列の後にlikeという単語が何回来たかをカウントしたい場合、文脈木のルートからwill→sheとレストランをたどり、sheというレストランにlikeという客を追加します。

こうすることでshe willに続いてlikeが1回来たとカウントされます。

同様に上記のデータにあるsingとcallもレストランに追加します（図の黒色の客）

この状態では先程と同じく、he willに続いてlikeが来る回数はheのノードにlikeという客がいないため$0$となります。

そこで代理客（図の白色の客）を親のレストランに送ります。

そうするとhe willの後にlikeが続く回数は依然$0$ですが、willに続いてlikeが来る回数が$1$になります。

よって$P(like\mid he\ will)$を求めるときに、ノード$he$が持つ3-gram確率$P(like\mid he\ will)$（これは0）と、その親ノード$will$が持つ2-gram確率$P(like\mid will)$をうまく補完すれば$0$ではない値にすることが可能になります。

これがスムージングの考え方で、HPYLMではPitman-Yor過程と呼ばれる確率過程を用いて補完しています。

## 学習

HPYLMにおけるパラメータは文脈木内の代理客を含めた全ての客の配置です。

全ての客の配置を$\boldsymbol\Theta$とすると、文脈$\boldsymbol u$に単語$w$が続く確率$P(w\mid\boldsymbol u)$は

$$
	\begin{align}
		P(w\mid\boldsymbol u)=P(w\mid\boldsymbol u, \boldsymbol\Theta)
	\end{align}\
$$

と表され、n-gram確率が$\boldsymbol\Theta$によって決まります。

我々の目標は真の$\boldsymbol\Theta$を推定することなので、HPYLMではギブスサンプリングを用いて推定します。

### $\boldsymbol\Theta$のサンプリング

まず${\rm RemoveCustomer(\boldsymbol u,w)}$によりレストラン$\boldsymbol u$から$w$を削除します。

削除された後の残りの客全ての配置を$\lnot\boldsymbol\Theta$とし、$\lnot\boldsymbol\Theta$のもとで$w$の配置を再サンプリングします（${\rm AddCustomer}(\boldsymbol u, w)$）。

こうすると新たな配置$\boldsymbol\Theta^{new}$がギブスサンプリングされたことになります。

以上の操作をランダムに選んだ訓練データを使って繰り返し行うことで$\boldsymbol\Theta$を更新していきます。


## 実装

C++での実装を[GitHub](https://github.com/musyoku/hpylm)に上げておきました。

ここからはHPYLMで単語2-gram言語モデルを学習させる前提で実装について説明します。

また用いる記号については論文に合わせて

- $w$
	- 単語
- $\boldsymbol u$
	- 単語$w$の左側にあるすべての単語列（文脈）
- $\pi(\boldsymbol u)$
	- 文脈$\boldsymbol u$のオーダーを1つ下げた文脈
- $\mid\boldsymbol u\mid$
	- 文脈$\boldsymbol u$に含まれる単語数
- $c_{\boldsymbol uwk}$
	- レストラン$\boldsymbol u$で単語$w$を提供しているいくつかのテーブルのうち、$k$番目のテーブルにいる客数
- $c_{\boldsymbol uw\cdot}$
	- レストラン$\boldsymbol u$で単語$w$を提供しているすべてのテーブルの客の総数
- $c_{\boldsymbol u\cdot\cdot}$
	- レストラン$\boldsymbol u$にいる客の総数
- $t_{\boldsymbol uw}$
	- レストラン$\boldsymbol u$で単語$w$を提供しているテーブルの総数
- $t_{\boldsymbol u\cdot}$
	- レストラン$\boldsymbol u$のテーブルの総数
- $d_{\mid\boldsymbol u\mid}$
	- Pitman-Yor過程のハイパーパラメータ
	- レストランごとではなく深さごとに共通の値を使う
- $\theta_{\mid\boldsymbol u\mid}$
	- Pitman-Yor過程のハイパーパラメータ
	- レストランごとではなく深さごとに共通の値を使う
- $G_0(w)$
	- 基底測度（単語0-gram確率）
	- 語彙数の逆数をパラメータに持つ一様分布とする

とします。

たとえば文が$she\ will\ sing$で$w$が$sing$である場合、

$$
	\begin{align}
		w &= sing\nonumber\\
		\boldsymbol u &= she\ will\nonumber\\
		\pi(\boldsymbol u) &= will\nonumber\\
		\mid\boldsymbol u\mid &= 2\nonumber\\
	\end{align}\
$$

となります。

### WordProbability($\boldsymbol u$, $w$)

文脈$\boldsymbol u$の後に$w$が続く確率は

$$
	\begin{align}
		{\rm WordProbability}(\boldsymbol u, w)&=\nonumber\\
		P(w\mid \boldsymbol u, \boldsymbol\Theta)&=\frac{c_{\boldsymbol uw\cdot} - d_{\mid\boldsymbol u\mid}t_{\boldsymbol uw}}{\theta_{\mid\boldsymbol u\mid}+c_{\boldsymbol u\cdot\cdot}}
		+\frac{\theta_{\mid\boldsymbol u\mid}+d_{\mid\boldsymbol u\mid}t_{\boldsymbol u\cdot}}{\theta_{\mid\boldsymbol u\mid}+c_{\boldsymbol u\cdot\cdot}}{\rm WordProbability}(\pi(\boldsymbol u), w)\nonumber
	\end{align}\
$$

となり、再帰的に文脈のオーダーを落として計算します。

### AddCustomer($\boldsymbol u$, $w$)

文脈$\boldsymbol u$のもとで単語$w$が観測された時、深さ1に対応するノード（HPYLMでは常にn-1の深さのノードに追加する）に$w$を追加します。

追加する際は、

- $max(0, c_{\boldsymbol uwk} - d_{\mid\boldsymbol u\mid}t_{\boldsymbol uw})$に比例する確率で、$w$を提供しているすべてのテーブルの中の$k$番目のテーブルに追加
- $(\theta_{\boldsymbol u} + d_{\boldsymbol u}t_{\boldsymbol u\cdot}){\rm WordProbability}(\pi(\boldsymbol u), w)$に比例する確率で、$w$を提供する新しいテーブルを作成しそこに追加
	- この時親レストラン$\pi(\boldsymbol u)$に対し${\rm AddCustomer}(\pi(\boldsymbol u), w)$
	- したがって親レストランでも同様に、新たなテーブルができればさらにその親に対して${\rm AddCustomer}(\pi(\pi(\boldsymbol u)), w)$

のように確率的に客をテーブルに追加します。

### RemoveCustomer($\boldsymbol u$, $w$)

客を削除する際は$c_{\boldsymbol uwk}$に比例する確率で、$w$を提供しているすべてのテーブルの$k$番目のテーブルから客を削除します。

$k$番目のテーブルに客が一人もいなくなった場合はそのテーブルを削除し、親レストラン$\pi(\boldsymbol u)$に対し${\rm RemoveCustomer}(\pi(\boldsymbol u), w)$を実行します。

### ハイパーパラメータの推定

$\theta_{\mid\boldsymbol u\mid}$や$d_{\mid\boldsymbol u\mid}$は初めに適当な初期値を与えておきますが、現在の文脈木のパラメータからサンプリングにより推定します。

[論文](http://www.gatsby.ucl.ac.uk/~ywteh/research/compling/hpylm.pdf)のAppendix Cに詳細が書いてありますが、まだ理解が追いついていないのでやり方だけ書いておきます。

まず以下の3つの補助変数を定義します。

$$
	\begin{align}
		x_{\boldsymbol u} &\sim {\rm Beta}(\theta_{\mid\boldsymbol u\mid}+1, c_{\boldsymbol u\cdot\cdot}-1)\\
		y_{\boldsymbol ui} &\sim {\rm Bernoulli}\left(\frac{\theta_{\mid\boldsymbol u\mid}}{\theta_{\mid\boldsymbol u\mid} + d_{\mid\boldsymbol u\mid}i}\right)\\
		z_{\boldsymbol uwkj} &\sim {\rm Bernoulli}\left(\frac{j-1}{j-d_{\mid\boldsymbol u\mid}}\right)
	\end{align}\
$$

これらの補助変数を用いてベータ分布・ガンマ分布からサンプリングします。

$$
	\begin{align}
		d_m &\sim {\rm Beta}\left(
		a_m + \sum_{\boldsymbol u:\mid u \mid=m,t_{\boldsymbol u\cdot}\geq2}\sum_{i=1}^{t_{\boldsymbol u\cdot - 1}}(1 - y_{\boldsymbol ui}),
		b_m + \sum_{\boldsymbol u,w,k:\mid u \mid=m,c_{\boldsymbol uwk}\geq2}\sum_{j=1}^{c_{\boldsymbol uwk - 1}}(1 - z_{\boldsymbol uwkj})
		\right)\\
		\theta_m &\sim {\rm Gamma}\left(
		\alpha_m + \sum_{\boldsymbol u:\mid u \mid=m,t_{\boldsymbol u\cdot}\geq2}\sum_{i=1}^{t_{\boldsymbol u\cdot - 1}}y_{\boldsymbol ui},
		\beta_m + \sum_{\boldsymbol u:\mid u \mid=m,t_{\boldsymbol u\cdot}\geq2}{\rm log}x_{\boldsymbol u}
		\right)\\
	\end{align}\
$$

$\sum_{\boldsymbol u:\mid u \mid=m,t_{\boldsymbol u\cdot}\geq2}$は深さ$m$であり、かつ$t_{\boldsymbol u\cdot}\geq2$となるような全てのレストランに対する総和です。

その他の$\sum$についても同様に考えます。

新たなハイパーパラメータ$a_m, b_m, \alpha_m, \beta_m$が出てきますが、これは適当な値を設定して固定します。

またガンマ分布からのサンプリングの実装には注意が必要です。

[英語版のウィキペディア](https://en.wikipedia.org/wiki/Gamma_distribution)に書いてありますが、ガンマ分布は${\rm Gamma}(k, \theta)$と${\rm Gamma}(\alpha, \beta)$の2種類の表記があり、計算方法が違います。

C++の[gamma_distribution](http://www.cplusplus.com/reference/random/gamma_distribution/)関数は${\rm Gamma}(k, \theta)$の方で実装されているため、この関数を用いる場合は式(5)の２つ目のパラメータ$$\beta_m + \sum_{\boldsymbol u:\mid u \mid=m,t_{\boldsymbol u\cdot}\geq2}{\rm log}x_{\boldsymbol u}$$の逆数をgamma_distributionの2つ目の引数とします。

初めはこの事に気づかず実装したため$\theta_m$の値が数千〜数万という巨大な値になりました。

正しく実装すると小さな実数になります。

## 実験

英語版Wikipediaからランダムに取ってきた5万文に対し、単語3-gramを学習させました。

学習後のHPYLMを使って文章生成を行わせてみた結果が以下になります。

```
the country s leading school of the royal geological society of america , and on the site of the seat of a parcel of land to the election campaign of the union . 
london borough of the series of all the fans of the camp was established in ## , ## . 
theoretically supports all games , but has been an important source of the power of the old man , and moved to the island with scrooge following behind . 
the white house chief of the year . 
of the band had a very large contingent of the waterfall model . 
to the hamlet of grangemill . 
amendments entered into the microphone to the united states , but the lead and prost took the solemn vows of obedience , win souls for her contribution to american performing groups raagapella and global rhythms , to take . 
, the organization has attempted to assassinate king george iii , and after a few missions to investigate alleged frauds in the future when fica contributions will be put forward the completed incidents via paper forms , computer and video games . 
was done , and the role of handmaiden is also a member of the heinemann writing in the morning of his death , the bank finally failed . 
the most in ## as a visiting professor at the time of the st . 
their only loss for the middle school , students will attend streamwood high school was originally part of the parish church near the end of the performance was playing bass guitar in the cdp has a total area of ## . 
word comes from the government and the county the population was spread out with ## . 
a football club , he quotes the ancient language of economic research . 
returned to france , with the truth is the head of house , the remaining ## . 
with many of the 3rd class of ## , ## , she was built in a textile design shop . 
, he won a scholarship to study an internal wastegate port and turbine housing . 
at the age of ## , a powerful weapon that had used the water warms . 
key to the world war i , he was eventually taken over by the end of the northern frontier , the regiment simply had a . 
died of a part of the census2 of ## , ## households , and their shadow allies . 
composed his mass in the music department at northwestern memorial hospital . 
and composed what are often expressed through different means . 
day was a party of canada . 
is the most obvious of these , preferring a few people believed that he had risen again and the upper imbrian epoch . 
final words to the second largest population of the oka bay and the rest of the u . 
he were to get to the study of the original playground is still alive . 
was claimed by the end of the united states to offer women training as aircraft mechanics . 
city has a shallow bowl , as well . 
was switched to disco and smooth jazz gained popularity . 
lies just outside the reported range . 
has been some attempts to get catalina to return to dancing . 
is located in the film industry . 
measure was enacted by the english . 
also known as a ## . 
were ## housing units at an average density of ## . 
then orders sundown and mainframe to keep the proletarians out or to be published in ## . 
named for a new version of the largest hammering man is by far the most significant influence came from that of portugal . 
inhabited by the new york , new york city . 
of the full moon . 
members of the kingdom of naples , but the most of his whereabouts , jing ke with it . 
has remained silent for almost half the legislation h . 
value of a day to the ## london , and gives him a new one is the southern portion of the 19th century , in october ## , ## to ## . 
course , and working at the base , a ritual on transforming oneself into a pocket of the new zealand government support her in a similar execution . 
cards for the second half of the movement . 
arizona feeling disenfranchised . 
postmodern artist michael jackson , and orissa in ## , ## , ## and over , there were ## . 
quebec city and the next five years . 
constituency was created a new suit does actually turn in the greater los angeles , and for all the way . 
## inmates , and the group . 
can be heard at all , and was named by the river severn floods in a french winery , for the most important contribution of the world . 
addition , the two . 
report also claimed sovereignty over the city has a total of ## to ## as of the liberals ran a total area of ## , ## meter long catwalk in the movie the vikings . 
, an advisory role to play the game . 
the methodist church and its derivatives . 
song , or for the use of the north side , the state in eastern orthodox theology . 
day before the first , the lower house of commons , to a form similar to the united states and the narn regime in the direction of the sixties revolution . 
became a nazi soldier chained to a severe blow to the next thunderstorm , which is not expressions themselves but what people use computers and websites . 
am tempted to make an impact on the new leader of the fill . 
tribute , though the book was added . 
until the late 1980s and early 20th century . 
ones who discovered zinedine zidane . 
middle school , and monte cook . 
, with the city had conquered the mon capital of the canadian forces , which is often taken to the 1990s . 
was promoted to the united states . 
they could be less influenced by 20th century . 
is endemic to north america and the liverpool philharmonic orchestra and the other regions of the area . 
, who regarded mankind as being the most common license sought by the san salvador cathedral was completed in ## he has been forced out of which is said to be overrepresented statistically in virtually every society . 
at the sharks and large profits associated with an e . 
some secretarial or notarial capacity . 
in a unique people with a set of all the dead bodies . 
course , and generally melancholy reflections on human development to justify intellectual property rights in the early 1980s , the main camp and a small section of the commission for the . 
was heavily involved in a bid to protect her from his children and adults are approximately ## in new york . 
was the main source of the 19th century suspension bridge , the european taper is conical and widens towards the end of the american society . 
was a member of the golden girdle of gaea that antiope gave to him . 
junction is a member of the library , an ancient artifact . 
was supposed to be a success , clear weakening of ottoman rule , the black market , and his wife at that time . 
slaty antwren , myrmotherula schisticolor , is a total area of the orbits which would become synonymous with judean and if the future in a game . 
moved to federal drug charges , and , in person , aphrodite really is a term used to be a successful prosecutor until others learned that he arranged for the ## census , the main language of instruction in the club at the base remained the tallest building in the 1980s and 1990s . 
the ## provincial election , when he was also nominated for the masses and were joined by the tigers and it was later to be an angry mob , and ran a full member of the cayman islands have no effect on the national register of historic places . 
of the united states senate in the san gabriel valley . 
will be a bad thing that the financial statements to the union of families located in the atlantic coast of africa to refer to the earth is not a candidate to evaluate the remainder to withdraw . 
```

## 関連

- [Pitman-Yor過程に基づく可変長n-gram言語モデル](/2016/07/28/Pitman-Yor%E9%81%8E%E7%A8%8B%E3%81%AB%E5%9F%BA%E3%81%A5%E3%81%8F%E5%8F%AF%E5%A4%89%E9%95%B7n-gram%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%87%E3%83%AB/)
- [ベイズ階層言語モデルによる教師なし形態素解析](/2016/12/14/%E3%83%99%E3%82%A4%E3%82%BA%E9%9A%8E%E5%B1%A4%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AB%E3%82%88%E3%82%8B%E6%95%99%E5%B8%AB%E3%81%AA%E3%81%97%E5%BD%A2%E6%85%8B%E7%B4%A0%E8%A7%A3%E6%9E%90/)
- [Pitman-Yor言語モデルのハイパーパラメータの推定に関して](/2016/10/16/Pitman-Yor%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AE%E3%83%8F%E3%82%A4%E3%83%91%E3%83%BC%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E3%81%AE%E6%8E%A8%E5%AE%9A/)

## おわりに

初めてDeepではない自然言語処理をやったので、書いたコードが正しいのかどうかわかりません。

