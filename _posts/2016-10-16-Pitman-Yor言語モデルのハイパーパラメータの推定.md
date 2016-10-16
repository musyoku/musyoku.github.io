---
layout: post
title: Pitman-Yor言語モデルのハイパーパラメータの推定に関して
category: 論文
tags:
- HPYLM
- VPYLM
excerpt_separator: <!--more-->
---

## 概要

- Pitman-Yor言語モデルのハイパーパラメータの推定における更新式の詳細な導出について

## はじめに

最近Deep LearningのRNN言語モデル（というかLSTM）が流行っていますが、私は[教師なし形態素解析](http://chasen.org/~daiti-m/paper/nl190segment.pdf)などにも応用できるベイズ階層言語モデルに注目し、過去に[階層Pitman-Yor言語モデル（HPYLM）](/2016/07/26/A_Hierarchical_Bayesian_Language_Model_based_on_Pitman-Yor_Processes/)や[可変長n-gram言語モデル（VPYLM）](/2016/07/28/Pitman-Yor%E9%81%8E%E7%A8%8B%E3%81%AB%E5%9F%BA%E3%81%A5%E3%81%8F%E5%8F%AF%E5%A4%89%E9%95%B7n-gram%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%87%E3%83%AB/)を実装してきました。（教師なし形態素解析も実装済みですので今後記事を書く予定です）

これらの実装において２つのハイパーパラメータをデータから推定する必要があり、この部分は[Teh先生の論文](http://www.gatsby.ucl.ac.uk/~ywteh/research/compling/hpylm.pdf)に載っている更新式を使うのが慣例となっていますが、この式は何の説明もなく出てくるため、一体どのようにして導出されたのかをこの記事でまとめます。

ここからの説明はすべて[論文](http://www.gatsby.ucl.ac.uk/~ywteh/research/compling/hpylm.pdf)をもとに行います。

また、この記事で説明されていない記号・変数はすべて論文に説明があります。

## 客の配置の確率について

論文では$seating\ arrangement$となっている、代理客も含めたすべての客の配置をこの記事では$\boldsymbol \Theta$と表すことにします。

この$\boldsymbol \Theta$の確率は論文の(23)式で表されますが、再掲しておきます。

$$
	\begin{align}
		p(\boldsymbol \Theta) &= \prod_w G_0(w)^{c_0w\cdot}\prod_{\boldsymbol u} 
			\frac{
				[\theta_{\mid \boldsymbol u \mid}]^{(t_{\boldsymbol u \cdot})}_{d_{\mid \boldsymbol u \mid}}
			}{
				[\theta_{\mid \boldsymbol u \mid}]^{(c_{\boldsymbol u \cdot\cdot})}_{1}
			}
			\prod_w \prod_{k=1}^{t_{\boldsymbol u \cdot}}
			[1-d_{\mid \boldsymbol u \mid}]^{(c_{\boldsymbol uwk - 1})}_{1}\\
		[a]^{(0)}_b &= [a]^{-1}_b = 1\\
		[a]^{(c)}_b &= a(a+b)\cdot\cdot\cdot(a+(c-1)b) = \frac{b^c\Gamma(a/b+c)}{\Gamma(a/b)}
	\end{align}\
$$

この式も説明なく出てくるので、ここではまずこの式の導出のやり方を説明します。

まず論文(10)式の、次に観測される客$x_{c\cdot+1}$の確率

$$
	\begin{align}
		p(x_{c\cdot+1} \mid x_{1}, x_{2}, ..., x_{c\cdot}, \boldsymbol \Theta) = 
			\sum_{k=1}^{t_{\cdot}}
			\frac {c_k-d}{\theta+c_{\cdot}}\delta_{\phi_k}
			+\frac {\theta+dt_{\cdot}}{\theta+c_{\cdot}}G_0(w)\nonumber
	\end{align}\
$$

から、次に観測される客が座るテーブル$k_{c\cdot+1}$がテーブル$K$になる確率を求めておくと、

$$
	\begin{align}
		p(k_{c\cdot+1} = K \mid x_{1}, x_{2}, ..., x_{c\cdot}, \boldsymbol \Theta) = 
			\frac {c_K-d}{\theta+c_{\cdot}}\delta_{\phi_K}
			+\frac {\theta+dt_{\cdot}}{\theta+c_{\cdot}}G_0(w)
	\end{align}\
$$

となります。

ここで、$\delta_{\phi_k}$は質量$1$の点（point mass）ですので、ただの数字の$1$とみてかまいません。

また$c_k$は$k$番目のテーブルの客数、$c_{\cdot}$は総客数です。

ちなみに論文の式(11)は、

$$
	\begin{align}
		p(x_{c\cdot+1}=w \mid \boldsymbol \Theta) = 
			\frac {c_w-dt_w}{\theta+c_{\cdot}}
			+\frac {\theta+dt_{\cdot}}{\theta+c_{\cdot}}G_0(w)\nonumber
	\end{align}\
$$

次に観測される単語$x_{c\cdot+1}$が$w$である確率であって、どのテーブルかは考慮されないため注意が必要です。

（式(4)で$w$を提供している全てのテーブルについて足し合わせるとこの式が得られます）

まず中華料理店過程（CRP）で、基底分布$G_0$から単語$w_1$が生成された状態を考えます。

![CRP](/images/post/2016-10-16/p_seating_arrangement_1.png)

（CRPではテーブルがないときは必ず$G_0$から生成されます）

この時の$p(\boldsymbol \Theta)$は

$$
	\begin{align}
		p(\boldsymbol \Theta) = G_0(w_1)
	\end{align}\
$$

となります。

次にこの状態から、経験分布から$w_1$が生成された状態を考えると、

![CRP](/images/post/2016-10-16/p_seating_arrangement_2.png)

$p(\boldsymbol \Theta)$は

$$
	\begin{align}
		p(\boldsymbol \Theta) = G_0(w_1)\frac{1-d}{\theta+1}
	\end{align}\
$$

となります。

これは式(4)の第1項で総客数$c_{\cdot}=1$、単語$w_1$を提供しているテーブル数$t_{w_1}=1$とすることで得られる確率を式(5)に掛けたものになっています。

（CRPでは単語は、$G_0$と、現在の客の配置からなる経験分布の混合分布から生成されます。経験分布から生成されたときは式(4)の第1項を考えます。）

次に$G_0$から$w_2$が生成された場合を考えます。

![CRP](/images/post/2016-10-16/p_seating_arrangement_3.png)

この時の$p(\boldsymbol \Theta)$は

$$
	\begin{align}
		p(\boldsymbol \Theta) = G_0(w_1)G_0(w_2)\frac{1-d}{\theta+1}\frac{\theta+d}{\theta+2}
	\end{align}\
$$

となります。

今回は基底分布から生成されたため、式(4)の第二項で$c_{\cdot}=2$、$t_{\cdot}=1$として得られた確率

$$
	\begin{align}
		p(x_3=w_2 \mid \boldsymbol \Theta) = \frac {\theta+d}{\theta+2}G_0(w_2)
	\end{align}\
$$

を式(6)に掛けると式(8)が得られます。

図では客は3人いますが、$c_{\cdot}=2$であることに注意が必要です。（3人目の客を生成するための確率ですので、客数を3にしてしまってはいけません）

次に経験分布から$w_2$が生成された場合を考えると、

![CRP](/images/post/2016-10-16/p_seating_arrangement_4.png)

この時の$p(\boldsymbol \Theta)$は

$$
	\begin{align}
		p(\boldsymbol \Theta) = G_0(w_1)G_0(w_2)\frac{1-d}{\theta+1}\frac{\theta+d}{\theta+2}\frac{1-d}{\theta+3}
	\end{align}\
$$

となります。

さらに経験分布から$w_2$が生成された場合を考えると、

![CRP](/images/post/2016-10-16/p_seating_arrangement_5.png)

この時の$p(\boldsymbol \Theta)$は

$$
	\begin{align}
		p(\boldsymbol \Theta) = G_0(w_1)G_0(w_2)\frac{1-d}{\theta+1}\frac{\theta+d}{\theta+2}\frac{1-d}{\theta+3}\frac{2-d}{\theta+4}
	\end{align}\
$$

となります。

式(4)の第1項で$c_{w_2}=2$、$t_{w_2}=1$、$c_{\cdot}=4$として得られた確率

$$
	\begin{align}
		p(x_5=w_2 \mid \boldsymbol \Theta) = \frac {2-d}{\theta+4}
	\end{align}\
$$

を式(9)に掛けると式(10)が得られます。

次に$w_3$が$G_0$から生成されたとすると、

![CRP](/images/post/2016-10-16/p_seating_arrangement_6.png)

この時の$p(\boldsymbol \Theta)$は

$$
	\begin{align}
		p(\boldsymbol \Theta) = G_0(w_1)G_0(w_2)G_0(w_3)\frac{1-d}{\theta+1}\frac{\theta+d}{\theta+2}\frac{1-d}{\theta+3}\frac{2-d}{\theta+4}\frac{\theta+2d}{\theta+5}
	\end{align}\
$$

となります。

次に$G_0$から$w_1$がもう一度生成されたとします。

![CRP](/images/post/2016-10-16/p_seating_arrangement_7.png)

この時の$p(\boldsymbol \Theta)$は

$$
	\begin{align}
		p(\boldsymbol \Theta) = G_0(w_1)^2G_0(w_2)G_0(w_3)\frac{1-d}{\theta+1}\frac{\theta+d}{\theta+2}\frac{1-d}{\theta+3}\frac{2-d}{\theta+4}\frac{\theta+2d}{\theta+5}\frac{\theta+3d}{\theta+6}
	\end{align}\
$$

となります。

最後に経験分布から$w_1$が生成されたとすると、

![CRP](/images/post/2016-10-16/p_seating_arrangement_8.png)

この時の$p(\boldsymbol \Theta)$は

$$
	\begin{align}
		p(\boldsymbol \Theta) = G_0(w_1)^2G_0(w_2)G_0(w_3)\frac{1-d}{\theta+1}\frac{\theta+d}{\theta+2}\frac{1-d}{\theta+3}\frac{2-d}{\theta+4}\frac{\theta+2d}{\theta+5}\frac{\theta+3d}{\theta+6}\frac{3-2d}{\theta+7}
	\end{align}\
$$

となります。
