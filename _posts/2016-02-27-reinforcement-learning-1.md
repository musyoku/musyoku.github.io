---
layout: post
title: 強化学習（1）- 価値反復
category: 強化学習
tags:
- 実装
- 強化学習
excerpt_separator: <!--more-->
---

最近（というか昨日）、Sutton本と呼ばれる「[強化学習](http://www.amazon.co.jp/dp/4627826613)」という本を使って勉強を始めました。

今回は価値反復の章（p.107）の例4.3「ギャンブラーの問題」についてプログラムを作成しました。

ちなみにこの本は英語版が全ページHTML化されておりWeb上で読むことができます。

今回の範囲は[こちら](http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node44.html)です。

<!--more-->

この例題の設定は以下のようになります。

- プレイヤーはコイントスを行う
- コインの表が出れば賭け金を所持金に加える
- 裏が出れば賭け金が所持金から引かれる
- 所持金が100ドルに到達すればプレイヤーの勝利
- 所持金がなくなればプレイヤーの敗北

このような状況で、所持金を100ドルにするためには何ドル賭ければ良いのか（これを最適方策と呼びます）を強化学習によって求めます。

価値反復では動的計画法により以下のように価値関数を更新します。

$$
	\begin{align}
		V_{k+1}(s)=\max_a\sum_{s'}{\cal P}_{ss'}^a[{\cal R}_{ss'}^a+\gamma V_k(s')]
	\end{align}
$$

エピソード的タスクなので割引率$\gamma=1$です。

ちなみに$${\cal P}_{ss'}^a$$と$${\cal R}_{ss'}^a$$は以下のように求めます。

$$
	\begin{eqnarray}
		{\cal P}_{ss'}^a&=&P(s'\mid s,a)\\
		{\cal R}_{ss'}^a&=&{\double E}_{P(s'\mid s,a)}[r_{t+1}]\\
		&=&P(s'\mid s,a)*r_{t+1}
	\end{eqnarray}
$$

$${\cal P}_{ss'}^a$$は$(s,a)$が与えられた時に$s'$に遷移する確率を表します。

また、$${\cal R}_{ss'}^a$$は$(s,a)$から$s'$に遷移した時に貰える報酬の期待値です。よって実際に貰った報酬$r_{t+1}$に対して、その状態$s'$が起こる確率$$P(s'\mid s,a)$$を掛けます。

## 実験

この本によれば学習結果は以下のようになるそうです。

![実行結果](https://raw.githubusercontent.com/musyoku/reinforcement-learning/master/value_iteration/figtmp17.png)
<http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node44.html>より引用

下段のグラフは学習完了後のもので、横軸が現在の所持金を表し、縦軸はその所持金のうちいくらを賭ければ勝つ確率が一番高いかを表しています。

これはコインの表が出る確率$p=0.4$とした時の結果です。

注目すべき所は、所持金が50ドルの時は50ドル全部賭けてしまうのに、所持金が51ドルだとたった1ドルしか賭けないということです。

練習問題4.7はこの現象の理由を考察せよとなっていますので、実装して検証しました。

コードは[GitHub](https://github.com/musyoku/reinforcement-learning/tree/master/value_iteration)にあります。

まず価値関数の変化は以下のようになりました。

![価値関数](https://raw.githubusercontent.com/musyoku/reinforcement-learning/master/value_iteration/value_estimates.png)

これは本に載っている結果と一致しました。

この結果は、現在の所持金が$s$の時の$V(s)$（つまり勝率）を表しています。

今回はコインの表が出る確率が$0.4$なので、所持金50ドルの時の勝率も$0.4$付近となっておりこの結果は正しいと言えると思います。（全部賭けたら$0.4$の確率で100ドルに到達できます）

次に最終的に得られた最適方策です。

![最適方策](https://raw.githubusercontent.com/musyoku/reinforcement-learning/master/value_iteration/final_policy.png)

これは本の結果と微妙に違うものになりました。

最適方策は以下の式で求められますが、

$$
	\begin{align}
		\pi(s)=\argmax_a\sum_{s'}{\cal P}_{ss'}^a[{\cal R}_{ss'}^a+\gamma V_k(s')]
	\end{align}
$$

今回のケースでは2つの$a$が、演算精度の影響により同一の$$\sum_{s'}{\cal P}_{ss'}^a[{\cal R}_{ss'}^a+\gamma V_k(s')]$$を与えてしまったために上記の結果になりました。

そして練習問題4.7の答えですが、今回はちょうど100ドルに達した時のみ報酬が与えられる（つまり101ドルでは何も貰えない）という設定のせいではないかと思います。

50ドル全賭けして当たれば100ドルに達しますが、51ドルの場合は全部賭けると当たれば102ドル（無報酬）、外れれば50ドル（無報酬、ただし次に当てれば100ドル到達）になりますので、51ドルの時はわざと1ドル賭けて負けて50ドルにしないと100ドルに到達しないからだと思います。