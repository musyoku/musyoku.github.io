---
layout: post
title:  負の二項分布の一般化線形モデルで最大単語長の予測
category: 実装
tags:
- HMM
excerpt_separator: <!--more-->
---

## 概要

- [Inducing Word and Part-of-Speech with Pitman-Yor Hidden Semi-Markov Models](http://chasen.org/~daiti-m/paper/acl2015pyhsmm.pdf)を読んだ
- C++で実装した

<!--more-->

## はじめに

半教師あり形態素解析の[NPYCRF](http://www.anlp.jp/proceedings/annual_meeting/2016/pdf_dir/D6-3.pdf)の実装をするために論文を読んでいたところ、単語の最大長を予測して状態数を削減し高速化すると書かれていました。

著者の持橋先生に教えていただいたのですが、この手法の詳細は[Inducing Word and Part-of-Speech with Pitman-Yor Hidden Semi-Markov Models](http://chasen.org/~daiti-m/paper/acl2015pyhsmm.pdf)に載っており、気になったこともあったので実装してみました。

<blockquote class="twitter-tweet" data-lang="ja"><p lang="ja" dir="ltr">この時の話には我々の方で若干誤解があって、単語長は離散なので、実際には負の二項分布の一般化線形モデルを使っています。ACL2015の論文でわかるように、長い単語もかなりの精度で予測できています。</p>&mdash; Daichi Mochihashi (@daiti_m) <a href="https://twitter.com/daiti_m/status/881843313501847553">2017年7月3日</a></blockquote>
<script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>

実装は[musyoku/negative-binomial-glm](https://github.com/musyoku/negative-binomial-glm)です。

ちなみに以下の内容の大半は私の推測なので誤っている可能性があります。

## 単語長の分布

単語長$l$を負の二項分布（Negative Binomial）でモデル化します。

$$
  \begin{align}
    l \sim \text{NB}(l \mid r, p) = \frac{\Gamma(r+l)}{\Gamma(r)l!}p^l(1-p)^r
  \end{align}\
$$

負の二項分布はポアソン分布$\text{Po}(l \mid \lambda)$をガンマ分布$\text{Ga}(\lambda \mid r, b)$で混合することでも得られます。

$$
  \begin{align}
    \text{NB}(l \mid r, p) &= \int{\text{Po}(l \mid \lambda)\text{Ga}(\lambda \mid r, b)d\lambda}\\
    &= \frac{\Gamma(r+l)}{\Gamma(r)l!}{\left(\frac{b}{1+b}\right)}^l{\left(\frac{1}{1+b}\right)}^r
  \end{align}\
$$

[NPYLM](/2016/12/14/%E3%83%99%E3%82%A4%E3%82%BA%E9%9A%8E%E5%B1%A4%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AB%E3%82%88%E3%82%8B%E6%95%99%E5%B8%AB%E3%81%AA%E3%81%97%E5%BD%A2%E6%85%8B%E7%B4%A0%E8%A7%A3%E6%9E%90/)では単語の種類を考え（漢字＋ひらがな等）、単語種$i$ごとに異なる$\lambda_i$を用いて単語長の分布を$\text{Po}(l \mid \lambda_i)$としていましたが、上の式では$\lambda$について周辺化することで$\lambda$の変動を考慮した単語長の分布になっています。

あとはこの分布が教師データに合えば良いので、重み$\boldsymbol {\rm w}_r^T$と$\boldsymbol {\rm w}_p^T$を用いて負の二項分布のパラメータを

$$
  \begin{align}
    r &= \text{exp}(\boldsymbol {\rm w}_r^T \boldsymbol f)\\
    p &= \text{sigmoid}(\boldsymbol {\rm w}_p^T \boldsymbol f)
  \end{align}\
$$

として、最適な$\boldsymbol {\rm w}_r^T$と$\boldsymbol {\rm w}_p^T$を求めれば良いことになります。

$\boldsymbol f$は各要素が2値を取る素性ベクトルで、文字列$c_1...c_t$から求めます。

用いる素性は以下の4種類です。

- $c_i$
  - 位置$t-i\ (0 \leq i \leq 1)$の文字
  - Unicodeのコードポイントなど
- $t_i$
  - 位置$t-i\ (0 \leq i \leq 4)$の文字種
  - Unicodeのブロックなら280種類
- $cont$
  - 位置$t$の文字に到達するまでに文字が何回変わったか
  - 位置$t$以前の8文字から計算
- $ch$
  - 位置$t$の文字に到達するまでに文字種が何回変わったか
  - 位置$t$以前の8文字から計算

$\boldsymbol f$は数式上はベクトルですが、実装時にはベクトルにする必要はなく、例えば$cont=1$であればそれに対応する$w_r$や$w_p$を取ってくるだけでOKです。

たとえばデータ中の（文字の）語彙サイズが5000だったとします。

この場合$c_i$は$i$ごとに5000通りの値を取りますが、それに対応する重みも$i$ごとに5000個用意する必要があります。

$t_i$は上の例だと280通りの値をとるので、$i$ごとに対応する280個の重みを用意します。

$(0 \leq i \leq 4)$であれば$5 \times 280 = 1400$個になり、$r$と$p$を考慮するとさらに2倍の重みが必要です。

$cont$、$ch$も同様です。

## 訓練データ

パラメータの学習に必要なデータは文字列から計算した素性ベクトル$\boldsymbol f$と正解単語長$l$のペアです。

ここで「ご注文は機械学習ですか？」という文を考えます。

単語に分割すると「ご注文」「は」「機械学習」「です」「か」「？」になると思いますが、（脳内mecab）

素性ベクトルを計算するための文字列と正解単語長のペアを`[(ご注文, 3), (は, 1), (機械学習, 4), (です, 2), (か, 1), (？, 1)]`のようにしてはいけません。

今回のモデルは、正解の分割が与えられていない文に対しても単語の最大長を予測するために用いられます。

そのため上の例のように$\boldsymbol f$を単語の構成文字列から作るようにしてしまうと、正解の分割が与えられていない文では単語がそもそも分からないので$\boldsymbol f$を作れなくなります。

そこで対象となる単語だけを切り出してくるのではなく、その単語を含む一定の長さの文字列を取り出して訓練データとします。

上の例では`[(ご注文, 3), (ご注文は, 1), (ご注文は機械学習, 4), (文は機械学習です, 2), (は機械学習ですか, 1), (機械学習ですか？, 1)]`のようになります。

文字列の一番右の文字を含む単語が正解単語になっています。

この部分のコードは以下のようになります。

```
int substr_end = -1;
for(const auto &word: words){
  substr_end += word.length();
  // 単語ではなく単語を含む部分文字列にする
  int substr_start = std::max(0, substr_end - _coverage + 1); // coverageの範囲の文字列を全て取る
  std::wstring substr(sentence.begin() + substr_start, sentence.begin() + substr_end + 1);
  int true_length = word.length();
  pair.first = true_length;
  pair.second = substr;
  auto itr = _length_substr_set.find(pair);
  if(itr == _length_substr_set.end()){
    _length_substr_set.insert(pair);
  }
}
```

このような訓練データ集合を${\cal D}$で表します。

## 学習

$\boldsymbol {\rm w}_r^T$と$\boldsymbol {\rm w}_p^T$の最適化にはランダムウォークによるMCMCを用います。

それぞれ事前分布を正規分布とし、

$$
  \begin{align}
    p(\boldsymbol {\rm w}) = {\cal N}(\boldsymbol {\rm w}; \boldsymbol 0, \sigma_{\text{prior}}^2\boldsymbol {\rm I})
  \end{align}\
$$

ランダムウォークによって時刻$t$の重み$\boldsymbol w^{(t)}$から次の候補$\boldsymbol w^{(t+1)}$を生成します。

$$
  \begin{align}
    \boldsymbol {\rm w}_r^{(\text{new})} &= \boldsymbol {\rm w}_r^{(t)} + \boldsymbol \epsilon_r, \ \ \boldsymbol \epsilon_r \sim {\cal N}(\boldsymbol 0, \sigma_{\text{step}}^2\boldsymbol {\rm I})\\
    \boldsymbol {\rm w}_p^{(\text{new})} &= \boldsymbol {\rm w}_p^{(t)} + \boldsymbol \epsilon_p, \ \ \boldsymbol \epsilon_p \sim {\cal N}(\boldsymbol 0, \sigma_{\text{step}}^2\boldsymbol {\rm I})\\
  \end{align}\
$$

$\boldsymbol {\rm w}_r^{(\text{new})},\boldsymbol {\rm w}_p^{(\text{new})}$を以下の採択確率$\alpha$に従って受理するかどうかを決定します。

$$
  \begin{align}
    \alpha(\boldsymbol {\rm w}_r^{(\text{new})},\boldsymbol {\rm w}_p^{(\text{new})}, \boldsymbol {\rm w}_r^{(t)},\boldsymbol {\rm w}_p^{(t)}) &= \min\left\{1, \frac{\pi(\boldsymbol {\rm w}_r^{(\text{new})},\boldsymbol {\rm w}_p^{(\text{new})})}{\pi(\boldsymbol {\rm w}_r^{(t)},\boldsymbol {\rm w}_p^{(t)})}\right\}\\
    \text{where}\ \pi(\boldsymbol {\rm w}_r, \boldsymbol {\rm w}_p) &= p({\rm w}_r)p({\rm w}_p)\prod_{(l, \boldsymbol f) \in {\cal D}}p(l \mid r, p)\\
    &=p({\rm w}_r)p({\rm w}_p)\prod_{(l, \boldsymbol f) \in {\cal D}}p(l \mid \text{exp}(\boldsymbol {\rm w}_r^T \boldsymbol f), \text{sigmoid}(\boldsymbol {\rm w}_p^T \boldsymbol f))\\
  \end{align}\
$$

$\pi(\boldsymbol {\rm w}_r, \boldsymbol {\rm w}_p)$は訓練データの尤度と重みの尤度との積になっていて、新旧の重みの尤度比を採択確率とします。

採択確率を計算できたら、$u$を一様分布${\cal U}(0, 1)$から発生させ、

$$
  \begin{align}
    \boldsymbol {\rm w}_r^{(t+1)} = \begin{cases}
      \boldsymbol {\rm w}_r^{(\text{new})} \ \ \ \ &\text{if}\ u \leq \alpha(\boldsymbol {\rm w}_r^{(\text{new})},\boldsymbol {\rm w}_p^{(\text{new})}, \boldsymbol {\rm w}_r^{(t)},\boldsymbol {\rm w}_p^{(t)})\\
      \boldsymbol {\rm w}_r^{(t)} \ \ \ \ &\text{otherwise}
    \end{cases}\\
    \boldsymbol {\rm w}_p^{(t+1)} = \begin{cases}
      \boldsymbol {\rm w}_p^{(\text{new})} \ \ \ \ &\text{if}\ u \leq \alpha(\boldsymbol {\rm w}_r^{(\text{new})},\boldsymbol {\rm w}_p^{(\text{new})}, \boldsymbol {\rm w}_r^{(t)},\boldsymbol {\rm w}_p^{(t)}) \\
      \boldsymbol {\rm w}_p^{(t)} \ \ \ \ &\text{otherwise}
    \end{cases}
  \end{align}\
$$

として更新します。

実装時には式(7)や式(8)のように重みのすべての要素を同時に更新する必要はないと思うので、ある要素にだけ注目して国保の値を生成し、採択確率に従って更新すれば良いと思います。

## 最大単語長の予測

