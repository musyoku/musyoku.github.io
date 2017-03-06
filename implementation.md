---
layout: page
title: 
permalink: /implementation/
---

実装した論文・実装中の論文のリストです。

## Deep Learning

### [Unrolled Generative Adversarial Networks](https://github.com/musyoku/unrolled-gan/)

- GANの学習を安定化させるテクニックの実装です
- 実装難易度: **1**

### [Improved Techniques for Training GANs](https://github.com/musyoku/improved-gan/)

- GANの学習を安定化させるテクニックの実装です
- 実装難易度: **1**

### [Auxiliary Deep Generative Models](https://github.com/musyoku/adgm/)

- VAEの拡張です
- 実装難易度: **3**

### [Distributional Smoothing with Virtual Adversarial Training](https://github.com/musyoku/vat/)

- クラス分類の汎化性能を高めます
- 予測分布$p(y \mid x)$が乱れてしまうようなノイズ$r$をデータ$x$に加え、$p(y \mid x+r)$が滑らかになるように学習します
- MNISTの半教師あり学習でほぼワンショット学習を達成できます
- 実装難易度: **2**

### [Adversarial AutoEncoder](https://github.com/musyoku/adversarial-autoencoder/)

- オートエンコーダの隠れ変数の正則化にGANを使います
- VAEに比べて隠れ変数を狙った分布に綺麗に押し込めます
- 実装難易度: **2**

### [Deep Directed Generative Models with Energy-Based Probability Estimation](https://github.com/musyoku/ddgm)

- 積モデル（Products of Experts）によりデータのエネルギーを計算します
- データ分布から外れた偽のデータに対してはエネルギーが高くなり、本物のデータのエネルギーは低くなります
- データ分布からのサンプリングをGeneratorによって近似します
- 実装難易度: **2**

### [Categorical Reparameterization with Gumbel-Softmax](http://musyoku.github.io/2016/11/12/Categorical-Reparameterization-with-Gumbel-Softmax/)

- Softmax層からの離散的なサンプリングを微分可能な形で行うことができる手法です
- 実装難易度: **1**

### [Weight Normalization Layer for Chainer](https://github.com/musyoku/weight-normalization/)

- 重みを正規化する手法です
- ChainerのLinkの拡張になっています
- あまり良い結果が得られていないので実装が合っているか不安です
- 実装難易度: **1**

### [WaveNet: A Generative Model for Raw Audio](https://github.com/musyoku/wavenet/)

- 人間とほぼ同等な品質の音声生成ができるモデルです
- 具体的なことが何も書かれていないため実装者によるところが大きいです
- 計算量が大きすぎるため個人では実験することが困難だと思います
- 実装難易度: **3**

### [Semi-Supervised Learning with Deep Generative Models](https://github.com/musyoku/variational-autoencoder/)

- VAEです
- データの尤度を直接計算するのは困難なため、その対数尤度の下限値を増大させることで学習を行います
- 実装難易度: **3**

## Deep Q-Network

### [Human-level control through deep reinforcement learning](https://github.com/musyoku/deep-q-network/)

- ご存知DQNです
- 強化学習をDeepにしています
- 実装難易度: **2**

### [Deep Reinforcement Learning with Double Q-learning](https://github.com/musyoku/double-dqn/)

- DQNの拡張です
- 行動の選択と評価を別々のQ関数で行ないます
- 実装難易度: **2**

### [Dueling Network Architectures for Deep Reinforcement Learning](https://github.com/musyoku/dueling-network/)

- DQNの拡張です
- Q関数を分解したDueling Architectureを提案しています
- 実装難易度: **2**

## 自然言語処理

### [ベイズ階層言語モデルによる教師なし形態素解析](https://github.com/musyoku/python-npylm/)

- 文字列データから教師なしで文字nグラムモデルと単語nグラムモデルを学習します
- 単語nグラムの学習の際に文字列の単語分割を推定します
- 学習が進むにつれて自然な単語分割を教師なしで獲得できます
- ノンパラメトリックベイズの知識が必要なため、論文の解読に2ヵ月、実装に3ヶ月を要しました
- デバッグが非常に辛いです
- 現時点の最高難易度(2017/01)
- 実装難易度: **6**

### [Pitman-Yor過程に基づく可変長n-gram言語モデル](https://github.com/musyoku/vpylm-python/)

- VPYLMです
- 言語モデルは通常バイグラムやトライグラムのように文脈長が固定ですが、これを可変長にできるようにしたモデルです
- 可変長といえばRNN言語モデルなどもそうですが、VPYLMの利点はどのオーダーからデータが生成されたかを予測することができ、必要最低限のオーダーでモデルが学習されていきます
- 実装難易度: **4**

### [A Hierarchical Bayesian Language Model based on Pitman-Yor Processes](https://github.com/musyoku/hpylm/)

- HPYLMです
- 言語モデルのスムージングをベイズ的に行えます
- Kneser-NeyスムージングがHPYLMの近似であることが判明しました
- 実装難易度: **3**

### [A Fully Bayesian Approach to Unsupervised Part-of-Speech Tagging](https://github.com/musyoku/unsupervised-pos-tagging/tree/master/bayesian-hmm/)

- ベイズ的なアプローチにより教師なし品詞推定が行なえます
- 品詞数は事前に要設定
- 実装難易度: **3**

## ツール類

### [Python tools for unsupervised POS tagging](https://github.com/musyoku/unsupervised-pos-tagging/)

- 教師なし品詞推定の代表的な手法を実装しています
- 現在も実装途中です

### [教師なし形態素解析でWordCloud](https://github.com/musyoku/unsupervised-wordcloud/)

- 教師なし形態素解析によりあらゆる言語データからワードクラウドを作れます
- 冠詞などの意味のない頻出語を除外することが今後の課題です