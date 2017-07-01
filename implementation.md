---
layout: page
title: 
permalink: /implementation/
---

実装した論文やツールなど。

実装難易度は10段階です。記載がないものは容易に実装できます。

## Chainer

### [chainer.Stream](https://github.com/musyoku/chainer-stream)

- ChainerのChainをもっと書きやすくするために作りました

## Deep Learning

### [Self-Normalizing Neural Networks](https://github.com/musyoku/self-normalizing-networks)

- 出力が自動的に正規化される活性化関数です
- 効く場合と効かない場合があります

### [Learning Discrete Representations via Information Maximizing Self Augmented Training](https://github.com/musyoku/IMSAT)

- Regularized Information MaximizationとVirtual Adversarial Trainingを組み合わせた手法です
- MNISTを教師なしで98%分類できます
- 実装難易度: **3**

### [Quasi-Recurrent Neural Networks](https://github.com/musyoku/chainer-qrnn)

- CNNでRNNの機構を模して高速化したモデルです

### [Boundary Equilibrium GAN](https://github.com/musyoku/began)

- オートエンコーダベースのGANをWasserstein距離を用いて学習します。
- 学習が非常に不安定です

### [Least Squares GAN](https://github.com/musyoku/LSGAN/)

- 正解ラベルに対する二乗誤差を用いて学習を行うGANです
- 綺麗な画像が生成されます

### [Wasserstein GAN](https://github.com/musyoku/wasserstein-gan/)

- Wasserstein距離を最小化することで学習を行うGANです
- mode collapseの回避能力に優れています
- 実験を行った範囲では生成される画像が実データにあまり近くありませんでした

### [Unrolled Generative Adversarial Networks](https://github.com/musyoku/unrolled-gan/)

- GANの学習を安定化させるテクニックの実装です

### [Improved Techniques for Training GANs](https://github.com/musyoku/improved-gan/)

- GANの学習を安定化させるテクニックの実装です

### [Auxiliary Deep Generative Models](https://github.com/musyoku/adgm/)

- VAEの拡張です
- 実装難易度: **3**

### [Distributional Smoothing with Virtual Adversarial Training](https://github.com/musyoku/vat/)

- クラス分類の汎化性能を高めます
- 予測分布$p(y \mid x)$が乱れてしまうようなノイズ$r$をデータ$x$に加え、$p(y \mid x+r)$が滑らかになるように学習します
- MNISTの半教師あり学習でほぼワンショット学習を達成できます

### [Adversarial AutoEncoder](https://github.com/musyoku/adversarial-autoencoder/)

- オートエンコーダの隠れ変数の正則化にGANを使います
- VAEに比べて隠れ変数を狙った分布に綺麗に押し込めます

### [Deep Directed Generative Models with Energy-Based Probability Estimation](https://github.com/musyoku/ddgm)

- 積モデル（Products of Experts）によりデータのエネルギーを計算します
- データ分布から外れた偽のデータに対してはエネルギーが高くなり、本物のデータのエネルギーは低くなります
- データ分布からのサンプリングをGeneratorによって近似します

### [Categorical Reparameterization with Gumbel-Softmax](http://musyoku.github.io/2016/11/12/Categorical-Reparameterization-with-Gumbel-Softmax/)

- Softmax層からの離散的なサンプリングを微分可能な形で行うことができる手法です

### [Weight Normalization](https://github.com/musyoku/weight-normalization/)

- 重みを正規化する手法です
- ChainerのLinkの拡張になっています

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

### [Deep Reinforcement Learning with Double Q-learning](https://github.com/musyoku/double-dqn/)

- DQNの拡張です
- 行動の選択と評価を別々のQ関数で行ないます

### [Dueling Network Architectures for Deep Reinforcement Learning](https://github.com/musyoku/dueling-network/)

- DQNの拡張です
- Q関数を分解したDueling Architectureを提案しています

## 自然言語処理

### [条件付確率場とベイズ階層言語モデルの統合による半教師あり形態素解析](https://github.com/musyoku/python-npycrf)

- 現在実装しています
- 実装難易度: **7**

### [無限木構造隠れMarkovモデルによる階層的品詞の教師なし学習](https://github.com/musyoku/unsupervised-pos-tagging/tree/master/infinite-tree-hmm/)

- 木構造棒折り過程を階層化して状態遷移確率を表すHMMです
- 状態数と状態の階層構造をデータから学習します
- デバッグが非常に辛いです
- 実装難易度: **6**

### [ベイズ階層言語モデルによる教師なし形態素解析](https://github.com/musyoku/python-npylm/)

- 文字列データから教師なしで文字nグラムモデルと単語nグラムモデルを学習します
- 単語nグラムの学習の際に文字列の単語分割を推定します
- 学習が進むにつれて自然な単語分割を教師なしで獲得できます
- デバッグが非常に辛いです
- 実装難易度: **6**

### [Pitman-Yor過程に基づく可変長n-gram言語モデル](https://github.com/musyoku/vpylm-python/)

- VPYLMです
- 言語モデルは通常バイグラムやトライグラムのように文脈長が固定ですが、これを可変長にできるようにしたモデルです
- 可変長といえばRNN言語モデルなどがありますが、VPYLMの利点はどのオーダーからデータが生成されたかを推定することができ、必要最低限のオーダーでモデルが学習されていきます
- 実装難易度: **5**

### [ガウス過程に基づく連続空間トピックモデル](https://github.com/musyoku/cstm)

- 単語に潜在空間における座標を与えたトピックモデルです
- 単語の文書との共起をモデル化しています
- 機能語の共起が抑えられるため、意味のある単語を抽出しやすくなっています
- 実装難易度: **4**

### [A Hierarchical Bayesian Language Model based on Pitman-Yor Processes](https://github.com/musyoku/hpylm/)

- HPYLMです
- 言語モデルのスムージングをベイズ的に行えます
- Kneser-NeyスムージングがHPYLMの近似であることが判明しました
- 実装難易度: **4**

### [A Fully Bayesian Approach to Unsupervised Part-of-Speech Tagging](https://github.com/musyoku/unsupervised-pos-tagging/tree/master/bayesian-hmm/)

- ベイズ的なアプローチにより教師なし品詞推定が行なえます
- 品詞数は事前に要設定
- 実装難易度: **4**

### [The Infinite Hidden Markov Model](https://github.com/musyoku/unsupervised-pos-tagging/tree/master/infinite-hmm/)

- ベイズ的なアプローチにより教師なし品詞推定が行なえます
- 品詞数もデータから推定できます
- 実装難易度: **4**

## ツール類

### [Python tools for unsupervised POS tagging](https://github.com/musyoku/unsupervised-pos-tagging/)

- 教師なし品詞推定の代表的な手法を実装しています
- 現在も実装途中です

### [教師なし形態素解析でWordCloud](https://github.com/musyoku/unsupervised-wordcloud/)

- 教師なし形態素解析によりあらゆる言語データからワードクラウドを作れます
- 冠詞などの意味のない頻出語を除外することが今後の課題です