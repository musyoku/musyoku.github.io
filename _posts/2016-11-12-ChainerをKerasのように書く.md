---
layout: post
title: ChainerをKerasのように書く
category: 論文
tags:
- Chainer
excerpt_separator: <!--more-->
---

## 概要

- Chainerのネットワーク構造をKerasのように書きたい
- 構造を保存したい

<!--more-->

## はじめに

KerasというDeep Learningフレームワークがあります。

Kerasではレイヤーを以下のように作ることができます。

```
from keras.layers import Dense, Activation

model.add(Dense(output_dim=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))
```

上から順にデータが流れていくので非常にわかりやすいですね。

Chainerでも同等のことはできなくもないのですが、ネットワーク構造をハードコーディングする必要があります。

それの何が問題かというと、様々なハイパーパラメータを試すときにコードを毎回変更しなければならないことです。

たとえばレイヤー数を変える、活性化関数を変える、重みの初期値を変える、Batch Normalizationのありなし、Dropoutのありなし、など、Depp Learningでは様々なハイパーパラメータがあります。

LeCun先生の論文[Energy-based Generative Adversarial Network](https://arxiv.org/abs/1609.03126)では、6,144通りの組み合わせを全て実験したと報告されています。

私も論文の追試を行う際はだいたい30通りくらい試すのですが、最近Kerasのように書きたい欲が高まってきたので作ってみました。

[chainer-sequential](https://github.com/musyoku/chainer-sequential)

あまり完成度は高くないですが、ネットワーク構造をjsonで保存できるようにし、ネットワークの組み立て方もKerasを意識して作りました。

以下に二乗誤差で学習する例を示します。

```
from link import Linear, BatchNormalization
from function import Activation
from chain import Chain

x = np.random.normal(scale=1, size=(128, 28*28)).astype(np.float32)
x = Variable(x)

model = Sequential(weight_initializer="GlorotNormal", weight_init_std=0.05)
model.add(Linear(28*28, 500, use_weightnorm=True))
model.add(BatchNormalization(500))
model.add(Activation("relu"))
model.add(Linear(None, 500, use_weightnorm=True))
model.add(BatchNormalization(500))
model.add(Activation("relu"))
model.add(Linear(500, 28*28, use_weightnorm=True))
model.build()

chain = Chain()
chain.add_sequence(model)
chain.setup_optimizers("adam", 0.001, momentum=0.9, weight_decay=0.000001, gradient_clipping=10)

for i in xrange(100):
    y = chain(x)
    loss = F.mean_squared_error(x, y)
    chain.backprop(loss)
    print float(loss.data)

chain.save("model")
```

構造をJSONに出力したり、JSONから構造を読み込むには、

```
json_str = model.to_json()
model.from_json(json_str)
```

のように行います。

実際にこのライブラリを使って作ったものが以下になります。

- [Auxiliary Deep Generative Models](https://github.com/musyoku/adgm)
- [Deep Directed Generative Models with Energy-Based Probability Estimation](https://github.com/musyoku/ddgm)
- [Improved Techniques for Training GANs](https://github.com/musyoku/improved-gan)

## おわりに

一見、ChainerのDefine by runの思想に反するような書き方をしていますが、内部的には入力されたデータに対して、あらかじめユーザーが定義した順でfunctionやlinkに通しています。

１つのネットワークしかないようなモデルでは最初からKerasを使っていたほうが良いと思いますが、最近のニューラルネットは複雑化してきており、例えばAuxiliary Deep Generative Modelsは5つのニューラルネットで5つの確率分布をモデル化し、それらの出力が複雑に目的関数に組み込まれているため、ネットワーク構造だけ決めておいて目的関数をDefine by runで組み立てていくほうがコードがわかりやすくなると思います。