---
layout: post
title:  ChainerのChainをもう少し楽に書く
category: 実装
tags:
- Chainer
excerpt_separator: <!--more-->
---

## 概要

- Chainerの小ネタ

<!--more-->

## はじめに

Chainerでネットワークを設計する時は、ChainにLinkを追加して`__call__`で各層の出力を設計すると思います。

コードで書くとこんな感じです。（わざと巨大なネットワークにしています）

```
class Chain(chainer.Chain):
  def __init__(self):
    super(Chain, self).__init__()
    with self.init_scope():
      self.l1 = L.Linear(None, 1024)
      self.l2 = L.Linear(None, 512)
      self.l3 = L.Linear(None, 256)
      self.l4 = L.Linear(None, 128)
      self.l5 = L.Linear(None, 10)
      self.bn1 = L.BatchNormalization(1024)
      self.bn2 = L.BatchNormalization(512)
      self.bn3 = L.BatchNormalization(256)
      self.bn4 = L.BatchNormalization(128)

  def __call__(self, x):
    out = self.l1(x)
    out = F.leaky_relu(out)
    out = self.bn1(out)
    out = self.l2(out)
    out = F.relu(out)
    out = self.bn2(out)
    out = self.l3(out)
    out = F.elu(out)
    out = self.bn3(out)
    out = self.l4(out)
    out = F.relu(out)
    out = self.bn4(out)
    out = self.l5(out)
    return out
```

residualな接続などを使わない場合、わざわざ`__call__`で各層の出力を手動で書いていくのは面倒なのと、層の削除などで毎回`__call__`の中身を変更するのが面倒です。

以前に[ChainerをKerasのように書くためのツール](https://github.com/musyoku/chainer-sequential)を作りましたが、わりと巨大なコードになっているので、軽く実験する程度の場合にこれを使うのは過剰な気がします。

そこで以下のようなChainを作ってみました。

```
import chainer

class Chain(chainer.Chain):
  def __init__(self, *layers):
    super(Chain, self).__init__()
    assert len(layers) > 0
    assert not hasattr(self, "layers")
    self.layers = layers
    with self.init_scope():
      for idx, layer in enumerate(layers):
        if isinstance(layer, chainer.Link):
          setattr(self, "layer_%d" % idx, layer)

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
```

モデルの定義はこうです。

```
model = Chain(
  L.Linear(None, 1024),
  F.leaky_relu,
  L.BatchNormalization(1024),
  L.Linear(None, 512),
  F.relu,
  L.BatchNormalization(512),
  L.Linear(None, 256),
  F.elu,
  L.BatchNormalization(256),
  L.Linear(None, 128),
  F.relu,
  L.BatchNormalization(128),
  L.Linear(None, 10),
)
```

レイヤーと活性化関数を順に追加していくと、`__call__`で自動的に追加した順番に処理されます。

活性化関数を`F.relu`のように直接渡すのが若干気持ち悪いですが、今までに比べると楽に書けるのではないかと思います。

（Python歴が浅いので本当にこれで良いのか不安はあります）

動作確認用のコードは[musyoku/chainer-sequential-chain](https://github.com/musyoku/chainer-sequential-chain)にあります。

# 追記

ResNetももっと簡単に書きたくなったので別のものを作りました。

[musyoku/chainer-stream](https://github.com/musyoku/chainer-stream)

