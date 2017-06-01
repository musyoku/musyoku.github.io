---
layout: post
title:  Quasi-Recurrent Neural Networks [1611.01576]
category: 実装
tags:
- 自然言語処理
excerpt_separator: <!--more-->
---

## 概要

- [Quasi-Recurrent Neural Networks](https://arxiv.org/abs/1611.01576)を読んだ
- Chainerで実装した

<!--more-->

## はじめに

Quasi-Recurrent Neural Networks（以下QRNN）は、RNNの機構をCNNで模したモデルです。

QRNNは半年前の論文ですので、ネットにはすでに[解説記事](http://qiita.com/icoxfog417/items/d77912e10a7c60ae680e)がいくつかあります。

そのためこの記事では実装面について書いていきます。

またQRNNの論文著者らはChainerで実装しており、[MetaMindのブログ](https://metamind.io/research/new-neural-network-building-block-allows-faster-and-more-accurate-text-understanding/)にコードが一部載っています。

私の実装は[musyoku/chainer-qrnn](https://github.com/musyoku/chainer-qrnn)です。

## 1-D Convolution

QRNNのCNNは1次元のCNNを用います。

![image](/images/post/2017-05-30/qrnn-conv.png)

QRNNの入力は単語の埋め込みベクトルを時系列に並べたものになっています。

埋め込みベクトルの次元数がチャネル数になります。

word2vecなどで200次元に埋め込んだ場合、チャネル数は200になります。

また畳み込みに用いるフィルタ（カーネル）は幅だけを持っていて、幅を変えることで畳み込む時刻の範囲を変えることができます。

2次元の畳込みではフィルタは縦横にスライドさせますが、1次元では横に（時刻方向に）のみスライドします。

Chainerでは`ConvolutionND`で次元数に1を設定すれば実装できます。

## プーリング

上記の1-D Convolutionで入力データを畳み込んで隠れ層の状態を計算します。

まず忘却ゲートや出力ゲートを以下のように計算します。

$$
  \begin{align}
    \boldsymbol Z &= {\rm tanh}(\boldsymbol W_z * \boldsymbol X)\\
    \boldsymbol F &= {\rm \sigma}(\boldsymbol W_f * \boldsymbol X)\\
    \boldsymbol O &= {\rm \sigma}(\boldsymbol W_o * \boldsymbol X)\\
    \boldsymbol I &= {\rm \sigma}(\boldsymbol W_i * \boldsymbol X)\\
  \end{align}\
$$

$*$は1次元の畳込みを表しています。

ここで$\boldsymbol Z$などが大文字なっていますが、これは行列を表しており、時刻$t$の出力ベクトル$\boldsymbol z_t$を$t=1$から$T$まで集めたものになっています。

$\boldsymbol F$や$\boldsymbol O$、$\boldsymbol I$も同様です。

これはどういうことかというと、QRNNではゲートの値を全時刻同時に計算します。

（正確に言うと、畳み込みで時刻方向にフィルタがスライドするので全時刻の値が同時に求まります）

次に各時刻の隠れ層の状態をゲートを使って個別に計算していきます。

時刻$t$の隠れ層出力を$h_t$とすると、RNNと同様に、1時刻前の$h_{t-1}$と現在の$z_t$から$h_t$を計算します。

この計算部分をQRNNではプーリングと読んでおり、f-pooling、fo-pooling、ifo-poolingの3種類が提案されています。

まずf-poolingは以下のようになります。

$$
  \begin{align}
    \boldsymbol h_t = \boldsymbol f_t \odot \boldsymbol h_{t-1} + (1-\boldsymbol f_t) \odot \boldsymbol z_t\\
  \end{align}\
$$

$\odot$は要素積を表しています。

これは2つのベクトルの同じ位置の値同士を掛け算します。（内積ではないので計算結果もベクトルになります）

fo-poolingはLSTMのようなセル出力を用いたもので、以下のようになります。

$$
  \begin{align}
    \boldsymbol c_t &= \boldsymbol f_t \odot \boldsymbol c_{t-1} + (1-\boldsymbol f_t) \odot \boldsymbol z_t\\
    \boldsymbol h_t &= \boldsymbol o_t \odot \boldsymbol c_t\\
  \end{align}\
$$

さらに、混合度合いを個別に調整するものがifo-poolingで、$1-\boldsymbol f_t$を置き換えた以下の形になります。

$$
  \begin{align}
    \boldsymbol c_t &= \boldsymbol f_t \odot \boldsymbol c_{t-1} + \boldsymbol i_t \odot \boldsymbol z_t\\
    \boldsymbol h_t &= \boldsymbol o_t \odot \boldsymbol c_t\\
  \end{align}\
$$

$\boldsymbol h_t$がその層の時刻$t$での出力になります。

まとめると、全時刻の入力データ$\boldsymbol X$に対して畳み込みを行ない、全時刻のゲートの値を同時に計算します。

その後forループで各時刻の$h_t$を順に計算していきます。

実装の際は、式(1)～式(4)を個別に実行するのではなく、$W_z$、$W_f$、$W_o$、$W_i$をまとめた１つの重み$W$で畳み込んでから分割したほうが速いです。

たとえばfo-poolingであれば必要になるのは$W_f$と$W_o$の2つなので、

```
W = ConvolutionND(1, in_channels, 2 * out_channels, kernel_size, stride=1, pad=kernel_size - 1)
```

のように出力チャネル数を2倍にしておいて、

```
WX = self.W(X)
F, O = split_axis(WX, 2, axis=1)
```

のように`split_axis`で分割します。

これらプーリング部分のコードを抜き出して載せておきます。

```
from chainer import link, functions, links, initializers

class QRNN(link.Chain):
  def __＄ｈinit__(self, in_channels, out_channels, kernel_size=2, pooling="f", zoneout=0, wgain=1., weightnorm=False):
    self.num_split = len(pooling) + 1
    if weightnorm:
      wstd = 0.05
      W = Convolution1D(in_channels, self.num_split * out_channels, kernel_size, stride=1, pad=kernel_size - 1, initialV=initializers.HeNormal(wstd))
    else:
      wstd = math.sqrt(wgain / in_channels / kernel_size)
      W = links.ConvolutionND(1, in_channels, self.num_split * out_channels, kernel_size, stride=1, pad=kernel_size - 1, initialW=initializers.HeNormal(wstd))

    super(QRNN, self).__init__(W=W)
    self._in_channels, self._out_channels, self._kernel_size, self._pooling, self._zoneout = in_channels, out_channels, kernel_size, pooling, zoneout
    self._using_zoneout = True if self._zoneout > 0 else False
    self.reset_state()

  def __call__(self, X, skip_mask=None):
    pad = self._kernel_size - 1
    WX = self.W(X)[..., :-pad]

    return self.pool(functions.split_axis(WX, self.num_split, axis=1), skip_mask=skip_mask)

  def pool(self, WX, skip_mask=None):
    Z, F, O, I = None, None, None, None

    # f-pooling
    if len(self._pooling) == 1:
      assert len(WX) == 2
      Z, F = WX
      Z = functions.tanh(Z)
      F = self.zoneout(F)

    # fo-pooling
    if len(self._pooling) == 2:
      assert len(WX) == 3
      Z, F, O = WX
      Z = functions.tanh(Z)
      F = self.zoneout(F)
      O = functions.sigmoid(O)

    # ifo-pooling
    if len(self._pooling) == 3:
      assert len(WX) == 4
      Z, F, O, I = WX
      Z = functions.tanh(Z)
      F = self.zoneout(F)
      O = functions.sigmoid(O)
      I = functions.sigmoid(I)

    assert Z is not None
    assert F is not None

    T = Z.shape[2]
    for t in xrange(T):
      zt = Z[..., t]
      ft = F[..., t]
      ot = 1 if O is None else O[..., t]
      it = 1 - ft if I is None else I[..., t]
      xt = 1 if skip_mask is None else skip_mask[:, t, None]  # will be used for seq2seq to skip PAD

      if self.ct is None:
        self.ct = (1 - ft) * zt * xt
      else:
        self.ct = ft * self.ct + it * zt * xt
      self.ht = self.ct if O is None else ot * self.ct


      if self.H is None:
        self.H = functions.expand_dims(self.ht, 2)
      else:
        self.H = functions.concat((self.H, functions.expand_dims(self.ht, 2)), axis=2)

    return self.H
```

余談ですが、Chainerでは慣例的に`chainer.functions`を`F`の別名でimportすると思うのですが、QRNNではFが変数として出てくるので引っかからないようにしましょう。

## パディング

CNNで時系列データを扱う場合、未来の信号を畳み込んではいけません。

時刻$t$の出力を計算する時に、時刻$t+1$以降のデータが畳み込まれないように注意する必要があります。

これは適切にパディングを設定することで回避できます。

以下の図はフィルタサイズ3で時刻1から4までのデータの畳み込みを表しています。

![image](/images/post/2017-05-30/qrnn-padding.png)

（1つのマスが単語ベクトルを表しています。スカラーではありません。）

時刻1の出力を得るためには、フィルタサイズ-1のパディングを入れる必要があります。

Chainerではパディングを入れると両端に挿入されるので、出力は図のように右側に2時刻ぶんの余計なデータが付いています。

そのため出力の右からフィルタサイズ-1までの領域を削除して最終的な出力を得ます。

実装は簡単です。

```
pad = self.kernel_size - 1
WX = self.W(X)[..., :-pad]
```

## Zoneout

QRNNの正則化としてzoneoutが提案されています。

zoneoutは$h_{t-1}$の要素を確率的に選択し、変更を加えずに$h_t$にコピーします。

これは忘却ゲート$\boldsymbol f_t$の各要素を確率的に選び値を1で上書きすることと同じなので、ドロップアウトを応用することで実現できます。

（ドロップアウトは確率的に0になるため、1から引く必要があります）

式で書くと以下の2通りが考えられます。

どちらを使っても構いません。

$$
  \begin{align}
    \boldsymbol F &= 1 - {\rm dropout}(1-\sigma(\boldsymbol W_f * \boldsymbol X))\\
    \boldsymbol F &= 1 - {\rm dropout}(\sigma(-\boldsymbol W_f * \boldsymbol X))\\
  \end{align}\
$$

コードは以下のようになります。

```
def zoneout(self, U):
  if self._using_zoneout and chainer.config.train:
    return 1 - dropout(functions.sigmoid(-U), 0.1)
  return functions.sigmoid(U)
```

ちなみにQRNNの論文の初版はzoneoutの式が誤っているので注意が必要です。

## Encoder–Decoder

QRNNはEncoder–Decoderモデルにも適用することができます。

まずAttentionなしのモデルは以下のようになります。