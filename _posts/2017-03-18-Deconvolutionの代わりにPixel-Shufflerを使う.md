---
layout: post
title: Deconvolutionの代わりにPixel Shufflerを使う
category: 実装
tags:
- Chainer
excerpt_separator: <!--more-->
---

## 概要

- ChainerでPixel Shufflerを実装した

<!--more-->

## はじめに

昨年に[Twitter社が発表した超解像の論文](https://arxiv.org/abs/1609.04802)を読んでいたところ、画像の拡大によく用いられるDeconvolutionを使わず、Pixel Shufflerと呼ばれる仕組みを利用して超解像を行っており、気になったので調べました。

Pixel Shufflerは[Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158)で提案されたもので（正式にはSub-Pixel Convolution）、入力特徴マップの各ピクセルを並べ替えて高解像度な特徴マップを出力します。

今回はDCGANのGeneratorをDeconvolutionとPixel Shufflerの両方で構成して実験しました。

ちなみにDeconvolutionの問題点については[Deconvolution and Checkerboard Artifacts](http://distill.pub/2016/deconv-checkerboard/)がよくまとまっています。

## Sub-Pixel Convolution

まず記号を定義しておきます。

入力マップの高さを$H_{in}$、幅を$W_{in}$とし、出力マップのチャネル数を$C_{out}$とします。

また拡大倍率を$r$とします。

論文の図によると、以下のように入力マップを並べ替えて高解像な出力を得ます。

![image](/images/post/2017-03-18/subpixel_conv.jpg)

この並べ替えの処理を行う関数を${\cal PS}$とすると、${\cal PS}$は入力された$C_{out}\cdot r^2\times H_{in} \times W_{in}$の特徴マップを$C_{out}\times rH_{in}\times rW_{in}$に変形して出力します。

（この「チャネル数$\times$高さ$\times$幅」の並びはChainerに合わせています。TensorFlowでは「幅$\times$高さ$\times$チャネル数」の並びになっているため注意が必要です。これ以降はChainerに合わせて表記します）

例えば入力マップが$8\times2\times2$で倍率$r=2$の場合、出力マップは$2\times4\times4$となり、2倍に拡大されたものが出力されます。

${\cal PS}$は具体的には以下のように表されます。

$$
	\begin{align}
		{\cal PS}(T)_{c,y,x} = T_{C_{out}\cdot r \cdot {\rm mod}(y,r) + C_{out}\cdot {\rm mod}(x,r) + c,\lfloor \frac{y}{r} \rfloor,\lfloor \frac{x}{r} \rfloor}
	\end{align}\
$$

$T$は入力マップ、$c$は出力マップのチャネル位置、$y$は出力マップの縦位置、$x$は出力マップの横位置を表しています。

$\lfloor \cdot \rfloor$は床関数（floor）です。

この式が意味することは、出力マップ$T'$の位置$c,y,x$の値が、入力マップ$T$の位置$$C_{out}\cdot r \cdot {\rm mod}(y,r) + C_{out}\cdot {\rm mod}(x,r) + c,\lfloor \frac{y}{r} \rfloor,\lfloor \frac{x}{r} \rfloor$$の値になるということです。

プログラムのコード風に書くと

```
t_out[c,y,x] = t_in[c_out * r * mod(y, r) + c_out * mod(x, r) + c, floor(y / r), floor(x / r)]
```

のようになります。

ただし、実装する場合はnumpyなどの機能を使って一度にすべての値を求めます。

##  実装

Pixel Shufflerを実装する場合、$H_{in}$・$W_{in}$・$C_{out}$・$r$の4つを決めれば残りの変数$H_{out}$・$W_{out}$・$C_{in}$は自動的に決定します。

Pixel Shufflerはレイヤーとして用いるので、入力マップは自分より下の層の出力マップになります。

この時、下層の出力マップのチャネル数・高さ・幅が必ずしもPixel Shufflerレイヤーの入力としての要求を満たしているとは限りません。

例えば、$256\times 2 \times 2$のマップを下層が出力したとします。（チャネル数256、高さ2、幅2）

このマップを$r=2$のPixel Shufflerによって$128\times 4\times 4$にしたい場合、$C_{out}=128$なので、$C_{in} = C_{out} \times r^2$より入力マップのチャネル数は$512$である必要があるため、このままでは入力のチャネル数が足りません。

そこでPixel ShufflerレイヤーにConvolutionレイヤーを追加し、下層が出力した特徴マップを畳み込んでPixel Shufflerの要求を満たす形にし、その後${\cal PS}$によって目的の高解像マップを得ます。

（実際には${\cal PS}$自体はただの配列操作であり、学習すべき重みを持っていないためConvolutionレイヤーが必ず必要になります）

${\cal PS}$の実装ですが、式(1)を見ると、以下のように`reshape`と`transpose`を用いると実装できることが分かります。

```
out_map = np.reshape(in_map, (batchsize, 1, r * r, out_channels * in_height * in_width, 1))
out_map = np.transpose(out_map, (0, 1, 3, 2, 4))
out_map = np.reshape(out_map, (batchsize, out_channels, in_height, in_width, r, r))
out_map = np.transpose(out_map, (0, 1, 2, 4, 3, 5))
out_map = np.reshape(out_map, (batchsize, out_channels, out_height, out_width))
```

### 追記（2017/03/26）

コメントを頂きましたが、3行で書けるそうです。

```
out_map = np.reshape(in_map, (batchsize, r, r, out_channels, in_height, in_width))
out_map =  np.transpose(out_map, (0, 3, 4, 1, 5, 2))
out_map = np.reshape(out_map, (batchsize, out_channels, out_height, out_width))
```

numpyによる${\cal PS}$の実装とテストコードをGistにあげておきました。

[https://gist.github.com/musyoku/849094afca2889d9024f59e683fa7036](https://gist.github.com/musyoku/849094afca2889d9024f59e683fa7036)

${\cal PS}$が具体的にどういう値を出力するかを確認できます。

バッチサイズ$2$、$r=2$、$C_{out} = 3$、$H_{in}=3$、$W_{in}=3$の場合、まず入力マップは以下のようになります。

```
[[[[  0   1   2]
   [  3   4   5]
   [  6   7   8]]

  [[  9  10  11]
   [ 12  13  14]
   [ 15  16  17]]

  [[ 18  19  20]
   [ 21  22  23]
   [ 24  25  26]]

  [[ 27  28  29]
   [ 30  31  32]
   [ 33  34  35]]

  [[ 36  37  38]
   [ 39  40  41]
   [ 42  43  44]]

  [[ 45  46  47]
   [ 48  49  50]
   [ 51  52  53]]

  [[ 54  55  56]
   [ 57  58  59]
   [ 60  61  62]]

  [[ 63  64  65]
   [ 66  67  68]
   [ 69  70  71]]

  [[ 72  73  74]
   [ 75  76  77]
   [ 78  79  80]]

  [[ 81  82  83]
   [ 84  85  86]
   [ 87  88  89]]

  [[ 90  91  92]
   [ 93  94  95]
   [ 96  97  98]]

  [[ 99 100 101]
   [102 103 104]
   [105 106 107]]]


 [[[108 109 110]
   [111 112 113]
   [114 115 116]]

  [[117 118 119]
   [120 121 122]
   [123 124 125]]

  [[126 127 128]
   [129 130 131]
   [132 133 134]]

  [[135 136 137]
   [138 139 140]
   [141 142 143]]

  [[144 145 146]
   [147 148 149]
   [150 151 152]]

  [[153 154 155]
   [156 157 158]
   [159 160 161]]

  [[162 163 164]
   [165 166 167]
   [168 169 170]]

  [[171 172 173]
   [174 175 176]
   [177 178 179]]

  [[180 181 182]
   [183 184 185]
   [186 187 188]]

  [[189 190 191]
   [192 193 194]
   [195 196 197]]

  [[198 199 200]
   [201 202 203]
   [204 205 206]]

  [[207 208 209]
   [210 211 212]
   [213 214 215]]]]
```

これを${\cal PS}$に通すと以下のようになります。

```
[[[[  0  27   1  28   2  29]
   [ 54  81  55  82  56  83]
   [  3  30   4  31   5  32]
   [ 57  84  58  85  59  86]
   [  6  33   7  34   8  35]
   [ 60  87  61  88  62  89]]

  [[  9  36  10  37  11  38]
   [ 63  90  64  91  65  92]
   [ 12  39  13  40  14  41]
   [ 66  93  67  94  68  95]
   [ 15  42  16  43  17  44]
   [ 69  96  70  97  71  98]]

  [[ 18  45  19  46  20  47]
   [ 72  99  73 100  74 101]
   [ 21  48  22  49  23  50]
   [ 75 102  76 103  77 104]
   [ 24  51  25  52  26  53]
   [ 78 105  79 106  80 107]]]


 [[[108 135 109 136 110 137]
   [162 189 163 190 164 191]
   [111 138 112 139 113 140]
   [165 192 166 193 167 194]
   [114 141 115 142 116 143]
   [168 195 169 196 170 197]]

  [[117 144 118 145 119 146]
   [171 198 172 199 173 200]
   [120 147 121 148 122 149]
   [174 201 175 202 176 203]
   [123 150 124 151 125 152]
   [177 204 178 205 179 206]]

  [[126 153 127 154 128 155]
   [180 207 181 208 182 209]
   [129 156 130 157 131 158]
   [183 210 184 211 185 212]
   [132 159 133 160 134 161]
   [186 213 187 214 188 215]]]]
```

この`reshape`と`transpose`を重ねるやり方は冗長な感じがするので、他の方の実装を探してみたところ2件見つかりました。

- [https://github.com/Tetrachrome/subpixel](https://github.com/Tetrachrome/subpixel)
- [https://github.com/Hi-king/superresolution_gan/blob/master/srcgan/models.py](https://github.com/Hi-king/superresolution_gan/blob/master/srcgan/models.py)

これらの実装では`concat`や`separate`などを用いていますが、いずれにしても一発で出すことはできないようです。

ChainerでConvolutionと組み合わせて行う場合は以下のようなコードになります。

```
def __call__(self, x):
	r = self.r
	out = self.conv(x) # 畳み込み
	batchsize = out.shape[0]
	in_channels = out.shape[1]
	out_channels = in_channels / (r ** 2)
	in_height = out.shape[2]
	in_width = out.shape[3]
	out_height = in_height * r
	out_width = in_width * r
	out = F.reshape(out, (batchsize, 1, r * r, out_channels * in_height * in_width, 1))
	out = F.transpose(out, (0, 1, 3, 2, 4))
	out = F.reshape(out, (batchsize, out_channels, in_height, in_width, r, r))
	out = F.transpose(out, (0, 1, 2, 4, 3, 5))
	out = F.reshape(out, (batchsize, out_channels, out_height, out_width))
	return out
```

すごく遅そうですが実はDeconvolutionより速いです。

### 追記（2017/03/26）

より短く書けることが分かりました。

```
def __call__(self, x):
  r = self.r
  out = self.conv(x) # 畳み込み
  batchsize = out.shape[0]
  in_channels = out.shape[1]
  out_channels = in_channels / (r ** 2)
  in_height = out.shape[2]
  in_width = out.shape[3]
  out_height = in_height * r
  out_width = in_width * r
  out = F.reshape(out, (batchsize, r, r, out_channels, in_height, in_width))
  out = F.transpose(out, (0, 3, 4, 1, 5, 2))
  out = F.reshape(out, (batchsize, out_channels, out_height, out_width))
  return out
```

## 実験

[LSGAN](/2017/03/06/Least-Squares-Generative-Adversarial-Networks/)と[WGAN](/2017/02/06/Wasserstein-GAN/)でGeneratorをDeconvolution・Pixel Shufflerの2通りで学習させました。

層の数や各層の入出力チャネル数は全て同一になっており、Generatorは以下のようにLinearでノイズベクトルをマップに変換し、Deconvolution（またはPixelShuffler）で画像を拡大していき、$96\times96$の画像を生成しています。

```
(512,) -> (512, 6, 6) -> (256, 12, 12) -> (128, 24, 24) -> (64, 48, 48) -> (3, 96, 96)
```

Deconvolutionの場合は上の入出力チャネル数をそのままDeconvolution2Dに渡せば良いのですが、Pixel Shufflerは上述のように出力チャネル数を決めると入力チャネル数が一意に決定してしまい、これが下層の出力チャネル数と合わないためConvolutionでチャネル数を調整する必要があります。

今回はフィルタサイズ$3\times3$でpaddingを$1$にした畳み込みを行うことで調整を行いました。

チャネル数を変更するだけならフィルタサイズ$1\times1$の畳み込みもよく用いられますが、時間がなかったので今回は実験していません。

## LSGAN

### Deconvolution

![image](/images/post/2017-03-18/lsgan_dc.png)
![image](/images/post/2017-03-18/lsgan_analogy_dc.png)

### Pixel Shuffler

![image](/images/post/2017-03-18/lsgan_ps.png)
![image](/images/post/2017-03-18/lsgan_analogy_ps.png)

実行速度ですが、700epochの学習にDeconvolution版は2354分、Pixel Shuffler版は1450分かかったため、Pixel Shufflerの方が1.62倍速いです。

## WGAN

### Deconvolution

![image](/images/post/2017-03-18/wgan_dc.png)
![image](/images/post/2017-03-18/wgan_analogy_dc.png)

### Pixel Shuffler

![image](/images/post/2017-03-18/wgan_ps.png)
![image](/images/post/2017-03-18/wgan_analogy_ps.png)

実行速度ですが、1000epochの学習にDeconvolution版は3485分、Pixel Shuffler版は2199分かかったため、Pixel Shufflerの方が1.58倍速いです。

WGANは出力画像が歪みやすいので改善していきたいです。

## おわりに

今後DCGANを作って実験するときはPixel Shufflerをデフォルトにしようと思います。