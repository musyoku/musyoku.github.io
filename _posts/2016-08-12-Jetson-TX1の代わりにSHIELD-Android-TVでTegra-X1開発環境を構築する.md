---
layout: post
title: Jetson TX1の代わりにSHIELD Android TVでTegra X1開発環境を構築する
category: ハードウェア
tags:
- Chainer
- Tegra X1
- Adversarial Autoencoder
excerpt_separator: <!--more-->
---

## 概要

- [NVIDIA SHIELD Android TV](https://shield.nvidia.com/android-tv)にUbuntu 14.04を入れた
- Chainerを動かしてみた
- 2万円でJetson TX1とほぼ同等な開発環境を構築した

<!--more-->

## はじめに

半精度（fp16）で1TFLOPの演算性能があるモバイル向けGPUのNVIDIA [Tegra X1](http://www.nvidia.co.jp/object/tegra-x1-processor-jp.html)が昨年リリースされ、その開発環境として[Jetson TX1](http://www.nvidia.co.jp/object/embedded-systems-jp.html)が日本でも今年3月にようやく発売されました。

Jetson TX1はUbuntuをOSとして利用しており、インターフェースも多く自由度の高い開発が行えるのですが、米国価格$599が例によって日本円で10万円前後という簡単には手の出せない値段になっています。

（さらに米Amazonでは購入不可）

しかしTegra X1自体は他のNVIDIA製品にも組み込まれており、その中でもSHIELD Android TVは$199（米Amazonで23,000円で購入可能）と安価であり、USB3.0x2、micro USBx1、HDMIx1、Wifi、Bluetooth、Gigabit Ethernet、microSDカードスロット、とインターフェースもそれなりに多いハードなので、海外ではこれにUbuntuを入れてJetson TX1の代わりに使う方法が公開されています。

（ちなみにSATVは技適に通っており日本語に対応しています。なぜ日本で販売しないのかは不明です。）

### SHILEDの分解

見た目の5倍くらいの重量があり分解してみるとフレームが金属製でかなり頑丈な作りになっていました。

私は16GBモデルを購入したのでHDDのスペースが空いています。

![SHIELD](/images/post/2016-08-12/shield.jpg)

HDDをSSDに換装したい方はPro版を購入しましょう。

16GB版ではSATA→FFCケーブルやFFCコネクタが存在しません。

## 始める前に

現在日本語の情報は以下のPDFしかありませんので、おおまかな流れを確認しておきましょう。

[Shield Android TVはJetson TX1の代わりに使えるか?](https://demotomohiro.github.io/hardware/jetson_tk1/slides/ShieldTV%20Linux.pdf)

## SHIELDのWifiのMACアドレスを調べる

Android TVを起動しWifiに接続するとMACアドレスが見れるようになります。

後で使うため控えておきます。

## ホストPCにfastbootとadbを入れる。

ホストPCはUbuntuを使います。

fastbootやadbだけならMacも使えます。

[Android Studio](https://developer.android.com/studio/index.html)をインストールすると自動的にfastbootとadbがインストールされます。

## SHIELDをfastbootモードで起動する

SHILEDを起動し、Android TVのホーム画面からSettings→Device→Aboutへ移動し、Buildを選択したらコントローラのAボタンを8回連打すると開発者モードになります。

Settings→Preferences→Developer options→Debugging→USB debuggingをonにします。

ホスト側で

```
adb devices
```

を実行するとSHILEDに確認のダイアログが表示されるのでOKします。

ホスト側から

```
adb reboot bootloader
```

を実行するとSHIELDが再起動してfastboot menuが表示されます。

またfastboot menuはSHIELDの電源ボタンを操作して表示させることもできます。

- SHIELDの電源が切れた状態で電源ボタンに触れて指を離します。
- その後すぐに電源ボタンに触れ、今度はfastboot menuが出るまで触れ続けます。

## Linux for Tegra (L4T)のインストール

先ほどの[PDF](https://demotomohiro.github.io/hardware/jetson_tk1/slides/ShieldTV%20Linux.pdf)はL4T R23.1をインストールしていますが、現在L4T R24.1がリリースれています。

そこでこの記事ではR24.1のインストール方法を紹介します。

（現行スレッド：[Build kernel from source and boot to Ubuntu using L4T (Linux for Tegra) rootfs](http://forum.xda-developers.com/shield-tv/general/build-kernel-source-boot-to-ubuntu-t3274632)）

### 手順

ここからの操作は全てホスト側のUbuntuで行います。SHIELDはまだ使いません。

まず[Linux For Tegra R24.1](https://developer.nvidia.com/embedded/linux-tegra)から**Jetson TX1 64-bit Driver Package**と**64-bit Sample Root Filesystem**をダウンロードします。

microSDカードをext4でフォーマットします。

```
sudo mkfs.ext4 /dev/mmcblk0p1
```

Driver Packageを展開します。

この時Linux_for_Tegraというディレクトリができ、その中にrootfsというディレクトリがあります。

microSDカードをこのrootfsにマウントします。

```
sudo mount /dev/mmcblk0p1 Linux_for_Tegra/rootfs
```

rootfsの中にSample Root Filesystemを展開します。

```
cd Linux_for_Tegra/rootfs/
sudo tar xpf ../../Tegra_Linux_Sample-Root-Filesystem_R24.1.0_aarch64.tbz2
```

NVIDIAのドライバなどを書き込むスクリプトを実行します。

```
cd ../
sudo ./apply_binaries.sh
```

microSDカードをアンマウントします

```
sudo umount rootfs
```

このmicroSDカードをSHIELDにセットします。

### boot.imgの書き込み

R24.1のboot.imgを[ダウンロード](http://forum.xda-developers.com/showpost.php?p=67433510&postcount=249)します。

展開すると`satvL4t24`というディレクトリができます。

まずSHIELDをfastbootモードで起動し、fastboot menuを表示させます。

ホストPCとSHIELDをUSBケーブルで繋ぎます。（コントローラ用のケーブルを使います）

boot loaderをアンロックします。

```
fastboot oem unlock
```

この時SHIELDの画面が切り替わり赤字で警告が出ます。（出ない場合unlockをやり直します。）

Confirmを選択します。（この時SHIELDが初期化されます。）

アンロックできたら、先ほどダウンロードしたboot.imgをSHIELDに書き込みます。

```
cd satvL4t24
sudo fastboot flash boot boot.img
./flash-dtb.sh
```

ちなみにdtbを書き換えないと起動できませんので、`sudo fastboot boot boot.img`をする際は注意が必要です。

書き込みが完了したらfastboot menuを操作して一度SHIELDの電源を落とし再起動します。

boot領域にboot.imgを書き込んだので、SHIELDを起動すると自動的にUbuntuが立ち上がります。

初期ユーザー名はubuntu、パスワードはubuntuです。

Ubuntuが起動したら

```
sudo apt-mark hold xserver-xorg-core
```

を実行しNVIDIAのドライバが上書きされるのを防ぎます。

## Jetpack for L4TでCUDAをインストール

Jetpackを使うと簡単にCUDAやCUDA SamplesをSHIELDにインストールできます。

この時注意点としては（常識かもしれませんが）Jetpackはホスト側のUbuntuに入れ、SHIELDに直接入れるわけではありません。

JetpackはLANを通じてホスト側からSHIELDに必要なコンポーネントを転送しインストール操作を行うようになっています。

[Jetpack for L4Tのダウンロード](https://developer.nvidia.com/embedded/jetpack)

ダンロード後起動するとまず対象のJetsonを選択しますが、私は64bit版のL4Tを入れているのでJetson TX1(64-bit)を選びました。

次にインストールしたいコンポーネントを選択します。

Linux for Tegraはすでに入っているのでno actionにしておきます。

![jetpack](/images/post/2016-08-12/jetpack_1.png)

転送先のSHIELDのIPアドレスとユーザー名（Ubuntu）、パスワード（Ubuntu）を入れます。

![jetpack](/images/post/2016-08-12/jetpack_2.png)

転送が始まります。

私はL4TをSDカードに入れているため非常に時間がかかりました。

![jetpack](/images/post/2016-08-12/jetpack_3.png)

![jetpack](/images/post/2016-08-12/jetpack_4.png)

## CUDA Samplesを動かす

いつもどおりにビルドすると

```
libGL.so :`drmMap` に対する定義されていない参照です
.
.
.
```

というエラーが出てうまくいきません。

これは`/usr/lib/aarch64-linux-gnu/libGL.so`のリンク先が`mesa/libGL.so`になっているせいなので、

```
cd /usr/lib/aarch64-linux-gnu
sudo rm libGL.so
sudo ln -s tegra/libGL.so libGL.so
```

でシンボリックリンクを作り直します。

#### nbody

![nbody](/images/post/2016-08-12/cuda_nbody.png)

#### particles

![particles](/images/post/2016-08-12/cuda_particles.png)

#### smoke

![smoke](/images/post/2016-08-12/cuda_smoke.png)

## Chainerを入れる

まずpipを入れます

```
sudo apt-get install python-pip
sudo pip install -U pip
```

apt-getでpipを入れると古いバージョンのものが入るためアップデートしておきます。

L4Tに標準で入っているPythonは2.7.6なので、このまま`sudo pip install chainer`をしてもInsecurePlatformWarningが出ます。

ですのでpyenvを入れてPython 2.7.12をインストールします。

参考：[pyenvとvirtualenvで環境構築](http://qiita.com/Kodaira_/items/feadfef9add468e3a85b)

途中で.bash_profileを編集する部分がありますが今回は.bashrcに対して同様の編集を行います。

またエラーが大量に出るので以下を実行して再度Python 2.7.12を入れます。

```
sudo apt-get install libssl-dev libbz2-dev libreadline-dev libsqlite3-dev
pyenv install 2.7.12
pyenv global 2.7.12
sudo apt-get install libhdf5-dev
pip install chainer
```

pyenvが有効の時はsudo pipではなくpipで良いようです。

Chainerがインストールできたので[Adversarial AutoEncoder](https://github.com/musyoku/adversarial-autoencoder)を学習させてみました。

||Tegra X1|Geforce GTX 970M|
|1 epoch|3分|1分|

画像処理系の小さいモデルならデスクトップ版GPUとあまり差がない速度が出ました。

RNNなどの巨大なモデルではおそらくまともに学習できないんじゃないかと思います。

（そもそもメモリが3GBしかない）


## CUDA Samplesの比較

![nbody](/images/post/2016-08-12/970m_vs_x1.jpg)