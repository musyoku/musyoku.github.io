---
layout: post
title: Linux 4 Tegraにv4l2loopbackモジュールを入れる時の注意点
category: ハードウェア
tags:
- Tegra X1
excerpt_separator: <!--more-->
---

## 概要

- 仮想ビデオデバイスを作るためのカーネルモジュール[v4l2loopback](https://github.com/umlaeute/v4l2loopback)を入れる

<!--more-->

## はじめに

[前回の記事](/2016/08/12/Jetson-TX1%E3%81%AE%E4%BB%A3%E3%82%8F%E3%82%8A%E3%81%ABSHIELD-Android-TV%E3%81%A7Tegra-X1%E9%96%8B%E7%99%BA%E7%92%B0%E5%A2%83%E3%82%92%E6%A7%8B%E7%AF%89%E3%81%99%E3%82%8B/)で2万円の[NVIDIA SHIELD Android TV](https://shield.nvidia.com/android-tv)を10万円の[Jetson TX1](http://www.nvidia.co.jp/object/embedded-systems-jp.html)の代わりに使ってTegra X1開発環境を整えました。

今回はUSBカメラモジュールをSHIELDで使う場合に必要になることもある**v4l2loopback**を入れる際に陥るエラーと対処法を紹介します。

## ビルド

```sudo apt-get install v4l2loopback-dkms```で一発インストールできるらしいのですが、L4T(Linux 4 Tegra)ではうまくいきませんでした。

さらにソースからビルドすると、

```
export ARCH=arm64
export CROSS_COMPILE=/usr/bin/aarch64-linux-gnu-
git clone https://github.com/umlaeute/v4l2loopback
cd v4l2loopback
make
```

以下のエラーが出ます

```
scripts/basic/fixdep: 1: scripts/basic/fixdep: Syntax error: "(" unexpected
```

解消するためには以下を実行します。

```
cd /usr/src/linux-headers-3.10.96-tegra/
sudo make modules_prepare
```

```3.10.96-tegra```の部分は環境によります。

これで```make```と```sudo make install```が通るので一見ビルドに成功したかに見えますが、

```
sudo modprobe v4l2loopback
```

すると

```
modprobe: ERROR: could not insert 'v4l2loopback': Exec format error 
```

と怒られます。

この原因はv4l2loopback.koのversion magicが（私の環境では）3.10.96になっていなければならないのに3.10.96-tegraになっているため不一致と判断されるからです。

（```insmod v4l2loopback```しても同様のエラーが出るので```dmesg```すると原因がわかります）

強制無視するオプションがあるのですがなぜかそれをしても全くエラーが消えませんでした。

そこで```/usr/src/linux-headers-3.10.96-tegra/include/generated/utsrelease.h```を開き、

```
#define UTS_RELEASE "3.10.96-tegra"
```

を

```
#define UTS_RELEASE "3.10.96"
```

に変え、もう一度```make```からやり直したところうまくいきました。

以下は[FLIR ONE](http://www.flir.jp/flirone/content/?id=62912)というスマホ用の熱画像センサモジュールをSHIELDで使ってみた例です。

![flie one](/images/post/2016-09-07/flir_one.jpg)