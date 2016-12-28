---
layout: post
title: PythonでNPYLMによる教師なし形態素解析をする
category: 実装
tags:
- 自然言語処理
- NPYLM
excerpt_separator: <!--more-->
---

## 概要

- NPYLMのPythonラッパーを作った

<!--more-->

## はじめに

[前回NPYLMを実装](/2016/12/14/%E3%83%99%E3%82%A4%E3%82%BA%E9%9A%8E%E5%B1%A4%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AB%E3%82%88%E3%82%8B%E6%95%99%E5%B8%AB%E3%81%AA%E3%81%97%E5%BD%A2%E6%85%8B%E7%B4%A0%E8%A7%A3%E6%9E%90/)しましたが、今回はPythonで動かせるようにしました。

ただしNPYLMは特許が取られているためソースコードを公開することができません。

そこで共有ライブラリとしてビルドしたものをPythonから使います。

また特許の関係でモデルの保存機能も付けるとまずいので省いています。

## 実行する

まず[python-npylm](https://github.com/musyoku/python-npylm)を`git clone`しローカルに落とします。

次に[Boost 1.62](http://www.boost.org/users/history/version_1_62_0.html)をインストールします。

Macユーザーの方は`model.mac.so`を`model.so`にリネーム後、`make install`して`model.so`を修正します。

Ubuntuユーザーの方は`model.linux.so`を`model.so`にリネームするだけで動くと思います。

ただしUbuntu版は16.04（64-bit）のデスクトップとノートで動作確認しただけですので動かない可能性があります。

Mac版はLate 2012 i5とMid 2014 i7で動作確認しています。

## おわりに

今回のPythonラッパーはNPYLMを実際に動かしていみたい方向けのものになります。

特許の関係でNPYLMを実用する場合は自分で作るしかないため、[前回の記事](/2016/12/14/ベイズ階層言語モデルによる教師なし形態素解析/)を参考に実装してみてください。