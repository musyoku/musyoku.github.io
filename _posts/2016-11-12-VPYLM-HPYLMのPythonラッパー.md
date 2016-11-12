---
layout: post
title: VPYLM・HPYLMのPythonラッパー
category: 論文
tags:
- VPYLM
- HPYLM
excerpt_separator: <!--more-->
---

## 概要

- Pythonで階層ベイズ言語モデルを使いたい

## はじめに

以前に[A Hierarchical Bayesian Language Model based on Pitman-Yor Processes](/2016/07/26/A_Hierarchical_Bayesian_Language_Model_based_on_Pitman-Yor_Processes/)と[Pitman-Yor過程に基づく可変長n-gram言語モデル](/2016/07/28/Pitman-Yor%E9%81%8E%E7%A8%8B%E3%81%AB%E5%9F%BA%E3%81%A5%E3%81%8F%E5%8F%AF%E5%A4%89%E9%95%B7n-gram%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%87%E3%83%AB/)の記事を書き、C++でVPYLMとHPYLMを実装しました。

しかしファイルの読み込み処理の書きやすさなど、Pythonの方が便利で書きやすいというのもあり、今回はC++とPythonを連携させPythonからモデルを学習できるようにしました。

[vpylm-python](https://github.com/musyoku/vpylm-python)

## データの前処理

今回は[不思議の国のアリスの原作](https://www.gutenberg.org/files/11/11-h/11-h.htm)を訓練データとします。

単語の区切りを半角スペースで与えます。

ピリオドやコンマなどの記号も一つの単語とみなし、全てスペースで区切りました。

```
‘ Curiouser and curiouser ! ’ cried Alice ( she was so much surprised , 
that for the moment she quite forgot how to speak good English ) ; 
‘ now I’m opening out like the largest telescope that ever was ! Good-bye , feet ! ’ 
( for when she looked down at her feet , they seemed to be almost out of sight , 
they were getting so far off ) . 
‘ Oh , my poor little feet , I wonder who will put on your shoes and stockings for you now , dears ?
```

## HPYLM

まず始めにビルドします。

```
make python_hpylm
```

hpylm.soが作られます。

実行します。

```
python hpylm.py -t alice -n 3
```

`-t`で訓練用のテキストファイルが入っているディレクトリを指定します。

このディレクトリ内のすべてのテキストファイルが読み込まれます。

`-n`でHPYLMのn-gram長を指定します。


## VPYLM

```
make python_vpylm
python vpylm.py -t alice
```