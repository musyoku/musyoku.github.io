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

<!--more-->

## はじめに

以前に[A Hierarchical Bayesian Language Model based on Pitman-Yor Processes](/2016/07/26/A_Hierarchical_Bayesian_Language_Model_based_on_Pitman-Yor_Processes/)と[Pitman-Yor過程に基づく可変長n-gram言語モデル](/2016/07/28/Pitman-Yor%E9%81%8E%E7%A8%8B%E3%81%AB%E5%9F%BA%E3%81%A5%E3%81%8F%E5%8F%AF%E5%A4%89%E9%95%B7n-gram%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%87%E3%83%AB/)の記事を書き、C++でVPYLMとHPYLMを実装しました。

しかしファイルの読み込み処理の書きやすさなどを考えるとPythonの方が便利というのもあり、今回はC++とPythonを連携させPythonからモデルを学習できるようにしました。

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

実行結果

```
2 ファイルを読み込み中 ...
文字種: 2915
行数: 899
VPYLMを初期化しています ...
VPYLMを読み込んでいます ...
G0 <- 0.000343
Epoch 1 / 100 - 8361.56 lps - 124.08 ppl - 5 depth - 2373 nodes - 40187 customers
Epoch 2 / 100 - 7085.21 lps - 96.82 ppl - 6 depth - 3689 nodes - 41979 customers
Epoch 3 / 100 - 6665.83 lps - 84.62 ppl - 8 depth - 4407 nodes - 43043 customers
Epoch 4 / 100 - 6049.07 lps - 76.93 ppl - 6 depth - 4867 nodes - 43869 customers
Epoch 5 / 100 - 5675.26 lps - 70.09 ppl - 6 depth - 5496 nodes - 44713 customers
Epoch 6 / 100 - 6260.92 lps - 65.71 ppl - 6 depth - 5714 nodes - 45290 customers
Epoch 7 / 100 - 5841.04 lps - 61.07 ppl - 6 depth - 6134 nodes - 46107 customers
Epoch 8 / 100 - 5920.36 lps - 57.25 ppl - 7 depth - 6525 nodes - 46869 customers
Epoch 9 / 100 - 5717.31 lps - 53.05 ppl - 7 depth - 6728 nodes - 47504 customers
Epoch 10 / 100 - 5585.48 lps - 49.10 ppl - 8 depth - 7255 nodes - 48512 customers
...
Epoch 90 / 100 - 3838.03 lps - 19.91 ppl - 7 depth - 13303 nodes - 65714 customers
Epoch 91 / 100 - 3815.90 lps - 19.83 ppl - 11 depth - 13390 nodes - 65549 customers
Epoch 92 / 100 - 3654.50 lps - 19.43 ppl - 8 depth - 13311 nodes - 65606 customers
Epoch 93 / 100 - 3834.65 lps - 19.49 ppl - 7 depth - 13283 nodes - 65674 customers
Epoch 94 / 100 - 3823.78 lps - 19.63 ppl - 7 depth - 13273 nodes - 65683 customers
Epoch 95 / 100 - 3692.37 lps - 19.41 ppl - 7 depth - 13428 nodes - 66017 customers
Epoch 96 / 100 - 3710.36 lps - 19.63 ppl - 7 depth - 13276 nodes - 65745 customers
Epoch 97 / 100 - 3841.90 lps - 19.11 ppl - 7 depth - 13379 nodes - 65946 customers
Epoch 98 / 100 - 3792.57 lps - 19.71 ppl - 7 depth - 13192 nodes - 65701 customers
Epoch 99 / 100 - 3771.37 lps - 19.37 ppl - 7 depth - 13519 nodes - 65898 customers
Epoch 100 / 100 - 3851.54 lps - 19.80 ppl - 8 depth - 13384 nodes - 65576 customers
```

文章生成例

```
Here the Queen , " and that you had been looking round her head , she found herself in a whisper . 
" No , I don't know , " added the March Hare said in a hurry ; " and most things ! " said the King . 
" But what am I to get out of a treacle from him to be a grin without a grin , and then the other side of the door — I do , and tried to say . " 
" But what am , it's a very poor hands , and after a few minutes . Alice led into its face . 
" But what am I to get out of the house if it had been broken glass table , but it doesn't matter which she had not attended to this , so that altogether , for her . She did not dare say you’re wondering why , that she was walking hand with the time they had to leave off this minute or two , they all returned from him to you never tasted — " I don't know what they’re about ! Who is , " said the King . 
" I don't know , " she said to herself , " I must be getting home ! " ( a loud , and Alice looked very anxiously fixed on it were nine feet high , and was going to leave off this minute or two , they all crowded round ! " said the King . 
For he can be seen such a curious dream of Wonderland , though , as they would go round her head , she was now about in the last few things being invited , " said the King . 
Soup of the evening , beautiful Soup ! 
Soo — oop of the e — e — evening , 
" But what am I to get hold of anything but it is you hate cats eat bats ? Do you think , " said the King . 
Alice was beginning to think that walk with an M , " and that you weren't to talk about her pet : " why , if I shall sit down upon their faces in their mouths . So they got settled down again , and went on without attending to her : its voice . 
" But what am I to do , and she tried to fancy , that : he came trotting along hand in hand , and Alice was so full effect : " but it had lost : she was to get out of the house down , I should like to be seen them , and she thought there was no one listening , and the Queen . 
" So she began singing a sort of lullaby 
" But what am , Alice noticed , had powdered hair that would have been changed into that lovely garden . ” " 
" A cat grins like a telescope ! I wouldn't say , " added the Queen . 
" Tell us ! ” “ Coming in a deep voice , and the Queen , " and that is — “ " 
```

オーダーの表示

```
<bos> " What did they live on ? " said Alice , who always took a great interest in questions of eating and drinking . 
  1   1   2   1    2    2   1 2 2   2    2   2  2     0     1  2   2       1     1     2      1    2    1      1    1 

<bos> Beau — ootiful Soo — oop ! 
  1     1  2    3     1  1  3  2 

<bos> " I don't believe it , " said the Pigeon ; " but if they do , why then they’re a kind of serpent , that's all I can say . " 
  1   1 2   3      1     1 3 1   3   2     2   1 4  3   1   1   1 0  0    1     2    1   2   1    2    2    2    1  2  1   1  1 2 

<bos> " Treacle , " said a sleepy voice behind her . 
  1   1    2    1 2   3  0    1     1      1    3  0 

<bos> And mentioned me to him : 
  1    1      3      1  2  2  3 

<bos> " I haven't the slightest idea , " said the Hatter . 
  1   1 2    2     1      3       1  2 2   3   3     2   1 

<bos> Beau — ootiful Soo — oop ! 
  1     1  2    3     1  1  2  2 
```

また推定された各n-gramオーダーのデータ全体の割合を表示できます。

```
 1-gram ####### 2914
 2-gram ############################## 12549
 3-gram ######################### 10405
 4-gram ######## 3018
 5-gram ## 698
 6-gram # 146
 7-gram # 28
 8-gram # 7
 9-gram # 2
```

## おわりに

私の最終目的はPythonでNPYLMを用いた教師なし形態素解析をすることで、HPYLMやVPYLMはその副産物です。

今回は、C++でメインの処理とインターフェースを作りPythonから利用する方法が知りたかったので練習として作成しました。

## 追記

今気づきましたがオーダーの推定値に0があるのはバグの可能性が高いのでコードを見直します。