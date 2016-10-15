---
layout: post
title: Deep Directed Generative Models with Energy-Based Probability Estimation [arXiv:1606.03439]
category: 論文
tags:
- HPYLM
- VPYLM
excerpt_separator: <!--more-->
---

## 概要

- Pitman-Yor言語モデルのハイパーパラメータの推定における式の詳細な導出について

<!--more-->

## はじめに

最近Deep LearningのRNN言語モデル（というかLSTM）が流行っていますが、私は[教師なし形態素解析](http://chasen.org/~daiti-m/paper/nl190segment.pdf)などにも応用できるベイズ階層言語モデルに注目し、過去に[階層Pitman-Yor言語モデル（HPYLM）](/2016/07/26/A_Hierarchical_Bayesian_Language_Model_based_on_Pitman-Yor_Processes/)や[可変長n-gram言語モデル（VPYLM）](/2016/07/28/Pitman-Yor%E9%81%8E%E7%A8%8B%E3%81%AB%E5%9F%BA%E3%81%A5%E3%81%8F%E5%8F%AF%E5%A4%89%E9%95%B7n-gram%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%87%E3%83%AB/)を実装してきました。

実装において２つのハイパーパラメータをデータから推定する必要があり、この部分は[Teh先生の論文](http://www.gatsby.ucl.ac.uk/~ywteh/research/compling/hpylm.pdf)に載っている更新式を使うのが慣例となっていますが、この式は何の説明もなく唐突に出てくるため、一体どのようにして導出したのかをこの記事でまとめます。