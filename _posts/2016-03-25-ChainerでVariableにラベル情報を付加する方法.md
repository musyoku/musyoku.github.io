---
layout: post
title: ChainerでVariableにラベル情報を付加する方法
category: Chainer
tags:
- Chainer
- 実装
excerpt_separator: <!--more-->
---

## 概要

- n次元ベクトルのVariableにone-hotなラベルベクトルのVariableを付加する方法

<!--more-->

## はじめに

[Adversarial Autoencoder](/2016/02/22/adversarial-autoencoder/)では教師あり学習の際、隠れ変数ベクトルにラベルベクトルを付加して次元拡張し、新たなベクトルを作ります。

たとえば隠れ変数がn次元ベクトル、ラベル情報がm次元ベクトルだった場合、新しいベクトルはn+m次元になります。

Variableに変換する前のnumpy配列であれば、numpy.appendすればいいだけなのですが、今回はラベルを付加する場所がネットワークの中間層なため、Chainerで適切に誤差逆伝播できるような付加手法が必要になりました。

### 例

```
x = [0.5, 0.7]
label = [0, 0, 1, 0, 0, 0, 0]
```

```
new_x = [0.5, 0.7, 0, 0, 1, 0, 0, 0, 0]
```

## 結合関数

今回は２つのベクトルを結合するChainerの関数を作ります。

コードは以下のようになりました。

```
class Adder(function.Function):
	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(n_in == 2)
		x_type, label_type = in_types

		type_check.expect(
			x_type.dtype == np.float32,
			label_type.dtype == np.float32,
			x_type.ndim == 2,
			label_type.ndim == 2,
		)

	def forward(self, inputs):
		xp = cuda.get_array_module(inputs[0])
		x, label = inputs
		n_batch = x.shape[0]
		output = xp.empty((n_batch, x.shape[1] + label.shape[1]), dtype=xp.float32)
		output[:,:x.shape[1]] = x
		output[:,x.shape[1]:] = label
		return output,

	def backward(self, inputs, grad_outputs):
		x, label = inputs
		return grad_outputs[0][:,:x.shape[1]], grad_outputs[0][:,x.shape[1]:]

def add_label(x, label):
	return Adder()(x, label)
```

Variableを入力すると拡張されたVariableが返ってきます。

```
new_variable = add_label(x_variable, label_variable)
```

backward時、ラベル情報は定数なので勾配を伝播する必要はないのですが、入力の順番が入れ替わっても対応できるようにするため余計なことはしないようにしました。

```
new_variable = add_label(x_variable, label_variable)
new_variable = add_label(label_variable, x_variable)	# 本質的には同じ
```