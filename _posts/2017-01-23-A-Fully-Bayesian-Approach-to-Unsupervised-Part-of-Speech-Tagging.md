---
layout: post
title: A Fully Bayesian Approach to Unsupervised Part-of-Speech Tagging
category: 実装
tags:
- 自然言語処理
excerpt_separator: <!--more-->
---

## 概要

- [A Fully Bayesian Approach to Unsupervised Part-of-Speech Tagging](http://homepages.inf.ed.ac.uk/sgwater/papers/acl07-bhmm.pdf)を読んだ
- C++で実装しPythonのラッパーを作った
- 日本語・英語のデータセットで教師なし品詞推定の実験を行った

<!--more-->

## はじめに

$$
	\begin{align}
		P(t=1 \mid \boldsymbol w)&=\frac{P(\boldsymbol w \mid t=1)P(t=1)}{P(\boldsymbol w)}\\
		&=\frac{P(\boldsymbol w \mid t=1)P(t=1)}{\sum_tP(\boldsymbol w \mid t)P(t)}\\
		&=\frac{P(\boldsymbol w \mid t=1)P(t=1)}{P(\boldsymbol w \mid t=0)P(t=0) + P(\boldsymbol w \mid t=1)P(t=1)}\\
	\end{align}\
$$

$$
	\begin{align}
		P(t=0) = P(t=1) = 0.5
	\end{align}\
$$

より

$$
	\begin{align}
		P(t=1 \mid \boldsymbol w)=\frac{P(\boldsymbol w \mid t=1)}{P(\boldsymbol w \mid t=0) + P(\boldsymbol w \mid t=1)}\\
	\end{align}\
$$

ここで

$$
	\begin{align}
		P(\boldsymbol w \mid t=0) &= \int_0^1 P(\boldsymbol w \mid t=0, \theta)P(\theta)d\theta\\
		P(\boldsymbol w \mid t=1) &= \int_0^1 P(\boldsymbol w \mid t=1, \theta)P(\theta)d\theta\\
		P(\boldsymbol w \mid t=1, \theta) &= \theta^{n_H}(1-\theta)^{n_T}\\
		P(\boldsymbol w \mid t=0, \theta) &= \frac{1}{2}^{n_H}\frac{1}{2}^{n_T} = \frac{1}{2}^{10}\\
		P(\theta) &= 1
	\end{align}\
$$

$$
	\begin{align}
		B(x, y) &= \int_0^1 t^{x-1}(1-t)^{y-1}dt\\
				&= \frac{\Gamma(x)\Gamma(y)}{\Gamma(x+y)}\\
		\Gamma(z + 1) &= z\Gamma(z)\\
					&= z!\\
	\end{align}\
$$

$$
	\begin{align}
		P(\boldsymbol w \mid t=1) &= \int_0^1 P(\boldsymbol w \mid t=1, \theta)P(\theta)d\theta\\
		&= \int_0^1 \theta^{n_H}(1-\theta)^{n_T}d\theta\\
		&= B(n_H+1, n_T+1)\\
		&= \frac{\Gamma(n_H+1)\Gamma(n_T+1)}{\Gamma(n_H+n_T+2)}\\
		&= \frac{n_H!n_T!}{(n_H+n_T+1)!}\\
		&= \frac{n_H!n_T!}{11!}\\
	\end{align}\
$$


$$
	\begin{align}
		P(t=1 \mid \boldsymbol w) &= \frac{\frac{n_H!n_T!}{11!}}{\frac{n_H!n_T!}{11!}+\frac{1}{2}^{10}}\\
		&=1/\left(1+\frac{11!}{n_H!n_T!2^{10}}\right)\\
	\end{align}\
$$

$$
	\begin{align}
		P(x = k \mid \boldsymbol x_{-1}, \beta) &= \int P(k \mid \boldsymbol \theta)P(\boldsymbol \theta \mid \boldsymbol x_{-1}, \beta) d\boldsymbol \theta\\
		&= \int \theta_k\frac{P(\boldsymbol x_{-1} \mid \boldsymbol \theta, \beta)P(\boldsymbol \theta)}{P(\boldsymbol x_{-1})}d\boldsymbol \theta\\
		&= \int \theta_k\theta_1^{n_1}\cdot\cdot\cdot\theta_k^{n_k}\cdot\cdot\cdot\theta_K^{n_K}\frac{P(\boldsymbol \theta)}{P(\boldsymbol x_{-1})}d\boldsymbol \theta\\
		&= \int \theta_1^{n_1}\cdot\cdot\cdot\theta_k^{n_k+1}\cdot\cdot\cdot\theta_K^{n_K}\frac{P(\boldsymbol \theta)}{P(\boldsymbol x_{-1})}d\boldsymbol \theta\\
	\end{align}\
$$

$$
	\begin{align}
		P(\boldsymbol \theta) &= {\rm Dir}(\beta_1, ..., \beta_k)\\
		&= \frac{\Gamma(\beta_1+\cdot\cdot\cdot+\beta_k)}{\Gamma(\beta_1)\cdot\cdot\cdot\Gamma(\beta_k)}\theta_1^{\beta_1-1}\cdot\cdot\cdot\theta_k^{\beta_k-1}\\
		&= \frac{\Gamma(K\beta)}{\Gamma(\beta)^K}\theta_1^{\beta-1}\cdot\cdot\cdot\theta_k^{\beta-1}
	\end{align}\
$$

$$
	\begin{align}
		\sum_{k=1}^{K} \theta_k = 1
	\end{align}\
$$

$$
	\begin{align}
		\int \theta_1^{\alpha_1-1}\cdot\cdot\cdot\theta_k^{\alpha_k-1}d\boldsymbol \theta = \frac{\Gamma(\alpha_1)\cdot\cdot\cdot\Gamma(\alpha_k)}{\Gamma(\alpha_1+\cdot\cdot\cdot+\alpha_k)}
	\end{align}\
$$

$$
	\begin{align}
		P(\boldsymbol x_{-1}) &= \int P(\boldsymbol x_{-1} \mid \boldsymbol \theta)P(\boldsymbol \theta)d\boldsymbol \theta\\
		&=\int \theta_1^{n_1}\cdot\cdot\cdot\theta_k^{n_k}\frac{\Gamma(K\beta)}{\Gamma(\beta)^K}\theta_1^{\beta-1}\cdot\cdot\cdot\theta_k^{\beta-1}\\
		&= \frac{\Gamma(K\beta)}{\Gamma(\beta)^K} \int \theta_1^{n_1 + \beta - 1}\cdot\cdot\cdot\theta_k^{n_k + \beta - 1}\\
		&= \frac{\Gamma(K\beta)\Gamma(n_1 + \beta)\cdot\cdot\cdot\Gamma(n_k + \beta)}{\Gamma(\beta)^K\Gamma(i-1+K\beta)}\\
		i &= n_1+\cdot\cdot\cdot n_k-1
	\end{align}\
$$



$$
	\begin{align}
		P(x = k \mid \boldsymbol x_{-1}, \beta) &= \int \theta_1^{n_1}\cdot\cdot\cdot\theta_k^{n_k+1}\cdot\cdot\cdot\theta_K^{n_K}\frac{P(\boldsymbol \theta)}{P(\boldsymbol x_{-1})}d\boldsymbol \theta\\
		&= \int \theta_1^{n_1}\cdot\cdot\cdot\theta_k^{n_k+1}\cdot\cdot\cdot\theta_K^{n_K}
		\frac{\Gamma(K\beta)}{\Gamma(\beta)^K}\theta_1^{\beta-1}\cdot\cdot\cdot\theta_k^{\beta-1}
		\frac{\Gamma(\beta)^K\Gamma(i-1+K\beta)}{\Gamma(K\beta)\Gamma(n_1 + \beta)\cdot\cdot\cdot\Gamma(n_K + \beta)}
		d\boldsymbol \theta\\
		&= \frac{\Gamma(i-1+K\beta)}{\Gamma(n_1 + \beta)\cdot\cdot\cdot\Gamma(n_K + \beta)}
		\int \theta_1^{n_1+\beta-1}\cdot\cdot\cdot\theta_k^{n_k+\beta}\cdot\cdot\cdot\theta_K^{n_K+\beta-1}d\boldsymbol \theta\\
		&= \frac{\Gamma(i-1+K\beta)}{\Gamma(n_1 + \beta)\cdot\cdot\cdot\Gamma(n_K + \beta)}
		\frac{\Gamma(n_1 + \beta)\cdot\cdot\cdot\Gamma(n_k + \beta + 1)\cdot\cdot\cdot\Gamma(n_K + \beta)}{\Gamma(i+K\beta)}\\
		&= \frac{\Gamma(i-1+K\beta)}{\Gamma(n_k + \beta)}\frac{\Gamma(n_k + \beta + 1)}{\Gamma(i+K\beta)}\\
		&= \frac{\Gamma(i-1+K\beta)}{\Gamma(n_k + \beta)}\frac{(n_k + \beta)\Gamma(n_k + \beta)}{(i-1+K\beta)\Gamma(i-1+K\beta)}\\
		&= \frac{n_k+\beta}{i-1+K\beta}
	\end{align}\
$$