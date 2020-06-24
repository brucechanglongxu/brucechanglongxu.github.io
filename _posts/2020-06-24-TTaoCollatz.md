---
layout: post
title: A review of Terry Tao's recent attempt at the Collatz Conjecture
---

Refer to the following paper for context: https://arxiv.org/pdf/1909.03562.pdf

Professor Terence Tao recent published a huge effort to prove the infamous Collatz Conjecture, and this blog post will primarily be for me to parse through the arguments step by step to try and understand his underlying intuition and arguments, hence this is post is primarily for myself. Hopefully any future readers will potentially find it beneficial for their own mathematical development. This post will be less about the Mathematical nuances of his argument (which I would be hard-pressed to understand anyways) and more about my own intuitions on his flow of thought throughout the paper. 

We begin with an important notational setup: $$Col_{min}(N)$$ is equal to the minimum element of the set of iterations of $$S = \{Col^1(N), Col^2(N), ... \}$$. Note that existence of the minimum is guaranteed due to the fact that $S$ is a subset of the positive integers. Let us call all numbers $$N$$ such that $$Col_{min}(N) = 1$$ a "good" number. Indeed, our hope is to show that all natural numbers $$N$$ are "good" numbers. 

One of the first claims he makes in the paper draws upon an analytic result from Krasikov and Lagarias, which essentially deduces a very strong exponential lower bound on the number of "good" numbers in the interval $$[1, x]$$ for all sufficiently large $$x$$. 

Result 1: The number of 'good' numbers in the interval $$[1, x]$$ is significantly larger than $$x^{0.84}$$ for sufficiently large $$x$$. 

Definition 1: Professor Tao definition of "almost all" is based upon the notion of the random variable $$\log(R)$$ with the following distribution:

$$ P(\log(R) \in A) = \frac{\sum_{N \in A \cap R} \frac{1}{N}}{\sum_{N \in R} \frac{1}{N}} $$

The primary advancement that Professor Tao makes in his paper is as follows:

The $Col_{min}(N)$ function does not grow as fast as a function that approaches infinity following $N$. In other words, if $\lim_{N \to \infty} f(N) = \infty$ then we must have that $Col_{min}(N) < f(N)$. 

How we get to this result requires a tremendous amount of analytic horsepower, and hopefully by the end of this blog post you and I will have a clearer understanding of the magic behind his argument. 

This paper was beautiful. You know, if I wasn't so head over heels of the direct impact of medicine on the human condition, I would be determined to pursue a career in pure Mathematics. 
