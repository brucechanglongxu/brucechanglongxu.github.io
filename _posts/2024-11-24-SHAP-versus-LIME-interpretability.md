---
layout: post
title: "SHAP, LIME and Model Interpretability"
date: 2024-11-24
categories: Intrepretability
author: Bruce Changlong Xu
---

In the growing field of machine learning, interpretability has become as important as accuracy. Understanding **why** a model makes a particular prediction helps build trust, identify biases, and debug issues. Among the most popular tools for explaining machine learning models are **SHAP (Shapley Additive Explanations)** and **LIME (Local Intpretabile Model-Agnostic Explanations)**. Whilst both are used widely, their methodologies, strengths and weaknesses differ. 

LIME works primarily but creating a local surrogate model - a simplified, interpretable model trained on perturbations of the input data. This surrogate model mimics the predictions of the original black-box model in the local neighborhood of a specific prediction. It works at a high level as follows:

- We perturb the input data by slightly altering feature values
- We observe how these perturbations affect the model's prediction
- We fit a linear model to approximate the decision boundary locally
- We output feature importance scores that explain the model's prediction for the given input 

The strengths of this approach is that it is **model-agnostic** (works with any model) and **fast and lightweight**, allowing us to quickly provide explanations suitable for ad-hoc debugging; in addition to **human-readable explanations** where the results are intuitive and easy to understand. The weakness of this is that the linear surrogate may not accurately represent the true decision boundary of complex models, and could be unstable - results can vary depending on the perturbation or random seeds used. It also lacks global insights - LIME only provides explanations for a single prediction. 

**SHAP** is based on **Shapley values** from cooperative game theory, and provides a unified approach to interpreting machine learning models. It attributes a prediction to its features by distributing the prediction difference between the actual model and a baseline (e.g. average prediction) among the input features fairly. It works as follows:

- We treat each feature as a "player" in a cooperative game
- We calculate the marginal contributin of each feature by adding or removing it from subsets of features
- We use the Shapley value formula to fairly distribute contributions across features

If a feature's contribution increases in the model, its SHAP value will always reflect that increase. It satisfies key properties like additivity, and explains individual predictions (local) and overall feature importance (global). The drawback is that Shapley is computationally expensive (evaluating over all feature subsets), making it slow for large datasets, and may struggle with highly non-linear or interacting features. 

