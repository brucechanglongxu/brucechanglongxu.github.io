---
layout: post
title: "AlphaGo Zero: Insights into Reinforcement Learning and Compute Efficiency"
date: 2024-11-22
categories: AI Reinforcement-Learning AlphaGo
author: Bruce Changlong Xu
---

## What is AlphaGo Zero?

AlphaGo Zero (AGZ) is a landmark in AI, showcasing how reinforcement learning (RL) can achieve superhuman performance in complex domains without human supervision. Unlike its predecessor, AlphaGo, which relied on human gameplay data, AGZ learns entirely from self-play, guided by:

- **Monte Carlo Tree Search (MCTS)**: Optimized by the Upper Confidence Tree (UCT) formula.
- **Residual Neural Networks**: Evaluating game states and predicting outcomes.

This case study dives into AGZ's algorithmic foundation and the compute characteristics that make it a compelling example of efficient AI design.

---

## Core Methodologies: Tree Search and Neural Networks

At the heart of AGZ lies the combination of tree search and neural network predictions. This synergy allows the agent to improve decision-making iteratively by exploring game trees and refining move probabilities.

### Key Innovations

1. **Tree Search Optimization**:
   - The UCT formula balances exploration (untried moves) and exploitation (promising moves):
     \[
     UCT(s, a) = \frac{W(s, a)}{N(s, a)} + c_{puct} \cdot \frac{P(s, a)}{\sum_b N(s, b)} \cdot \frac{1}{1 + N(s, a)}
     \]
     Where:
     - \( W(s, a) \): Cumulative value for action \( a \) at state \( s \).
     - \( N(s, a) \): Number of visits to action \( a \) at state \( s \).
     - \( P(s, a) \): Neural network’s prior probability of action \( a \) at state \( s \).
     - \( c_{puct} \): Hyperparameter balancing exploration and exploitation.

2. **Neural Network Integration**:
   - A **residual neural network** predicts:
     - **Move Scores** (\( \pi \)): Probabilities of promising moves.
     - **Game Outcomes** (\( v \)): Likelihood of winning from the current state.

Unlike AlphaGo, AGZ avoids full rollouts to terminal states, relying instead on neural network evaluations to backpropagate value estimates during tree search.

---

## Training Pipeline: A Three-Stage Workflow

AGZ’s training pipeline uses a recursive loop of self-play, supervised training, and evaluation:

### 1. Self-Play
AGZ generates training data by playing games against itself. Each move’s state, probabilities, and result are stored for training.

### 2. Supervised Training
The data collected from self-play is used to train the neural network to predict:
- **Move Scores** (\( \pi \)).
- **Game Outcomes** (\( v \)).

### 3. Evaluation
The newly trained model competes against the current best model. If the new model wins by a significant margin, it replaces the current best model, closing the loop.

---

## Compute Characteristics: Challenges and Patterns

The compute workload of AGZ spans both **GPU-intensive neural network operations** and **CPU-intensive tree search tasks**.

### GPU Workload: Neural Network Training and Inference
1. **Training**:
   - Each forward pass of the neural network for a 19x19 board requires **16 GFLOPs**.
   - Training a batch of 32 positions involves **48 GFLOPs** (forward, delta, and gradient passes).
   - **Efficient Kernels**: Winograd convolutions maximize GPU utilization during training by reducing FLOP overhead.

2. **Inference**:
   - During self-play and evaluation, the neural network evaluates up to 1600 positions per move.
   - Batch sizes are limited to 8, leading to lower GPU utilization compared to training.

### CPU Workload: Tree Search Operations
Tree search is a CPU-heavy task, requiring:
- UCT formula evaluations for each explored node.
- Updates to metrics (\( W, N, P \)) along the search path.

For a 19x19 board with **1600 lookahead searches per move**, the compute requirements per game are substantial:
- \( 1.2 \, \text{MFLOPS/move} \times 400 \, \text{moves/game} = 480 \, \text{MFLOPS/game} \).

### Bottlenecks
1. **Inference-Heavy Workload**:
   - Over 90% of the compute occurs during self-play and evaluation stages, dominated by neural network inference.
2. **Small Batch Sizes**:
   - Batch sizes are constrained to 8 during inference, limiting GPU utilization to ~30%.

---

## Algorithmic Insights and Takeaways

### 1. Efficient Balance of Exploration and Exploitation
The UCT formula ensures AGZ explores moves that might lead to long-term wins, rather than only short-term gains. This strategy led to groundbreaking moves, like move 37 in AlphaGo’s historic match against Lee Sedol.

### 2. Residual Networks as a Bottleneck
While residual networks are highly effective for move prediction, they dominate the compute workload. Optimizing these networks or reducing their precision could yield significant efficiency gains.

### 3. Reinforcement Learning Beyond Games
AGZ's RL-based approach to solving Go can extend to other domains where traditional supervised learning is infeasible. This underscores the potential of RL to learn strategies from scratch in complex environments.

---

## Implementation Notes

To deepen understanding, the AGZ algorithm was reimplemented from scratch using PyTorch. This provided valuable insights into:
- Training residual networks for both 9x9 and 19x19 boards.
- Benchmarking compute characteristics across different stages of the pipeline.

The reimplementation confirmed AGZ’s remarkable efficiency but highlighted opportunities for improvement, particularly in GPU utilization during inference.

---

## Conclusion

AlphaGo Zero remains a milestone in AI research, showcasing how reinforcement learning and neural networks can work together to achieve superhuman performance in complex, combinatorial domains. The lessons from AGZ extend far beyond Go, offering a blueprint for future AI systems in diverse fields.

Through careful design, AGZ achieves a balance between compute efficiency and performance, paving the way for scalable AI solutions without reliance on large labeled datasets.

---

**References**:
- Silver, David, et al. "Mastering the game of Go without human knowledge." *Nature* (2017).
- Minigo Codebase: [GitHub Link](https://github.com/tensorflow/minigo)
- Upper Confidence Trees: A foundational algorithm for game search.

---
