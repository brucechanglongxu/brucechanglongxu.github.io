---
layout: post
title: "AlphaGo, DeepSeek and Deep Reinforcement Learning"
date: 2024-12-21
categories: RL
author: Bruce Changlong Xu
---

For centuries, the ancient board game of Go was considered the ultimate test of human intuition and strategic thinking, far beyond the reach of artificial intelligence. In 2016, DeepMind’s AlphaGo shattered this belief by defeating world champion Lee Sedol. But AlphaGo relied on extensive training with human games, leaving an open question: Can AI achieve superhuman performance without any human input? DeepMind’s groundbreaking paper, Mastering the Game of Go without Human Knowledge, introduced AlphaGo Zero, an AI that learned to master Go tabula rasa—from scratch—using only the game’s basic rules. Unlike its predecessors, AlphaGo Zero received no human data, no expert games, and no handcrafted features. Instead, it trained itself entirely through reinforcement learning and self-play, evolving strategies more powerful than any human-derived techniques. In just three days, AlphaGo Zero surpassed all previous versions of AlphaGo and ultimately defeated AlphaGo Lee 100-0.

"The more we take out specialized knowledge, the more we... everytime we specialize something, we actually hurt our generalizability." 
"Scaling up trial and error interaction" 


## Deep Reinforcement Learning

AlphaGo Zero's breakthrough was made possible by deep reinforcement learning (RL), a paradigm in which an AI learns by interacting with an environment and improving based on feedback. Instead of mimicking human gameplay, AlphaGo Zero trained itself purely through self-play using the following techniques:

1. **Monte Carlo Tree Search:** AlphaGo Zero combines deep learning with search by using MCTS to evaluate moves more effectively than brute-force methods. The MCTS process balances exploration and exploitation to refine move selection over time Instead of conducting exhaustive rollouts, AlphaGo Zero leverages neural network evaluations to improve search efficiency, ensuring more informed move selection.
2. **Policy and Value Networks:** A deep residual neural network predicts the best moves (policy network) and evaluates board positions (value network). These networks evolve over time as AlphaGo Zero optimizes its predictions through gradient descent and experience replay. The architecture features multiple residual blocks, allowing for efficient feature extraction and learning from raw board states without handcrafted features.
3. **Self-Play with Reinforcement Learning:** AlphaGo Zero generates its own training data by playing against itself, continuously improving its strategies. The AI assigns probabilities to moves based on prior training and refines them through iterative learning. Through self-play, the model undergoes a continual process of policy improvement, exploring novel strategies that would be unlikely to emerge in human-guided training.

AlphaGo Zero exemplifies the power of reinforcement learning and self-play, achieving superhuman performance without human guidance. This represents a paradigm shift in AI development, moving towards more general and self-sufficient learning systems that can surpass human expertise in a variety of domains.

- Silver, D., Schrittwieser, J., Simonyan, K. et al. Mastering the game of Go without human knowledge. Nature 550, 354–359 (2017). https://doi.org/10.1038/nature24270
- Silver, D., Huang, A., Maddison, C. et al. Mastering the game of Go with deep neural networks and tree search. Nature 529, 484–489 (2016). https://doi.org/10.1038/nature16961

