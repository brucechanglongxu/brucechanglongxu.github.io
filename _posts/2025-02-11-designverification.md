---
layout: post
title: "Optimizing Llama memory footprint for inference"
date: 2025-02-11
categories: AI
author: Bruce Changlong Xu
---

The challenge of inferencing models with ever longer inputs (augmenting LLMs with a repository of domain specific data, and thereafter performing inference), leads to massive memory overhead due to caching key/value activations (KV cache). 

(*) Note that the reason **Query (Q) matrices aren't cached** while Key (K) and Value (V) matrices are in transformer-based models (such as those used in LLMs, retrieval-augmented generation (RAG), and search-based attention), comes down to the way attention mechanisms operate. In _autoregressive decoding_ the model generates one token at at time, and at each decoding step, a new query $$Q_t$$ is computed from the previous token. Since queries are derived from the current hidden state, they cannot be precomputed or stored for future reuse. The KV-cache on the otherhand stores the keys (K), and values (V) from previous tokens in the sequence. 