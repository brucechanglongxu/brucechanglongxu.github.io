---
layout: post
title: "Training a simple BERT module on a single A10G"
date: 2024-11-23
categories: AI GPU
author: Bruce Changlong Xu
---

This post documents the process of training a simple BERT module on a single A10G GPU. There are minor difficulties and challenges that are used to manage GPU clusters that one must familiarize themselves with in order to truly work productively with AI. Recall, that in a typical AI workflow, we have two phases - "**training**" and "**inference**".

Our first order of business, is typically to ssh into a server with access to a compute cluster that we can run our workloads on. For instance, I can run the following command in my VScode terminal "ssh b{....}@b{.....}", which allows me to ssh into my desired server. From here, we should clone the github repo that we want to run (which contains our AI model infrastructure).  

