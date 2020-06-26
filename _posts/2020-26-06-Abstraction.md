---
layout: post
title: Layers of Abstraction
---

As I am working on the CS107 final project heap allocator, I think to myself the importance of finding the right balance between understanding an issue and being able 
to address its needs adequately. A lot of the time we are faced with this gigantic task - go build something, go find something, do something with the resources you have 
in your hands right now. There are two ways you can look at this: Either seek for guidance and mentorship from some higher level of authority on how to do the job, or 
get started with your current knowledge and abilities - there is a subtle difference between learning by doing and learning to do. Whilst the former creates some form of 
output that can be appreciated or used by others, the latter, whilst enriching for the learner, offers little immediate reward to society as a whole. 

Diving into existing code bases that can be unreadable at times is extremely scary, yet one of the most rewarding activities that an aspiring computer scientist can hope to do. It's a little frightening to invest time and energy into a project that may not pan out well, but that is exactly why you should have several backup projects lined up and a strong motive that does not depend on the project succeeding. This is why the focus should be on learning and growing, not current results. 

It is important to be thrown into an environment that challenges your status quo, imperceptible growth is thrust upon you and often-times unique contributions to 
the current body of human knowledge arise as a result. Atul Gawande, famous surgeon-writer began his penmanship journey as an outlet for the feelings, emotions and unique experiences
that he has encountered in the operating room. As a result of continual tension and reflection, he was able to craft a unique perspective with which he could offer the broader medical community.

Hence I urge any readers, find a problem or issue you care about, throw yourself in the deep end and try and solve it with the resources you have. Maybe you will learn a thing or two along the way! 

Addendum (Reference to B&O's Computer Organizations and Systems Textbook): 

An analogy to understand the idea of "peak utilization" in computer science. The setup is as follows:

Given a sequence of free and myalloc requests: $$R_0, R_1, ..., R_{n-1}$$, we define the "aggregate payload" as follows: $$P_k = R_0 + R_1 + ... + R_k$$. At some point in our $$n$$ free and allocate requests, there exists a maximum aggregate payload (this is due to the fact that not every memory location on the heap is necessarily allocated). Subsequently, we define peak utilization to be:

$$U_k = \frac{\max_i \le k P_i}{H_k}$$ 

An analogous phenomenon in everyday life is to track an individual's baseline aerobic fitness over the course of the year. There will be times of laziness / unproductivity and moments of productive training / health, which correspond to the operations free() and mymalloc() respectively. To maximize utilization means to allocate as much memory as possible over as short a period of time, whilst to m
