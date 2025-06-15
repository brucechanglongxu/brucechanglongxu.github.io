---
title: "Enhancing Surgeon-Machine Interface in Neurosurgery: Methodology for Efficient Annotation and Segmentation Model Development by Leveraging Embedding-Based Clustering Techniques"
collection: publications
category: manuscripts
permalink: /publication/smi-diversity-sampling
excerpt: "This paper introduces an embedding-based diversity sampling approach to reduce annotation burden in neurosurgical video segmentation, demonstrating performance improvements over random and similar sampling baselines."
date: 2025-05-30
venue: "19th World Congress of Neurosurgery (WFNS 2025), Dubai, United Arab Emirates"
#paperurl: 'http://academicpages.github.io/files/paper3.pdf'
citation: "Jay J. Park, Bruce Changlong Xu*, Josh Cho, Ethan Htun, Hwanhee Lee, Jungwoo Tomasz Jan Zaluska, Nehal Doiphode, John Y.K. Lee, Ehsan Adeli, and Vivek P. Buch. (2025). Enhancing Surgeon-Machine Interface in Neurosurgery: Methodology for Efficient Annotation and Segmentation Model Development by Leveraging Embedding-Based Clustering Techniques. In Proceedings of the 19th World Congress of Neurosurgery (WFNS 2025), Dubai, United Arab Emirates."
---

The emergence of computer vision in surgery is revolutionizing intraoperative analytics and anatomical understanding. Developing real-time, surgeon-assistive technologies—such as the Surgeon-Machine Interface (SMI)—requires extensive annotation of surgical video data, especially for instance segmentation. However, annotation is time-consuming, costly, and prone to bias. Embedding-based diversity sampling provides a pathway toward more sample-efficient model training and generalizable system design. 

We implemented diversity sampling by generating frame-level embeddings using a pre-trained CLIP ViT-B/32 model, capturing both visual and semantic cues. KMeans clustering was applied to these embeddings, with the optimal cluster count determined via the elbow method. From each cluster, the top 13 frames nearest to the cluster centroid were selected for annotation, ensuring broad coverage of visual diversity. For baseline comparison, a similar sampling strategy selected 169 frames closest to a single anchor frame in embedding space. Models were trained using YOLOv8 on a dataset of eight microvascular decompression (MVD) surgeries.

Diversity sampling outperformed both random and similar sampling baselines. It achieved an mAP@0.5 of 0.408 versus 0.286 (random) and demonstrated higher precision (up to 0.81) and recall (up to 0.42). The method yielded better generalization across rare visual scenes and resulted in more stable and faster convergence during training. Compared to similar sampling, it provided better coverage of the dataset’s semantic manifold, especially in underrepresented surgical views.

Embedding-based diversity sampling significantly enhances annotation efficiency and segmentation performance in surgical video datasets. This method advances foundational SMI infrastructure by offering a scalable and generalizable approach to dataset curation and training. Future work will extend this methodology across varied surgical domains and explore hybrid strategies incorporating clinician feedback and uncertainty estimation.