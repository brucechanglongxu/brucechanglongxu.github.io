---
title: "Intraoperative Absolute Depth Estimation in MVD Surgery"
collection: publications
category: conferences
permalink: /publication/mvd-depth-estimation/
excerpt: "This paper presents a real-time, monocular depth estimation framework for microvascular decompression (MVD) surgery that achieves sub-2mm accuracy without requiring stereo cameras or depth sensors. By integrating the Depth-Anything-V2 model with optical calibration from surgical microscopes, our method translates relative depth maps into absolute metric distances—enhancing spatial understanding of complex neurovascular anatomy. Validated against 3D reconstructions from preoperative MRI using 3D Slicer, this technique enables precise intraoperative localization of critical structures like cranial nerves and offending vessels. It represents a significant step toward AI-assisted neurosurgical navigation that is both low-cost and clinically practical. Future work will extend this system to other neurosurgical procedures and incorporate dynamic scene reconstruction."
date: 2025-06-20
venue: "38th IEEE CBMS International Symposium, Special Track on Surgery AI"
paperurl: 'http://brucechanglongxu.github.io/files/cbms_mvd_depth.pdf'
citation: "Jinhee Lee, Hwanhee Lee, Jay Park, Ethan Htun, Bruce Changlong Xu*, Sang Hoon Cho, Sanghoon Lee, and Vivek Buch. (2025). Intraoperative absolute depth estimation in MVD surgery. In Proceedings of the 38th IEEE International Symposium on Computer-Based Medical Systems (CBMS 2025), Special Track on Surgery AI."

---

This paper presents a method for absolute depth estimation during microvascular decompression (MVD) surgery using monocular surgical video. By calibrating depth predictions from the Depth-Anything-V2 model with focal measurements from a surgical microscope, the system estimates real-world distances between anatomical structures with sub-2mm accuracy. Validated against preoperative 3D MRI reconstructions, this technique enhances spatial understanding in complex neurovascular scenes and enables real-time intraoperative guidance without requiring stereo imaging or depth sensors.