---
layout: post
title: "Kinases"
date: 2024-12-29
categories: oncology
author: Bruce Changlong Xu
---

**"Studies on the [structures and functions of individual kinases](https://www.nature.com/articles/s41598-019-56499-4) have been used to understand the biological properties of other kinases that do not yet have experimental structures."** This is "inference by homology". A multiple sequence alignment is a bioinformatics method that is used to align three or more biological seuqences to identify regions of similarity, which can provide insight into evolutionary relationships, structural conservation and functional properties of sequences. Recall that protein kinases catalyze the transfer of a phosphoryl group from an ATP molecule to substrate proteins during cellular signaling. Kinase mutations that lead to gain of function are often observed in cancer, and kinase mutations lend resistance to existing drugs. Humans have over 500 genes that catalyze the phosphorylation of proteins, which are collectively called the 'kinome'. 

An MSA is able to identify _conserved regions_ across sequences, which often correspond to structurally or functionally important sites (e.g. ATP-binding sites). MSA's can also emphasize _gaps (insertions/deletions)_ where some sequences may have extra/missing nucleotides or amino acids which leads to gaps in alignment to maintain overall similarity. By analyzing aligned sequences, researchers can construct evolutionary trees to understand the relationships between genes/species. This allows us to infer secondary structures, enzyme active sites and even protein folding dynamics. 

The final output of a **Multiple Sequence Alignment** is essentially a tabular alignment of multiple sequences arranged horizontally, where each column represents a position in the alignment across all sequences. The sequences are adjusted by inserting gaps to ensure that similar residues/nucleotides/amino acids are aligned vertically (this can help tell us that a D in the DFG motif of kinases is highly conserved for instance). An example of a Multiple Sequence Alignment would be:

                                                | Sequence | Alignment   |
                                                |----------|-------------|
                                                | Seq 1    | MKT00PLTAV  |
                                                | Seq 2    | MKTAAALTAV  |
                                                | Seq 3    | MKTAA0LT0V  |
                                                | Seq 4    | MKTAAALTAV  |
                                                | Seq 5    | MK000ALTAV  |

Aligned letters within the same column suggest homology i.e. that these positions likely evolved from a common ancestor. Here, we use the number 0 to represents "gaps" in the sequence for homology alignment. A recent study constructed a high-accuracy MSA of 497 human protein kinase domains, integrating both sequence and structural alignment data. The methodology used