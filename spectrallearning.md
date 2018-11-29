# Annotated Bibliography for Spectral Learning

Hannah Lawrence
Description

## Tutorials
? Also code?

## Papers

### Foundational
[Spectral Learning](https://people.eecs.berkeley.edu/~klein/papers/spectral-learning.pdf) IJCAI '03

Sepandar D. Kamvar, Dan Klein, and Christopher D. Manning

Summary

### Word Embeddings
[Eigenwords: Spectral Word Embeddings](http://jmlr.csail.mit.edu/papers/volume16/dhillon15a/dhillon15a.pdf) JMLR '15

Paramveer S. Dhillon, Dean P. Foster, Lyle H. Ungar

[Model-Based Word Embeddings from Decompositions of Count Matrices](http://www.cs.columbia.edu/~djhsu/papers/count_words.pdf) ACL '15

Karl Stratos, Michael Collins, Daniel Hsu

[Spectral Graph-Based Method of Multimodal Word Embedding](http://www.aclweb.org/anthology/W17-2405) ACL '17

Kazuki Fukui, Takamasa Oshikiri, Hidetoshi Shimodaira

[Continuous Word Embedding Fusion via Spectral Decomposition](http://aclweb.org/anthology/K18-1002) CoNLL '18

Tianfan Fu, Cheng Zhang, Stephan Mandt

### Hidden Markov Models

[A Spectral Algorithm for Learning Hidden Markov Models](https://arxiv.org/pdf/0811.4413) JCSS '09

[Spectral Dimensionality Reduction for HMMs](https://arxiv.org/pdf/1203.6130.pdf) '12

Dean P. Foster, Jordan Rodu, Lyle H. Ungar


[Spectral Learning of Mixture of Hidden Markov Models](https://paris.cs.illinois.edu/pubs/subakan-nips2014.pdf) NIPS '14

Y. Cem Subakan, Johannes Traa, Paris Smaragdis

[Unsupervised Part-of-Speech Tagging with Anchor Hidden Markov Models](http://www.aclweb.org/anthology/Q16-1018) ACL '16

Karl Stratos, Michael Collins, Daniel Hsu


### Dependency Parsing and PCFGs

[Spectral Dependency Parsing with Latent Variables](http://www.pdhillon.com/spectral-dep-parsing.pdf) EMNLP '12

Paramveer S. Dhillon, Jordan Rodu, Michael Collins, Dean P. Foster, Lyle H. Ungar

[Diversity in Spectral Learning for Natural Language Processing](http://www.aclweb.org/anthology/D15-1214) EMNLP '15

Shashi Narayan and Shay B. Cohen

[Optimizing Spectral Learning for Parsing](http://www.aclweb.org/anthology/P16-1146) ACL '16

Shashi Narayan and Shay B. Cohen

The authors build on existing spectral algorithms for natural language parsing, improving on previous experimental results. They present a search algorithm akin to beam search which optimizes the number of latent states per non-terminal, as the number of latent states output by spectral algorithms is usually the number of non-zero singular values of a inside-outside tree covariance matrix. However, it often suffers from errors - both due to noisy estimation of variables from data, and estimation errors in the algorithm itself - even capping the number of latent states arbitrarily. Instead, this paper uses a search algorithm to intelligently fix the number of latent states for each nonterminal. It works with any spectral algorithm, and remains efficient as the spectral algorithms are efficient. They test on French, German, Hebrew, Hungarian, Korean, Polish, Swedish, and Basque, and outperform the Berkeley parser for many languages. 

[Spectral Learning of Latent-Variable PCFGs: ALgorithms and Sample Complexity](http://jmlr.org/papers/volume15/cohen14a/cohen14a.pdf) JMLR '14

Shay B. Cohen, Karl Stratos, Michael Collins, Dean P. Foster, Lyle Ungar

[Experiments with Spectral Learning of Latent Variable PCFGs](http://www.cs.columbia.edu/~scohen/naacl13spectral.pdf) NAACL '13

Shay B. Cohen, Karl Stratos, Michael Collins, Dean P. Foster, Lyle Ungar


### Topic Modeling

[A Spectral Algorithm for Latent Dirichlet Allocation](https://arxiv.org/pdf/1204.6703.pdf) NIPS '12

Animashree Anandkumar, Dean P. Foster, Daniel Hsu, Sham M. Kakade, Yi-Kai Liu

This was among the first papers to present a spectral algorithm for Latent Dirichlet Allocation, used in topic modeling. The model only requires trigram statistics, and uses spectral decomposition of low order moments (simple SVDs). The SVDs are of size proportional to the number of latent factors, and thus are efficient. In addition, they provide theoretical guarantees for sample complexity and parameter recovery via this method, which they call Excess Correlation Analysis. Note that there is little empirical testing in this paper, aside from a dataset of New York Times articles. 

[Spectral Learning for Supervised Topic Models](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7879308&tag=1) IEEE '17

Yong Ren, Yining Wang, and Jun Zhu

[Tensor Decompositions for Learning Latent Variable Models](http://jmlr.org/papers/volume15/anandkumar14b/anandkumar14b.pdf) JMLR '14

Animashree Anandkumar, Rong Ge, Daniel Hsu, Sham M. Kakade, Matus Telgarsky

This is another foundational paper in spectral learning. Similar to an earlier paper specific to Latent Dirichlet Allocation, their approach is to estimate low-order observable moments, and retrieve model parameters by a symmetric tensor decomposition (a generalization of SVD). To address the difficulty of tensor decompositions in general, they apply and analyze the robust tensor power method, resulting in a scalable latent variable estimation method with robustness to noise. Their method applies not only to LDA, but also to Hidden Markov and Gaussian Mixture models.

[Spectral Leader: Online Spectral Learning for Single Topic Models](https://arxiv.org/pdf/1709.07172.pdf) ECML-PKDD '18

Tong Yu, Branislav Kveton, Zheng Wen, Hung Bui, Ole J. Mengshoel

The authors develop a spectral learning algorithm for online topic modeling, SpectralLeader, as a competitor to online Expectation-Maximization. In this setup, one must learn a latent variable model form an ongoing stream of data. In this setup, the distribution of topics may change over time, but the conditional distribution of words is unchanging. At each timestep, the algorithm observes a new document sampled iid from the model at that time; the goal is to predict a sequence of model parameters with low cumulative regret. In contrast to EM, which may get stuck at local optima, SpectralLeader provably converges to the global optimum. The authors derive regret bounds, and test it on synthetic data as well as news and Twitter data, and show that SpectralLeader is competitive with, or in some cases superior to, EM.












### Spectral Clustering
### Miscellaneous
