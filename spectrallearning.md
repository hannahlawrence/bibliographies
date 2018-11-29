# Annotated Bibliography for Spectral Learning in NLP

Hannah Lawrence
Description, disclaimers (focus on NLP; many papers outside too, esp. clustering)

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

In this paper, the authors provide an extensive development of four spectral algorithms for word embeddings, including novel algorithms, theoretical analysis, and experimental results. Their first algorithm uses a single step of CCA between words and their contexts to achieve embeddings; the second uses two CCAs (between left and right contexts, and then with the words); the third and fourth algorithms reduce the dimension of context vectors using word embeddings themselves, iteratively updating the embeddings. This provides considerable speedup. The authors provide theoretical (sample, accuracy) guarantees for algorithms and test on a variety of downstream NLP tasks, including POs tagging, word similarity, sentiment classification, named entity recognition, and chunking, and achieve results comparable to SOTA.

[Model-Based Word Embeddings from Decompositions of Count Matrices](http://www.cs.columbia.edu/~djhsu/papers/count_words.pdf) ACL '15

Karl Stratos, Michael Collins, Daniel Hsu

This paper presents a first interpreretation of canonical correlation analysis for generating word embeddings, using a one-hot word vector as one view and the context as another view. Their algorithms use co=occurrence counts and low-rank SVD steps, but leave flexibility for generation of a word-context matrix (e.g. from PPMI, as in word2vec, or CCA). They evaluate on word similarity and analogy tasks, in which the spectral algorithms do well, but only outperform the skip-gram models on certain cases in 500 and 1000 dimensions. 

[Spectral Graph-Based Method of Multimodal Word Embedding](http://www.aclweb.org/anthology/W17-2405) ACL '17

Kazuki Fukui, Takamasa Oshikiri, Hidetoshi Shimodaira

This recent paper explores generation of word embeddings using, in addition to the usual corpus context information, images associated with words in the corpus. Their method, "Multimodal Eigenwords," uses both context and image information, employing a variant of CCA and spectral graph embeddings. They evaluate their representations on word similarity and concept-to-image search, outperforming skip-gram and eigenwords in 5 out of 7 tasks.

[Continuous Word Embedding Fusion via Spectral Decomposition](http://aclweb.org/anthology/K18-1002) CoNLL '18

Tianfan Fu, Cheng Zhang, Stephan Mandt

This paper considers a realistic variant of word embedding problems: suppose you have a large body of pre-trained word embeddings available, but need word embeddings for a specialized set of vocabulary not included, for which you only have a small corpus. Your corpus includes normal words as well; you'd like to obtain embeddings for the specialized vocabulary, which are coherent with embeddings for the standard vocabulary. In this paper, the authors use the established view of skipgram word embeddings as factoring a shifted positive pointwise mutual information matrix to formulate this problem as an application of online SVD. They show how to (1) obtain an SVD from the low-rank factorization provided in existing word embeddings, (2) estimate the full matrix, given new word embeddings, and (3) efficiently find the SVD of the extended matrix via online SVD. In tests, their algorithm is very efficient and improves over baselines.

### Hidden Markov Models

[A Spectral Algorithm for Learning Hidden Markov Models](https://arxiv.org/pdf/0811.4413) JCSS '09

Daniel Hsu, Sham M. Kakade, Tong Zhang

This paper is foundational, not just for learning Hidden Markov Models, but for spectral learning more generally; the methods defined here inspired algorithms in later papers for topic modeling, dependency parsing, etc. In this paper, the authors present the first spectral algorithm for learning Hidden Markov Models, under certain assumptions such as separability: the observation distributions arising from hidden states must be distinct, for example. The algorithm has polynomial sample and computational complexity, and consists of a SVD between past and future operations (which can be interpreted as an application of Canonical Correlation Analaysis). The algorithm does not explicitly learn the model parameters, but learns a linear function of them which can be used to compute joint probabilities. They provide theoretical guarantees for this algorithm, taking into account estimation error.

[Spectral Dimensionality Reduction for HMMs](https://arxiv.org/pdf/1203.6130.pdf) '12

Dean P. Foster, Jordan Rodu, Lyle H. Ungar

As in the previous paper, there is a fast spectral method based on co-occurrence of pairs and triples for learning HMMs, which is much faster than EM or Gibbs sampling. In this paper, the authors present a similar spectral method which improves upon the previous parameter and sample complexity: it reduces the number of model parameters that must be estimated, and the sample complexity required does not depend on the size of the observation vocabulary. They do so by reducing the dimension of intermediate per-emission matrices, allowing for similar computation of the probabilities of emission sequences. They provide the usual sample and accuracy guarantees, with accuracy measured as a ratio of probabilities of sequences, although they do not empirically test this modified HMM algorithm on NLP tasks.

[Spectral Learning of Mixture of Hidden Markov Models](https://paris.cs.illinois.edu/pubs/subakan-nips2014.pdf) NIPS '14

Y. Cem Subakan, Johannes Traa, Paris Smaragdis

[Unsupervised Part-of-Speech Tagging with Anchor Hidden Markov Models](http://www.aclweb.org/anthology/Q16-1018) ACL '16

Karl Stratos, Michael Collins, Daniel Hsu


### Dependency Parsing and PCFGs

[Spectral Dependency Parsing with Latent Variables](http://www.pdhillon.com/spectral-dep-parsing.pdf) EMNLP '12

Paramveer S. Dhillon, Jordan Rodu, Michael Collins, Dean P. Foster, Lyle H. Ungar

One of the original papers in spectral learning for dependency parsing, the authors propose a latent variable generative model for dependency parsing and a spectral method for parameter estimation. They build on spectral algorithms for HMMs (Hsu et al. 2008), and assume there is a hidden variable for each word. By computing certain spectral parameters from word counts in training data, it is possible to compute the probability of a given tree. In experiments on the Penn Treebank, they can improve the performance of the baseline MST parser by using the tree probabilities estimated by their model to re-rank the outputs of an existing parser, achieving error reduction of up to 4.6%.

[Diversity in Spectral Learning for Natural Language Processing](http://www.aclweb.org/anthology/D15-1214) EMNLP '15

Shashi Narayan and Shay B. Cohen

The authors build on spectral algorithms for parsing with latent-variable PCFGs, presenting a new clustering algorithm that, when combined with careful noise to produce diverse parses, achieves performance on English and German on par with state-of-the-art. Their algorithm splits parse trees in the training trees at each nonterminal in the data, computes representation for the "inner" and "outer" trees, and uses SVD on the empirical covariance matrix for each non-terminal to achieve a low-dimensional representation of each non-terminal. By clustering with k-means on this data, they compute hidden states, followed by a simple counting step to compute rule probabilities. To achieve even better results, they add noise over many iterations to the input data, and rerank and/or combine the parsers to achieve final parses.

[Optimizing Spectral Learning for Parsing](http://www.aclweb.org/anthology/P16-1146) ACL '16

Shashi Narayan and Shay B. Cohen

The authors build on existing spectral algorithms for natural language parsing, improving on previous experimental results. They present a search algorithm akin to beam search which optimizes the number of latent states per non-terminal, as the number of latent states output by spectral algorithms is usually the number of non-zero singular values of a inside-outside tree covariance matrix. However, it often suffers from errors - both due to noisy estimation of variables from data, and estimation errors in the algorithm itself - even capping the number of latent states arbitrarily. Instead, this paper uses a search algorithm to intelligently fix the number of latent states for each nonterminal. It works with any spectral algorithm, and remains efficient as the spectral algorithms are efficient. They test on French, German, Hebrew, Hungarian, Korean, Polish, Swedish, and Basque, and outperform the Berkeley parser for many languages. 

[Spectral Learning of Latent-Variable PCFGs](http://www.cs.columbia.edu/~mcollins/papers/ACL2012final.long.pdf) ACL '12

Shay B. Cohen, Karl Stratos, Michael Collins, Dean P. Foster, Lyle Ungar

The authors present one of the first spectral algorithms for learning latent-variable PCFGs. By putting the inside-outside algorithm into tensor form, they project representations of inside and outside trees, take an SVD of the sample covariance matrix, calculate correlations, and use these to compute final parameters. They also provide theoretical recovery guarantees, under certain assumptions on the L-PCFG.


[Experiments with Spectral Learning of Latent Variable PCFGs](http://www.cs.columbia.edu/~scohen/naacl13spectral.pdf) NAACL '13

Shay B. Cohen, Karl Stratos, Michael Collins, Dean P. Foster, Lyle Ungar

Following the original paper by Cohen et al. in 2012 with a spectral learning algorithm for L-PCFGs, this paper provides extensive empirical tests. They provide a few heuristic improvements to the spectral algorithm, such as backoff smoothing to estimate the number of parameters, clarification of the inside/outside tree embeddings, and scaling feature by their inverse variance. Compared to the EM algorithm, the spectral method is much more efficient and about equally as accurate.


### Topic Modeling

[A Spectral Algorithm for Latent Dirichlet Allocation](https://arxiv.org/pdf/1204.6703.pdf) NIPS '12

Animashree Anandkumar, Dean P. Foster, Daniel Hsu, Sham M. Kakade, Yi-Kai Liu

This was among the first papers to present a spectral algorithm for Latent Dirichlet Allocation, used in topic modeling. The model only requires trigram statistics, and uses spectral decomposition of low order moments (simple SVDs). The SVDs are of size proportional to the number of latent factors, and thus are efficient. In addition, they provide theoretical guarantees for sample complexity and parameter recovery via this method, which they call Excess Correlation Analysis. Note that there is little empirical testing in this paper, aside from a dataset of New York Times articles. 

[Spectral Learning for Supervised Topic Models](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7879308&tag=1) IEEE '17

Yong Ren, Yining Wang, and Jun Zhu

The authors build on existing results in spectral learning for latent dirichlet location in the special case of supervised LDA. Under supervised LDA, documents are observed in which words arise due jointly to the topic and per-topic word distributions. However, they also assume a response variable is known, which is some linear combination of the topic mixing vector. The goal is to recover not only the model parameters, but also the linear coefficients of the observed, "supervised" information. They present both a two-stage spectral method, which learns the regular parameters independently from the new linear parameters using the standard method-of-moments and tensor decomposition approach, as well as a superior joint method, which estimates all parameters together by incorporating the supervised variable into the moment estimations. In addition to theoretical guarantees, empirical tests on synthetic and real datasets show comparable or superior performance to state-of-the-art supervised topic modeling approaches.

[Tensor Decompositions for Learning Latent Variable Models](http://jmlr.org/papers/volume15/anandkumar14b/anandkumar14b.pdf) JMLR '14

Animashree Anandkumar, Rong Ge, Daniel Hsu, Sham M. Kakade, Matus Telgarsky

This is another foundational paper in spectral learning. Similar to an earlier paper specific to Latent Dirichlet Allocation, their approach is to estimate low-order observable moments, and retrieve model parameters by a symmetric tensor decomposition (a generalization of SVD). To address the difficulty of tensor decompositions in general, they apply and analyze the robust tensor power method, resulting in a scalable latent variable estimation method with robustness to noise. Their method applies not only to LDA, but also to Hidden Markov and Gaussian Mixture models.

[Spectral Leader: Online Spectral Learning for Single Topic Models](https://arxiv.org/pdf/1709.07172.pdf) ECML-PKDD '18

Tong Yu, Branislav Kveton, Zheng Wen, Hung Bui, Ole J. Mengshoel

The authors develop a spectral learning algorithm for online topic modeling, SpectralLeader, as a competitor to online Expectation-Maximization. In this setup, one must learn a latent variable model form an ongoing stream of data. In this setup, the distribution of topics may change over time, but the conditional distribution of words is unchanging. At each timestep, the algorithm observes a new document sampled iid from the model at that time; the goal is to predict a sequence of model parameters with low cumulative regret. In contrast to EM, which may get stuck at local optima, SpectralLeader provably converges to the global optimum. The authors derive regret bounds, and test it on synthetic data as well as news and Twitter data, and show that SpectralLeader is competitive with, or in some cases superior to, EM.


### Spectral Clustering

[Computing Word Classes Using Spectral Clustering](https://arxiv.org/pdf/1808.05374.pdf) '18

Effi Levi, Saggy Herman, Ari Rappoport

The authors explore spectral clustering for natural language processing. They provide a basic overview of existing spectral clustering methods, and perform downstream evaluation on semantic role labeling and depenency parsing. They achieve similar results to Brown clustering, and outperform other clustering methods.

[Semantic Word Clusters Using Signed Spectral Clustering](http://aclweb.org/anthology/P17-1087) ACL '17

Joao Sedoc, Jean Gallier, Lyle Ungar, Dean Foster

A common feature of word embeddings is that, by the distributional hypothesis, words in similar contexts have similar meanings. However, word embedding methods such as word2vec then often derive vector representations of synonyms and antonym that are "close" to each other by various metrics. For word clustering, it is useful to connect antonyms with negative weigths, rather than vanilla vector space distance, e.g. cosine or Euclidean distance. In this paper, the authors present a normalized graph cut algorithm for graphs with signed weights, overlaying thesauri (containing synonym and antonym information) on word embeddings. This allows their word clusters to capture both distributional and synonym relations, and through randomized spectral decomposition, the algorithm is efficient and scalable.

### Miscellaneous

[Connecting Weighted Automata and Recurrent Neural Networks through Spectral Learning](https://arxiv.org/pdf/1807.01406.pdf) '18

Guillaume Rabusseau, Tianyu Li, Doina Precup

[Learning Linear Dynamical Systems via Spectral Filtering](https://papers.nips.cc/paper/7247-learning-linear-dynamical-systems-via-spectral-filtering.pdf) NIPS '17

Elad Hazan, Karan Singh, Cyril Zhang

[Multitask Spectral Learning of Weighted Automata](https://papers.nips.cc/paper/6852-multitask-spectral-learning-of-weighted-automata.pdf) NIPS '17

Guillaume Rabusseau, Borja Balle, Joelle Pineau





