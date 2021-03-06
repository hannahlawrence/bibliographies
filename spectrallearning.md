# Annotated Bibliography for Spectral Learning in NLP

Hannah Lawrence

The following is an annotated bibligraphy for the broad area of spectral learning, primarily as it has been applied to a variety of natural language processing tasks. Papers are organized by NLP task and include the relevant conference or journal, publication year and authors; they are also annotated with a brief description. A handful of tutorials and recent, non-NLP papers are included as well. Spectral learning is an enormous area, even restricted to NLP; this is intended to be a thorough, but necessarily incomplete, curation of many relevant resources.

## Tutorials

### Spectral Learning

[Spectral Learning Algorithms for Natural Language Processing](http://homepages.inf.ed.ac.uk/scohen/naacl13tutorial/naacl13tutorial-slides.pdf) NAACL '13 - Slides

Shay Cohen, Michael Collins, Dean Foster, Karl Stratos, Lyle Ungar

[Spectral Methods for Natural Language Processing](http://www.karlstratos.com/publications/thesis.pdf) '16 - PhD Thesis

Karl Stratos

[Spectral Learning Techniques for Weighted Automata, Transducers, and Grammars](http://emnlp2014.org/tutorials/10_notes.pdf) EMNLP '14 - Slides

Borja Balle, Ariadna Quattoni, Xavier Carreras

[Introduction to Spectral Learning](http://www.cs.cmu.edu/~hanxiaol/slides/spectral_learning.pdf) '13 - Slides

Hanxiao Liu

### Linear Algebraic Methods
Singular value decompositions and canonical correlation analysis are at the heart of many of the spectral methods used in the papers below.

[Singular Value Decomposition](https://www.cs.cmu.edu/~venkatg/teaching/CStheory-infoage/book-chapter-4.pdf) '12 - Book Chapter

John Hopcroft, Ravi Kannan

[Canonical Correlation Analysis](https://www.cs.cmu.edu/~tom/10701_sp11/slides/CCA_tutorial.pdf) '01 - Tutorial Document

Magnus Borga

### Spectral Clustering
Spectral clustering takes a different, more graph-theoretic flavor than most other spectral techniques; these tutorials may be useful for gaining familiarity with the methods used in spectral clustering.

[Spectral Clustering for Beginners](https://towardsdatascience.com/spectral-clustering-for-beginners-d08b7d25b4d8) '18 - Blog Post

Amine Aoullay

[A Tutorial on Spectral Clustering](http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf) '07 - Article (Statistics and Computing, 17 (4), 2007)

Ulrike von Luxburg

### Latent Variable Models

[Latent Variable Models in NLP](http://www.nactem.ac.uk/tsujii/T-FaNT2/T-FaNT.files/Slides/haghighi.pdf) - Slides

Aria Haghighi, Slav Petrov, John DeNero, Dan Klein

[Probabilistic CFG with Latent Annotations](http://www.aclweb.org/anthology/P05-1010) ACL '05 - Paper

Takuya Matsuzaki, Yusuke Miyao, Jun'ichi Tsujii

## Papers

### Foundational
[Spectral Learning](https://people.eecs.berkeley.edu/~klein/papers/spectral-learning.pdf) IJCAI '03

Sepandar D. Kamvar, Dan Klein, and Christopher D. Manning

This very early paper in spectral learning presents a spectral clustering approach to classification. Their algorithm is flexible, in that it can incorporate supervised information, but also takes advantage of unlabeled data - in empirical tests, classification accuracy increases even with the addition of unlabeled documents. 

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

This paper is foundational, not just for learning Hidden Markov Models, but for spectral learning more generally; the methods defined here inspired algorithms in later papers for topic modeling, dependency parsing, etc. In this paper, the authors present the first spectral algorithm for learning Hidden Markov Models, under certain assumptions such as separability: the observation distributions arising from hidden states must be distinct, for example. The algorithm has polynomial sample and computational complexity, and consists of a SVD between past and future operations (which can be interpreted as an application of Canonical Correlation Analysis). The algorithm does not explicitly learn the model parameters, but learns a linear function of them which can be used to compute joint probabilities. They provide theoretical guarantees for this algorithm, taking into account estimation error.

[Spectral Dimensionality Reduction for HMMs](https://arxiv.org/pdf/1203.6130.pdf) '12

Dean P. Foster, Jordan Rodu, Lyle H. Ungar

As in the previous paper, there is a fast spectral method based on co-occurrence of pairs and triples for learning HMMs, which is much faster than EM or Gibbs sampling. In this paper, the authors present a similar spectral method which improves upon the previous parameter and sample complexity: it reduces the number of model parameters that must be estimated, and the sample complexity required does not depend on the size of the observation vocabulary. They do so by reducing the dimension of intermediate per-emission matrices, allowing for similar computation of the probabilities of emission sequences. They provide the usual sample and accuracy guarantees, with accuracy measured as a ratio of probabilities of sequences, although they do not empirically test this modified HMM algorithm on NLP tasks.

[Spectral Learning of Mixture of Hidden Markov Models](https://paris.cs.illinois.edu/pubs/subakan-nips2014.pdf) NIPS '14

Y. Cem Subakan, Johannes Traa, Paris Smaragdis

This paper presents a spectral algorithm for learning mixture of Hidden Markov Models, for which expectation maximization is often computationally infeasible. They employ the standard method of moments procedure to recover model parameters, representing a mixture HMM as an HMM with block diagonal transition matrix. Folloiwng this step, they resolve a permutation ambiguity using spectral properties of the global transition matrix, resulting in parameter estimation for individual HMMs. With empirical tests, they show that while the spectral algorithm alone does not always perform best, EM initialized with the spectral algorithm enjoys considerable improvement over random initialization.

[Unsupervised Part-of-Speech Tagging with Anchor Hidden Markov Models](http://www.aclweb.org/anthology/Q16-1018) ACL '16

Karl Stratos, Michael Collins, Daniel Hsu

In this paper, the authors apply a different spectral method for learning a new variety of Hidden Markov Models, designed specifically for part-of-speech tagging. These HMMs, called Anchor Hidden Markov Models, are unique: they have at least one emission per hidden state such that it only has non-zero emission probability for that single hidden state (thus "anchoring" it). Their method for parameter recovery involves non-negative matrix factorization, and recovers the exact model parameters. They also explore various projection methods for dimensionality reduction for an observation-context expectation matrix, which is an input to their algorithm. In experiments on POS tagging with the universal treebank dataset, they achieve the best performance on 5 out of 10 languages. 

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

Although the following recent papers do not necessarily interface directly with NLP, they present different facets and areas of development for spectral learning that are worth including nonetheless.

[Connecting Weighted Automata and Recurrent Neural Networks through Spectral Learning](https://arxiv.org/pdf/1807.01406.pdf) '18

Guillaume Rabusseau, Tianyu Li, Doina Precup

This paper reveals a novel connection between weighted finite automata and second-order recurrent neural networks, allowing them to apply a spectral learning algorithm to linear 2-RNNs with provable learning guarantees. 

[Learning Linear Dynamical Systems via Spectral Filtering](https://papers.nips.cc/paper/7247-learning-linear-dynamical-systems-via-spectral-filtering.pdf) NIPS '17

Elad Hazan, Karan Singh, Cyril Zhang

To learn linear dynamical systems, the authors' approach includes a "spectral filtering" technique for convex relaxation, specifically using the eigenvectors of a Hankel matrix. 

[Multitask Spectral Learning of Weighted Automata](https://papers.nips.cc/paper/6852-multitask-spectral-learning-of-weighted-automata.pdf) NIPS '17

Guillaume Rabusseau, Borja Balle, Joelle Pineau

The authors develop a spectral learning algorithm for vector-valued weighted finite automata, as one solution to the multitask learning problem. 





