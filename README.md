## Improving Learned Index Structures

Search engines use highly complex ranking function to return high-quality results. Recently, major engines have started using transformers such as BERT for better ranking, but this can significantly increase the computational cost. One approach to increase efficiency for this case involves using neural networks to automatically derive better inverted index structures that can be used to efficiently approximate complex transformer-based ranking systems. The goal of this research project will be to improve the state of the art in this area in one or more ways, for example by learning index structures for pairs of terms, by pruning learned indexes, or by finding new processing methods that can deal with the wacky impact score distributions that occur in these learned structures.

## Resources

### Transformers
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding ([paper](https://arxiv.org/pdf/1810.04805.pdf))

### Document Expansion
- Document Expansion by Query Prediction ([paper](https://arxiv.org/pdf/1904.08375.pdf))
- From doc2query to docTTTTTquery ([paper](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf))
- Doc2Query--: When Less is More ([paper](https://arxiv.org/pdf/2301.03266.pdf))

### Information Retrieval
- Passage Re-ranking with BERT ([paper](https://arxiv.org/pdf/1901.04085.pdf))
- Context-Aware Sentence/Passage Term Importance Estimation For First Stage Retrieval ([paper](https://arxiv.org/pdf/1910.10687.pdf))
- Context-Aware Document Term Weighting for Ad-Hoc Search ([paper](https://www.cs.cmu.edu/~callan/Papers/TheWebConf20-Zhuyun-Dai.pdf))
- Efficiency Implications of Term Weighting for Passage Retrieval ([paper](https://www.cs.cmu.edu/~zhuyund/papers/SIGIR2020DeepCT-efficiency.pdf))
- ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT ([paper](https://arxiv.org/pdf/2004.12832.pdf))
- SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking ([paper](https://arxiv.org/pdf/2107.05720.pdf))
- Wacky Weights in Learned Sparse Representations and the Revenge of Score-at-a-Time Query Evaluation ([paper](https://arxiv.org/pdf/2110.11540.pdf))

### DeepImpact
- Learning Passage Impacts for Inverted Indexes ([paper](https://arxiv.org/pdf/2104.12016.pdf))
- Faster Learned Sparse Retrieval with Guided Traversal ([paper](https://arxiv.org/pdf/2204.11314.pdf))
- A Few Brief Notes on DeepImpact, COIL, and a Conceptual Framework for Information Retrieval Techniques ([paper](https://arxiv.org/pdf/2106.14807.pdf))

### Text REtrieval Conference (TREC)
- Overview of the TREC 2019 Deep Learning Track ([paper](https://arxiv.org/pdf/2003.07820.pdf))
- Overview of the TREC 2020 Deep Learning Track ([paper](https://arxiv.org/pdf/2102.07662.pdf))
- Overview of the TREC 2021 Deep Learning Track ([paper](https://www.microsoft.com/en-us/research/uploads/prod/2022/05/trec2021-deeplearning-overview.pdf))

---