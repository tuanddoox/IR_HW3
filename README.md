[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-f4981d0f882b2a3f0472912d15f9806d57e124e0fc890972558857b51b24a6f9.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=10366963)
# Assignment 2 - Part 1: Learning to Rank <a class="anchor" id="toptop"></a>

## Introduction
Welcome to the first part of the LTR assignment. In the previous assignment, you experimented with retrieval with different ranking functions and in addition, different document representations. 

This assignment deals directly with learning to rank (LTR). In offline LTR, you will learn how to implement methods from the three approaches associated with learning to rank: pointwise, pairwise and listwise. 

**Learning Goals**:
  - Extract features from (query, document) pairs 
  - Implement pointwise, pairwise and listwise algorithms for learning-to-rank
  - Train and evaluate learning-to-rank methods

## Guidelines

### How to proceed?
We have prepared a notebook: `hw2.ipynb` including the detailed guidelines of where to start and how to proceed with this part of the assignment. Alternatively, you can use `hw2.py` if you prefer a Python script that can easily be run from the command line.
The two files are equivalent.

You can find all the code of this assignment inside the `ltr` package. The structure of the `ltr` package is shown below, containing various modules that you need to implement. For the files that require changes, a :pencil2: icon is added after the file name. This icon is followed by the points you will receive if all unit tests related to this file pass. 

**NOTICE THAT YOU NEED TO PUT YOUR IMPLEMENTATION BETWEEN `BEGIN \ END SOLUTION` TAGS!** All of the functions that need to be implemented in modules directory files have a #TODO tag at the top of their definition.

**Structure of the ``ltr`` package:**

📦ltr\
 ┣ 📜dataset.py :pencil2: (25 points)\
 ┣ 📜eval.py\
 ┣ 📜loss.py :pencil2: (50 points)\
 ┣ 📜model.py :pencil2: (10 points)\
 ┣ 📜train.py :pencil2: (15 points)\
 ┗ 📜utils.py

In the `ltr.dataset.FeatureExtraction.extract()` method, you are required to extract various features from a (query, document) pair. The list of features and its definitions can be found in the following table:

| Feature                | Definition |
|------------------------|------------|
| bm25| [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) score. Parameters: k1 = 1.5, b = 0.75 |
|query_term_coverage| Number of query terms in the document |
|query_term_coverage_ratio| Ratio of # query terms in the document to #query terms in the query  |
|stream_length| Length of document |
|idf| Sum of document frequencies. idf_smoothing = 0.5 , formula: Log((N+1)/(df[term]+smoothing))| 
|sum_stream_length_normalized_tf| Sum over the ratios of each term to document length |
|min_stream_length_normalized_tf| Min over the ratios of each term to document length |
|max_stream_length_normalized_tf| Max over the ratios of each term to document length |
|mean_stream_length_normalized_tf| Mean over the ratios of each term to document length|
|var_stream_length_normalized_tf| Variance over the ratios of each term to document length |
|sum_tfidf| Sum of tf-idf|
|min_tfidf| Min of tf-idf|
|max_tfidf| Max of tf-idf |
|mean_tfidf| Mean of tf-idf |
|var_tfidf| Variance of tf-idf |


This assignment part is worth 150 points total. By completing all required implementations correctly, you can earn a total of 100 points. We allocate 50 additional points that are awarded based on:

- (1) an evaluation of the result files you provide at: 
  - `outputs/pontwise_res.json`
  - `outputs/pairwise_res.json`
  - `outputs/listwise_res.json`
  - `outputs/new_features_res.json`
- (2) a short summary of new features you came up with and your findings in the [analysis.md](analysis.md) file.

Make sure to __commit__ and __push__ the `.json` results files after training and running your models. For the analysis, you can base your observations on the reported metrics for the _test_ splits of the datasets that you train your models on. To receive all 50 points for the analysis, you will need to come up with new features (`hint`: you can use features from previous assignments), justify their choice, and analyze the performance of learning-to-rank algorithms with the new features added in. See the [analysis.md](analysis.md) file for a detailed explanation. 

**Important Remarks**:

Please note that you **SHOULD NOT** change params, seed(always=42), or any PATH variable given in the notebook for training LTR models with various loss functions, otherwise, you might lose points. We do NOT check the notebook; we ask this since we evaluate your results with the given parameters.

---
**Recommended Reading**:
  - Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. Learning to rank using gradient descent. InProceedings of the 22nd international conference on Machine learning, pages 89–96, 2005.
  - Christopher J Burges, Robert Ragno, and Quoc V Le. Learning to rank with nonsmooth cost functions. In Advances in neural information processing systems, pages 193–200, 2007
  - (Sections 1, 2 and 4) Christopher JC Burges. From ranknet to lambdarank to lambdamart: An overview. Learning, 11(23-581):81, 2010
  

Additional Resources: 
- This assignment requires knowledge of [PyTorch](https://pytorch.org/). If you are unfamiliar with PyTorch, you can go over [these series of tutorials](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)




## Table of Contents

Table of contents:

 - Chapter 1: Offline LTR
     - Section 1: Dataset and feature extraction
     - Section 2: Pointwtise LTR
     - Section 3: Pairwise LTR
     - Section 4: Pairwise Speed-up RankNet
     - Section 5: Listwise LTR
     - Section 6: Evaluation

