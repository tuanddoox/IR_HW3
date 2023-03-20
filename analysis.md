
# Overview
The analysis portion of this assignment is similar to the analysis in assignment1-part2.
To receive all 50 points for this portion of the assignment, you need to:
1. Come up with at least 1-2 new features, saving your results to `outputs/new_features_res.json` and adding them to the table below. To receive full credit, your new features need to be well-motivated and to improve performance over the original feature set.
3. Describe what the new features are and why you believe they may lead to improved performance. In the description, make sure to include which loss you are using with the new features. (Fill in _Description of Your Creativity_.)
4. Conduct an analysis of how the three losses perform with the original features (first 3 table rows) and how your new features perform (Fill in _Your Summary_.)

# Results table (fill this in):
| Method | R@5 | P@5 | NDCG |
|--------|-----|-----|------|
| Pointwise| 0.75 | 0.265 | 0.78 |
| Pairwise| 0.77 | 0.273 | 0.79 |
| Listwise| 0.76 | 0.271 | 0.80 |
| New features | x | x | x |

# Description of Your Creativity (fill this in; max 200 words)
> In this section, describe the new features you have came up with. A reader who is familiar with the above methods should be able to read this description and understand exactly what you have changed

# Your summary (fill this in; max 250 words):
Overall, we observed that "position-awared" methods (pairwise and listwise) performed better (even just slightly) than pointwise method, and all three methods show a similar trend. With listwise, it is evident from NCDG, P@1 and P@5 values that it optimizes ranking directly by discounting errors at higher rank more than errors at lower rank, hence getting highest results in such metrics. Ideally, we expect position-based methods like pairwise would get relatively better result than pointwise approach does, due to their priority on minimizing inversions. However, such expectation is not shown on the current dataset and implementation. 

It is also noted that while R@5 and NDCG results of all 3 methods are relatively high, their P@5 results are only around 0.27. This suggests that while these LTR methods succeed in ranking document, they do not perform really well if using them alone as an approach to document relevance classification. In the end, the problem of ranking and problem of document retrieval are different, and it is shown in this implementation.  

In terms of convergence when training, we found that when training all methods with multiple epochs (30 epochs), listwise converges most quickly (however, just by a small margin). All three methods show similar trend in terms of NDCG result at the initial stage of training, before quickly reaching saturation. 

About complexity, pointwise method performs most inexpensively as it uses the simple MSE loss, and the complexity of pointwise algorithm only scales linearly with the number of documents per query. For pairwise and listwise, their complexity is quadratic, as it needs to calculate all possible combination or swap of documents. Training these methods naively take a substantial amount of time. When training pairwise with spedup approach, however, time decreases drastically. 
