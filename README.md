# The Generalized Topic Model (GTM)

The Generalized Topic Model (GTM) is a neural topic model that can be used for exploration, causal inference, and prediction tasks on large text corpora. 

It can handle all forms of document-level metadata:

- **Prevalence covariates** influence topic choice.
- **Content covariates** influence topic content.
- **Document labels** are outcomes you want to predict based on document topic shares.
- **Prediction covariates** influence the prediction of document labels.

It supports two types of document representations: 
- **Document embeddings** (e.g., Doc2Vec, SBERT)
- **Bag-of-words** (e.g., ngrams, TF-IDF)

## Installation

```pip install generalized_topic_model```

## Tutorials 

Get started with one of our [notebooks](notebooks/).

## References

- The Generalized Topic Model (Elliott Ash, Germain Gauthier, Philine Widmer)

## Note

Remember that this is a research tool :)