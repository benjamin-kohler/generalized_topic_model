# The Generalized Topic Model (GTM)

The Generalized Topic Model (GTM) is a neural topic model that can be used for exploration, causal inference, and prediction tasks on large (multilingual) text corpora. 

It can handle all forms of document-level metadata:

- **Prevalence covariates** influence topic choice.
- **Content covariates** influence topic content conditional on topic choice.
- **Document labels** are outcomes you want to predict based on document topic shares.
- **Prediction covariates** influence the prediction of document labels.

It supports two types of document representations: 
- **Document embeddings** 
- **Word frequencies** 

## Tutorials 

Get started with one of our [notebooks](notebooks/).

## References

[**Deep Latent Variable Models for Unstructured Data**](https://www.dropbox.com/scl/fi/c30hibel8ad93owfiz2lh/Deep_Latent_Variable_Models_for_Unstructured_Data.pdf?rlkey=xn9u9og0d0a603i4b7j4i511a&st=pisq7110&dl=0) \
(with Elliott Ash & Philine Widmer) 

[**generalized_topic_models: A Python Package to Estimate Neural Topic Models**](https://www.dropbox.com/scl/fi/g8j1wec3uy7g1w37gapdc/GTM_JSS_draft.pdf?rlkey=pdfmylxxcs5r6w2f0hilb74xo&st=vhvci1kz&dl=0) \
(with Elliott Ash & Philine Widmer) 

## Disclaimers

The package is still in development :)
