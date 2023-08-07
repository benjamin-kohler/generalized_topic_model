#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from patsy import dmatrix
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import scipy 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils import bert_embeddings_from_list


class GTMCorpus(Dataset):
    """
    Corpus for the GTM model.
    """

    def __init__(self, df, prevalence=None, content=None, labels=None, embeddings_type=None,
                 count_words=True, normalize_doc_length=False, vectorizer=None,
                 sbert_model_to_load=None, batch_size=200, max_seq_length=100000, 
                 vector_size=300, window=10, min_count=1, workers=4, epochs=10, seed=42):
        """
        Initialize GTMCorpus.

        Args:
            df : pandas DataFrame. Must contain a column 'doc' with the text of each document. If count_words=True, it must also contain 'doc_clean' with the cleaned text of each document.
            prevalence : string, formula for prevalence covariates (of the form "~ cov1 + cov2 + ..."), but allows for transformations of e.g., "~ g(cov1) + h(cov2) + ...)". Use "C(your_categorical_variable)" to indicate a categorical variable. See the Patsy package for more details.
            content : string, formula for content covariates (of the form "~ cov1 + cov2 + ..."). Use "C(your_categorical_variable)" to indicate a categorical variable. See the Patsy package for more details.
            labels : string, formula for labels (of the form "~ label1 + label2 + ...")
            embeddings_type : (optional) string, type of embeddings to use. Can be 'Doc2Vec' or 'SentenceTranformer'
            count_words : boolean, whether to produce a document-term matrix or not
            normalize_doc_length : boolean, whether to normalize the document-term matrix by document length (to accomodate for varying document lengths)
            vectorizer : sklearn CountVectorizer object, if None, a new one will be created
            sbert_model_to_load : string, name of the SentenceTranformer model to load
            batch_size : int, batch size for SentenceTranformer embeddings
            max_seq_length : int, maximum sequence length for SentenceTranformer embeddings
            vector_size : int, dimension of the Doc2Vec model
            window : int, window size for the Doc2Vec training
            min_count : int, minimum count of tokens for the Doc2Vec training
            workers : int, number of workers for the Doc2Vec training
            epochs : int, number of epochs for the Doc2Vec training
            seed : int, random seed for the Doc2Vec training (ensures reproducibility when combined with a python hash seed and workers=1)
        """

        # Basic params and formulas
        self.prevalence = prevalence
        self.content = content
        self.labels = labels
        self.embeddings_type = embeddings_type  
        self.count_words = count_words
        self.normalize_doc_length = normalize_doc_length
        self.vectorizer = vectorizer
        self.sbert_model_to_load = sbert_model_to_load
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.seed = seed
        self.df = df

        # Compute bag of words matrix
        if self.count_words:
            if vectorizer is None:
                self.vectorizer = CountVectorizer()
                self.M_bow = self.vectorizer.fit_transform(df['doc_clean'])
            else:
                self.vectorizer = vectorizer
                self.M_bow = self.vectorizer.transform(df['doc_clean'])
            if normalize_doc_length:
                self.M_bow = self.M_bow / self.M_bow.sum(axis=0)
            self.vocab = self.vectorizer.get_feature_names_out()
            self.id2token = {k: v for k, v in zip(range(0, len(self.vocab)), self.vocab)}
            self.log_word_frequencies = torch.FloatTensor(np.log(np.array(self.M_bow.sum(axis=0)).flatten()))
        else:
            self.M_bow = None
            self.vocab = None
            self.id2token = None

        # Create embeddings matrix
        self.M_embeddings = None
        self.V_embeddings = None

        if embeddings_type == 'Doc2Vec':
            clean_docs = [doc.split() for doc in df['doc_clean']]
            tagged_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(clean_docs)]
            self.Doc2Vec_model = Doc2Vec(tagged_documents, vector_size=vector_size, window=window, min_count=min_count, workers=workers, epochs=epochs, seed=seed)
            self.M_embeddings = np.array([self.Doc2Vec_model.infer_vector(doc) for doc in clean_docs])
            self.V_embeddings = np.array([self.Doc2Vec_model.infer_vector([token]) for token in self.vocab])

        if embeddings_type == 'SentenceTransformer':
            self.M_embeddings = bert_embeddings_from_list(df['doc'], sbert_model_to_load, batch_size, max_seq_length)
            self.V_embeddings = bert_embeddings_from_list(self.vocab, sbert_model_to_load, batch_size, max_seq_length)

        # Extract prevalence covariates matrix
        if prevalence is not None:
            self.prevalence_colnames, self.M_prevalence_covariates = self._transform_df(prevalence)
        else:
            self.M_prevalence_covariates = np.zeros((len(df['doc']),1), dtype=np.float32)

        # Extract content covariates matrix
        if content is not None:
            self.content_colnames, self.M_content_covariates = self._transform_df(content)
        else:
            self.M_content_covariates = None

        # Extract labels matrix
        if labels is not None:
            self.labels_colnames, self.M_labels = self._transform_df(labels)
        else:
            self.M_labels = None

    def _transform_df(self, formula):
        """
        Uses the patsy package to transform covariates into appropriate matrices
        """
        
        M = dmatrix(formula, self.df)
        colnames = M.design_info.column_names
        M = np.asarray(M, dtype=np.float32)

        return colnames, M

    def __len__(self):
        """Return length of dataset."""
        return len(self.df)

    def __getitem__(self, i):
        """Return sample from dataset at index i"""

        d = {}

        if self.M_bow is not None:
            if type(self.M_bow[i]) == scipy.sparse.csr_matrix:
                M_bow_sample = torch.FloatTensor(self.M_bow[i].todense())
            else:
                M_bow_sample = torch.FloatTensor(self.M_bow[i])
            d["M_bow"] = M_bow_sample

        if self.M_embeddings is not None:
            d['M_embeddings'] = self.M_embeddings[i]

        if self.labels is not None:
            d['M_labels'] = self.M_labels[i]

        if self.content is not None:
            d['M_content_covariates'] = self.M_content_covariates[i]

        if self.prevalence is not None:
            d['M_prevalence_covariates'] = self.M_prevalence_covariates[i]

        return d