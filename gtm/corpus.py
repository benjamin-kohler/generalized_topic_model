#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from patsy import dmatrix
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import scipy
from utils import bert_embeddings_from_list
from typing import Optional, Dict, Any
import pandas as pd

class GTMCorpus(Dataset):
    """
    Corpus for the GTM model.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        prevalence: Optional[str] = None,
        content: Optional[str] = None,
        prediction: Optional[str] = None,
        labels: Optional[str] = None,
        normalize_doc_length: bool = False,
        vectorizer: Optional[CountVectorizer] = None,
        vectorizer_args: Dict[str, Any] = {},
        sbert_model_to_load: Optional[str] = None,
        batch_size: int = 64,
        max_seq_length: int = 100000,
        device: Optional[str] = None,
    ):
        """
        Initialize GTMCorpus.

        Args:
            df (pd.DataFrame): Must contain a column 'doc' with the text of each document. 
                If count_words=True, it must also contain 'doc_clean' with the cleaned text of each document.
            prevalence (Optional[str]): Formula for prevalence covariates (of the form "~ cov1 + cov2 + ..."),
                but allows for transformations of e.g., "~ g(cov1) + h(cov2) + ...)". 
                Use "C(your_categorical_variable)" to indicate a categorical variable. 
                See the Patsy package for more details.
            content (Optional[str]): Formula for content covariates (of the form "~ cov1 + cov2 + ..."). 
                Use "C(your_categorical_variable)" to indicate a categorical variable. 
                See the Patsy package for more details.
            prediction (Optional[str]): Formula for covariates used as inputs for the prediction task 
                (also of the form "~ cov1 + cov2 + ..."). See the Patsy package for more details.
            labels (Optional[str]): Formula for labels used as outcomes for the prediction task 
                (of the form "~ label1 + label2 + ...")
            normalize_doc_length (bool): Whether to normalize the document-term matrix by document length 
                (to accommodate for varying document lengths)
            vectorizer (Optional[CountVectorizer]): If None, a new one will be created
            vectorizer_args (Dict[str, Any]): Arguments for the CountVectorizer object
            sbert_model_to_load (Optional[str]): Name of the SentenceTransformer model to load
            batch_size (int): Batch size for SentenceTransformer embeddings
            max_seq_length (int): Maximum sequence length for SentenceTransformer embeddings
            device (Optional[str]): Device to use for SentenceTransformer embeddings
        """

        # Basic params and formulas
        self.prevalence = prevalence
        self.content = content
        self.prediction = prediction
        self.labels = labels
        self.count_vectorizer_args = vectorizer_args
        self.normalize_doc_length = normalize_doc_length
        self.vectorizer = vectorizer
        self.sbert_model_to_load = sbert_model_to_load
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.df = df
        self.device = device

        # Compute bag of words matrix
        if vectorizer is None:
            self.vectorizer = CountVectorizer(**vectorizer_args)
            self.M_bow = self.vectorizer.fit_transform(df["doc_clean"])
        else:
            self.vectorizer = vectorizer
            self.M_bow = self.vectorizer.transform(df["doc_clean"])
        if normalize_doc_length:
            self.M_bow = self.M_bow / self.M_bow.sum(axis=0)
        self.vocab = self.vectorizer.get_feature_names_out()
        self.id2token = {
            k: v for k, v in zip(range(0, len(self.vocab)), self.vocab)
        }

        # Create embeddings matrix
        self.M_embeddings = None

        if sbert_model_to_load is not None:
            self.M_embeddings = bert_embeddings_from_list(
                df["doc"], sbert_model_to_load, batch_size, max_seq_length, device
            )

        # Extract prevalence covariates matrix
        if prevalence is not None:
            self.prevalence_colnames, self.M_prevalence_covariates = self._transform_df(
                prevalence
            )
        else:
            self.M_prevalence_covariates = np.zeros(
                (len(df.index), 1), dtype=np.float32
            )

        # Extract content covariates matrix
        if content is not None:
            self.content_colnames, self.M_content_covariates = self._transform_df(
                content
            )
        else:
            self.content_colnames = None
            self.M_content_covariates = None

        # Extract prediction covariates matrix
        if prediction is not None:
            self.prediction_colnames, self.M_prediction = self._transform_df(prediction)
        else:
            self.M_prediction = None

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

        if type(self.M_bow[i]) == scipy.sparse.csr_matrix:
            M_bow_sample = torch.FloatTensor(self.M_bow[i].todense())
        else:
            M_bow_sample = torch.FloatTensor(self.M_bow[i])
        d["M_bow"] = M_bow_sample

        if self.M_embeddings is not None:
            d["M_embeddings"] = self.M_embeddings[i]

        if self.prevalence is not None:
            d["M_prevalence_covariates"] = self.M_prevalence_covariates[i]

        if self.content is not None:
            d["M_content_covariates"] = self.M_content_covariates[i]

        if self.prediction is not None:
            d["M_prediction"] = self.M_prediction[i]

        if self.labels is not None:
            d["M_labels"] = self.M_labels[i]

        return d
