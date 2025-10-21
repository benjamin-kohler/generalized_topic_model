#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from patsy import dmatrix
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import scipy
from utils import auto_embeddings_from_list
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
        # sbert_model_to_load: Optional[str] = None,
        model_to_load: Optional[str] = None,
        # siglip_model_to_load: Optional[str] = None,
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
        # self.sbert_model_to_load = sbert_model_to_load
        self.model_to_load = model_to_load
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

        if model_to_load is not None:
            self.M_embeddings = auto_embeddings_from_list(
                df["doc"], model_to_load, batch_size, max_seq_length, device
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

    def set_content(self,content):
        if content is not None:
            self.content_colnames, self.M_content_covariates = self._transform_df(
                content
            )
            self.content = content
        else:
            self.content_colnames = None
            self.M_content_covariates = None
    
    def set_prevalence(self,prevalence):
        if prevalence is not None:
            self.prevalence_colnames, self.M_prevalence_covariates = self._transform_df(
                prevalence
            )
            self.prevalence = prevalence
        else:
            self.prevalence_colnames = None
            self.M_prevalence_covariates = None

    def remove_documents(self, indices: list):
        """
        Remove documents based on the provided indices.

        Args:
            indices (list): List of indices of the documents to remove.
        """


        self.M_bow = scipy.sparse.vstack([self.M_bow[i] for i in range(len(self.df)) if i not in indices])

        self.df = self.df.drop(indices).reset_index(drop=True)

        if self.M_embeddings is not None:
            self.M_embeddings = np.delete(self.M_embeddings, indices, axis=0)


        if self.prevalence is not None:
            self.prevalence_colnames, self.M_prevalence_covariates = self._transform_df(
                self.prevalence
            )
        else:
            self.M_prevalence_covariates = np.zeros(
                (len(self.df.index), 1), dtype=np.float32
            )

        # Extract content covariates matrix
        if self.content is not None:
            self.content_colnames, self.M_content_covariates = self._transform_df(
                self.content
            )
        else:
            self.content_colnames = None
            self.M_content_covariates = None

        if self.prediction is not None:
            self.prediction_colnames, self.M_prediction = self._transform_df(self.prediction)
        else:
            self.M_prediction = None

        # Extract labels matrix
        if self.labels is not None:
            self.labels_colnames, self.M_labels = self._transform_df(self.labels)
        else:
            self.M_labels = None
            

    def remove_words(self, words, match_substring=False):
        """
        Remove words from the vocabulary and the document-term matrix.

        Args:
            words (list): List of words to remove.
        """

        if match_substring:
            indices_to_remove = [self.vectorizer.vocabulary_[k] for k in self.vectorizer.vocabulary_.keys() if any(word in k for word in words)]
        else:
            indices_to_remove = [self.vectorizer.vocabulary_[word] for word in words if word in self.vectorizer.vocabulary_]

        # Remove the words from the vocabulary
        if match_substring:
            # Remove the words from the vocabulary
            self.vectorizer.vocabulary_ = {k: v for k, v in self.vectorizer.vocabulary_.items() if not any(word in k for word in words)}
        else:
            self.vectorizer.vocabulary_ = {k: v for k, v in self.vectorizer.vocabulary_.items() if k not in words}

        # Remove the words from the document-term matrix
        self.M_bow = self.M_bow[:, np.delete(np.arange(self.M_bow.shape[1]), indices_to_remove)]
        self.df["doc_clean"] = self.df["doc_clean"].apply(lambda x: ' '.join([word for word in x.split() if word not in words]))
        self.vocab = self.vectorizer.get_feature_names_out()
        self.id2token = {k: v for k, v in zip(range(0, len(self.vocab)), self.vocab)}


    def adjust_covariates_to_train_dim(self, train_dataset: "GTMCorpus"):
        """
        Adjust the labels and matrices of the content and prevalence covariates
        to match those of the train dataset.
    
        Args:
            train_dataset (GTMCorpus): The training dataset to match dimensions with.
        """

        if self.prevalence is not None:
            train_prevalence_cols = train_dataset.prevalence_colnames
            original_prevalence_covariates = self.M_prevalence_covariates.copy()
            # print(f"Original prevalence covariates shape: {original_prevalence_covariates.shape}")
            # print(f"Train prevalence covariates shape: {train_prevalence_cols}")  
            self.M_prevalence_covariates = np.zeros(
                (len(self.df), len(train_prevalence_cols)), dtype=np.float32
            )
            # print(f"New prevalence covariates shape: {self.M_prevalence_covariates.shape}")
            # print(f"prevalence_colnames: {len(self.prevalence_colnames)}")
            for i, col in enumerate(train_prevalence_cols):
                if col in self.prevalence_colnames:
                    col_idx = self.prevalence_colnames.index(col)
                    self.M_prevalence_covariates[:, i] = original_prevalence_covariates[:, col_idx]
            self.prevalence_colnames = train_prevalence_cols
    

        if self.content is not None:
            train_content_cols = train_dataset.content_colnames
            original_content_covariates = self.M_content_covariates.copy()  
            # print(f"Original content covariates shape: {original_content_covariates.shape}")
            # print(f"Train content covariates shape: {train_content_cols}")
            self.M_content_covariates = np.zeros(
                (len(self.df), len(train_content_cols)), dtype=np.float32
            )
            # print(f"New content covariates shape: {self.M_content_covariates.shape}")
            # print(f"content_colnames: {len(self.content_colnames)}")
            for i, col in enumerate(train_content_cols):
                if col in self.content_colnames:
                    col_idx = self.content_colnames.index(col)
                    self.M_content_covariates[:, i] = original_content_covariates[:, col_idx]
            self.content_colnames = train_content_cols
    
        # Adjust labels
        if self.labels is not None:
            train_labels_cols = train_dataset.labels_colnames
            original_labels = self.M_labels.copy() 
            self.M_labels = np.zeros(
                (len(self.df), len(train_labels_cols)), dtype=np.float32
            )
            for i, col in enumerate(train_labels_cols):
                if col in self.labels_colnames:
                    col_idx = self.labels_colnames.index(col)
                    self.M_labels[:, i] = original_labels[:, col_idx]
            self.labels_colnames = train_labels_cols 
            

    def merge_datasets(self, other):
            """
            Merge two GTMCorpus datasets. Note that covariates need to be the same in both datasets 
    
            Args:
                other (GTMCorpus): The other dataset to merge with.
            """
            self.df = pd.concat([self.df, other.df], ignore_index=True)
            self.M_bow = scipy.sparse.vstack([self.M_bow, other.M_bow])
            if self.M_embeddings is not None and other.M_embeddings is not None:
                self.M_embeddings = np.vstack([self.M_embeddings, other.M_embeddings])
            elif self.M_embeddings is None and other.M_embeddings is not None:
                self.M_embeddings = other.M_embeddings
            elif self.M_embeddings is not None and other.M_embeddings is None:
                pass
            else:
                self.M_embeddings = None
            if self.prevalence is not None and other.prevalence is not None:
                self.prevalence_colnames = list(set(self.prevalence_colnames) | set(other.prevalence_colnames))
                self.M_prevalence_covariates = np.vstack([self.M_prevalence_covariates, other.M_prevalence_covariates])
            else:
                self.prevalence_colnames = None
                self.M_prevalence_covariates = None
            if self.content is not None and other.content is not None:
                self.content_colnames = list(set(self.content_colnames) | set(other.content_colnames))
                self.M_content_covariates = np.vstack([self.M_content_covariates, other.M_content_covariates])
            else:
                self.content_colnames = None
                self.M_content_covariates = None
            if self.prediction is not None and other.prediction is not None:
                self.prediction_colnames = list(set(self.prediction_colnames) | set(other.prediction_colnames))
                self.M_prediction = np.vstack([self.M_prediction, other.M_prediction])
            else:
                self.prediction_colnames = None
                self.M_prediction = None
            if self.labels is not None and other.labels is not None:
                self.labels_colnames = list(set(self.labels_colnames) | set(other.labels_colnames))
                self.M_labels = np.vstack([self.M_labels, other.M_labels])
            else:
                self.labels_colnames = None
                self.M_labels = None

