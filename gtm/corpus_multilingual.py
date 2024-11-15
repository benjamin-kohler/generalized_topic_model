import torch
from torch.utils.data import Dataset, DataLoader, Subset
from patsy import dmatrix
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import scipy
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

class GTMCorpus_Multilingual(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        prevalence: Optional[str] = None,
        content: Optional[str] = None,
        prediction: Optional[str] = None,
        labels: Optional[str] = None,
        vectorizer_args: Dict[str, Dict] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize GTMCorpus.
        """
        self.prevalence = prevalence
        self.content = content
        self.prediction = prediction
        self.labels = labels
        self.vectorizer_args = vectorizer_args or {"all_languages": {}}
        self.device = device

        self.df = df.reset_index(drop=True)
        self.languages = [col.replace("doc_clean_", "") for col in df.columns if col.startswith("doc_clean_")]
        self.ref_lang = self.languages[0]

        self.prevalence_colnames, self.M_prevalence_covariates = self._extract_covariates(prevalence, df)
        self.content_colnames, self.M_content_covariates = self._extract_covariates(content, df)
        self.prediction_colnames, self.M_prediction = self._extract_covariates(prediction, df)
        self.labels_colnames, self.M_labels = self._extract_covariates(labels, df)

        self._initialize_vectorizers()
        self.language_bow_matrices = self._create_language_specific_bow_matrices()
        self.M_bow_combined = self._create_combined_bow_matrix()
        self._categorize_alignment()

    def _extract_covariates(self, formula: Optional[str], df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
        if formula:
            colnames, M = self._transform_df(formula, df)
            return colnames, M
        return None, None

    def _initialize_vectorizers(self):
        self.vectorizers = {}
        if "all_languages" in self.vectorizer_args:
            for lang in self.languages:
                self.vectorizers[lang] = CountVectorizer(**self.vectorizer_args['all_languages'])
        else:
            for lang in self.languages:
                self.vectorizers[lang] = CountVectorizer(**self.vectorizer_args.get(lang, {}))

    def _create_language_specific_bow_matrices(self) -> Dict[str, scipy.sparse.csr_matrix]:
        """
        Creates a dictionary of language-specific Bag-of-Words matrices (M_bow),
        with word frequencies for each language separately.
        """
        language_bow_matrices = {}
        for lang in self.languages:
            lang_data = self.df[["doc_clean_" + lang]]
            vectorizer = self.vectorizers.get(lang)
            bow_matrix = vectorizer.fit_transform(lang_data["doc_clean_" + lang].fillna(""))
            #full_bow_matrix = scipy.sparse.csr_matrix((self.df.shape[0], bow_matrix.shape[1]))
            #non_missing_indices = lang_data.dropna().index
            #full_bow_matrix[non_missing_indices, :] = bow_matrix
            language_bow_matrices[lang] = bow_matrix
        return language_bow_matrices

    def _create_combined_bow_matrix(self) -> scipy.sparse.csr_matrix:
        """
        Creates a combined Bag-of-Words matrix (M_bow_combined) with word frequencies from all languages.
        """
        bow_matrices = [self.language_bow_matrices[lang] for lang in self.languages]
        combined_bow_matrix = scipy.sparse.hstack(bow_matrices).tocsr()
        return combined_bow_matrix

    def _categorize_alignment(self):
        self.fully_aligned_indices = []
        self.partially_aligned_indices = defaultdict(list)
        self.unaligned_indices = defaultdict(list)

        for idx, row in self.df.iterrows():
            present_langs = {lang for lang in self.languages if pd.notnull(row["doc_clean_" + lang])}

            if len(present_langs) == len(self.languages):
                self.fully_aligned_indices.append(idx)
            elif len(present_langs) > 1:
                self.partially_aligned_indices[frozenset(present_langs)].append(idx)
            elif len(present_langs) == 1:
                self.unaligned_indices[next(iter(present_langs))].append(idx)

    def _transform_df(self, formula: str, df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
        """
        Uses the patsy package to transform covariates into appropriate matrices.
        """
        M = dmatrix(formula, df)
        colnames = M.design_info.column_names
        M = np.asarray(M, dtype=np.float32)
        return colnames, M

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.df)

    def __getitem__(self, i: int) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """Return sample from dataset at index i, excluding any None entries."""
        sample = {
            "M_bow_combined": torch.FloatTensor(self.M_bow_combined[i].todense() if scipy.sparse.issparse(self.M_bow_combined) else self.M_bow_combined[i])
        }

        if self.M_prevalence_covariates is not None:
            sample["M_prevalence_covariates"] = torch.FloatTensor(self.M_prevalence_covariates[i])
        
        if self.M_content_covariates is not None:
            sample["M_content_covariates"] = torch.FloatTensor(self.M_content_covariates[i])
        
        if self.M_prediction is not None:
            sample["M_prediction"] = torch.FloatTensor(self.M_prediction[i])
        
        if self.M_labels is not None:
            sample["M_labels"] = torch.FloatTensor(self.M_labels[i])

        for lang in self.languages:
            lang_bow_matrix = self.language_bow_matrices.get(lang)
            if lang_bow_matrix is not None:
                sample[f"M_bow_{lang}"] = torch.FloatTensor(lang_bow_matrix[i].todense() if scipy.sparse.issparse(lang_bow_matrix) else lang_bow_matrix[i])

        return sample

    def _get_dataloaders(self, batch_size: int, num_workers: int, shuffle: bool = True) -> List[DataLoader]:
        """
        Returns DataLoaders for unaligned, partially aligned, and fully aligned data.

        Args:
            batch_size (int): The batch size for the DataLoaders.
            num_workers (int): The number of worker processes for the DataLoaders.
            shuffle (bool): Whether to shuffle the data in the DataLoaders.

        Returns:
            List[DataLoader]: A list of DataLoader objects.
        """
        dataloaders = []
        langs = []

        # Unaligned DataLoaders
        for lang, indices in self.unaligned_indices.items():
            #print(lang)
            unaligned_dataset = Subset(self, indices)
            dataloaders.append(DataLoader(
                unaligned_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers
            ))
            langs.append(lang)

        # Partially aligned DataLoaders
        for langs, indices in self.partially_aligned_indices.items():
            #print(langs)
            partially_aligned_dataset = Subset(self, indices)
            dataloaders.append(DataLoader(
                partially_aligned_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers
            ))
            langs.append(langs)

        # Fully aligned DataLoader
        if self.fully_aligned_indices:
            fully_aligned_dataset = Subset(self, self.fully_aligned_indices)
            dataloaders.append(DataLoader(
                fully_aligned_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers
            ))
            langs.append(self.languages)

        return langs, dataloaders
