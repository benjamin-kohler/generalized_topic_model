# Standard Library
from random import random

# Third Party Library
import numpy as np

# from gtp import *


def generate_docs_by_gtm(
    num_topics,
    num_iters,
    prior_topic_prop,
    decoder_func,
    num_covs,
    doc_args=None,
    is_output=True,
):
    # return df_true_dist_list, docs
    if prior_topic_prop not in ["logistic_normal", "dirichlet"]:
        raise ValueError(
            "Only two options for prior on topic proportions: \
            logistic_normal, and dirichlet."
        )
    if decoder_func not in ["SAGE", "ETM"]:
        raise ValueError(
            "Only two options for decoder function: \
            SAGE, and ETM."
        )

    default_data_args_dict = {
        "min_words": 50,
        "max_words": 100,
        "num_docs": 5000,
        "voc_size": 1000,
    }
    if doc_args is None:
        doc_args = default_data_args_dict
    for k in default_data_args_dict.keys():
        if k not in doc_args.keys():
            doc_args[k] = default_data_args_dict[k]
    lambda_ = np.random.rand(num_covs, num_topics)
    prevalence_covariates = np.randint(0, 10, num_covs)
    if prior_topic_prop == "logistic_normal":
        sqrt_sigma = np.random.rand(num_topics, num_topics)
        sigma = sqrt_sigma * sqrt_sigma.T
        mean = np.exp(np.matmul(prevalence_covariates, lambda_))
        doc_topic_pro_raw = np.random.multivariate_normal(
            mean, sigma, doc_args["num_docs"]
        )
        doc_topic_pro = np.exp(doc_topic_pro_raw) / np.sum(
            np.exp(doc_topic_pro_raw), axis=1, keepdims=True
        )
    else:
        alpha = np.exp(np.matmul(prevalence_covariates, lambda_))
        doc_topic_pro = np.random.dirichlet(alpha, doc_args["num_docs"])

    if decoder_func == "SAGE":
        # plain SAGE
        # TODO1 check if my understanding of softmax and topic_word_pro is correct
        # TODO2 add complete
        topic_word_pro = np.array(
            [
                [random() for _ in range(doc_args["voc_size"])]
                for _ in range(doc_args["num_docs"])
            ]
        )
        doc_word_pro_raw = np.dot(doc_topic_pro, topic_word_pro)
        doc_word_pro = np.exp(doc_word_pro_raw) / np.sum(
            np.exp(doc_word_pro_raw), axis=1, keepdims=True
        )
        pass
    else:
        num_embeddings = 300

    if is_output:
        pass


def estimate_dist_by_gtm():
    # return df_doc_topic_list, df_topic_word_list
    pass
