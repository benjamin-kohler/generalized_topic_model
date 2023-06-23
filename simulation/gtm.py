# Standard Library
import pathlib
import pickle
from random import random

# Third Party Library
import numpy as np
import pandas as pd
from tqdm import tqdm


def generate_docs_by_gtm(
    num_topics,
    num_covs,
    prior_topic_prop,
    decoder_func,
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
    prevalence_covariates = np.random.randint(0, 10, num_covs)
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

    content_covariates = np.array(
        [[np.random.randint(0, 10, num_covs)] for _ in range(doc_args["num_docs"])]
    )
    if decoder_func == "SAGE":
        # plain SAGE
        # TODO1 check if my understanding of softmax and topic_word_pro is correct
        # TODO2 add complete SAGE
        # TODO3 add complete
        topic_word_pro = np.array(
            [
                [random() for _ in range(doc_args["voc_size"])]
                for _ in range(doc_args["num_docs"])
            ]
        )
        doc_word_pro_raw = np.dot(doc_topic_pro, topic_word_pro)
    else:
        num_embeddings = 300
        rho = np.array(
            [
                [random() for _ in range(num_embeddings)]
                for _ in range(doc_args["num_docs"])
            ]
        )
        alpha = np.array(
            [[random() for _ in range(num_embeddings)] for _ in range(num_topics)]
        )
        phi = np.array(
            [[random() for _ in range(num_embeddings)] for _ in range(num_covs)]
        )
        doc_word_pro_raw = np.dot(
            rho, (np.dot(doc_topic_pro, alpha) + np.dot(content_covariates, phi)).T
        )

    doc_word_pro = np.exp(doc_word_pro_raw) / np.sum(
        np.exp(doc_word_pro_raw), axis=1, keepdims=True
    )
    topicnames = ["Topic" + str(i) for i in range(num_topics)]
    docnames = ["Doc" + str(i) for i in range(doc_args["num_docs"])]
    words = ["word_" + str(i) for i in range(doc_args["voc_size"])]

    df_doc_topic = pd.DataFrame(doc_topic_pro, index=docnames, columns=topicnames)
    df_topic_word = pd.DataFrame(topic_word_pro, index=topicnames, columns=words)
    df_doc_word = pd.DataFrame(doc_word_pro, index=docnames, columns=words)
    df_true_dist_list = [df_doc_topic, df_topic_word]

    docs = []
    for docname in tqdm(docnames):
        sentence = []
        num_words = np.random.randint(
            default_data_args_dict["min_words"], default_data_args_dict["max_words"]
        )
        word_pro = df_doc_word.loc[docname, :]
        for _ in range(num_words):
            word = np.random.choice(words, p=word_pro)
            sentence.append(word)
        docs.append(" ".join(sentence))

    if is_output:
        p = pathlib.Path()
        current_dir = p.cwd()
        if not current_dir.joinpath("data").exists():
            current_dir.joinpath("data").mkdir()
        if not current_dir.joinpath("data", "gtm").exists():
            current_dir.joinpath("data", "gtm").mkdir()

        true_df_doc_topic_path = (
            current_dir.joinpath("data", "gtm", "true_df_doc_topic.pickle")
            .resolve()
            .as_posix()
        )
        true_df_topic_word_path = (
            current_dir.joinpath("data", "gtm", "true_df_topic_word.pickle")
            .resolve()
            .as_posix()
        )
        true_df_doc_word_path = (
            current_dir.joinpath("data", "gtm", "true_df_doc_word.pickle")
            .resolve()
            .as_posix()
        )
        original_docs_path = (
            current_dir.joinpath("data", "gtm", "original_docs.pickle")
            .resolve()
            .as_posix()
        )
        with open(true_df_doc_topic_path, "wb") as f:
            pickle.dump(df_doc_topic, f)
        with open(true_df_topic_word_path, "wb") as f:
            pickle.dump(df_topic_word, f)
        with open(true_df_doc_word_path, "wb") as f:
            pickle.dump(df_doc_word, f)
        with open(original_docs_path, "wb") as f:
            pickle.dump(docs, f)

    return df_true_dist_list, docs


def estimate_dist_by_gtm():
    # return df_doc_topic_list, df_topic_word_list
    pass
