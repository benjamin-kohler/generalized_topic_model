# Standard Library
import pathlib
import pickle
from random import random

# Third Party Library
import numpy as np
import pandas as pd
from corpus import GTMCorpus
from tqdm import tqdm

# First Party Library
from gtm import GTM


def generate_docs_by_gtm(
    num_topics,
    num_covs,
    doc_topic_prior,
    decoder_type,
    doc_args=None,
    is_output=True,
):
    # return df_true_dist_list, docs
    if doc_topic_prior not in ["logistic_normal", "dirichlet"]:
        raise ValueError(
            "Only two options for prior on topic proportions: \
            logistic_normal, and dirichlet."
        )
    if decoder_type not in ["sage", "mlp"]:
        raise ValueError(
            "Only two options for decoder function: \
            sage, and mlp."
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
    if doc_topic_prior == "logistic_normal":
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
    if decoder_type == "sage":
        # plain SAGE
        # TODO1 check if my understanding of softmax and topic_word_pro is correct
        # TODO2 add complete sage
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
    words = ["word_" + str(i) for i in range(docgtm_args["voc_size"])]

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
            current_dir.joinpath(
                "data",
                "gtm",
                "true_df_doc_topic_{}_{}.pickle".format(doc_topic_prior, decoder_type),
            )
            .resolve()
            .as_posix()
        )
        true_df_topic_word_path = (
            current_dir.joinpath(
                "data",
                "gtm",
                "true_df_topic_word_{}_{}.pickle".format(doc_topic_prior, decoder_type),
            )
            .resolve()
            .as_posix()
        )
        true_df_doc_word_path = (
            current_dir.joinpath(
                "data",
                "gtm",
                "true_df_doc_word_{}_{}.pickle".format(doc_topic_prior, decoder_type),
            )
            .resolve()
            .as_posix()
        )
        original_docs_path = (
            current_dir.joinpath(
                "data",
                "gtm",
                "original_docs_{}_{}.pickle".format(doc_topic_prior, decoder_type),
            )
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


def _create_input_for_gtm_from_generated_docs(
    data,
):
    """
    input
        generated_new_docs[list]
    output
        model_data_df
            doc
            doc_clean
            cov
    """
    # TODO check if this setting is fine
    num_docs = len(data)
    cov_list = [np.random.randint(0, 1) for _ in range(num_docs)]
    model_data_df = pd.DataFrame({"doc": data, "doc_clean": data, "cov": cov_list})
    return model_data_df


def estimate_dist_by_gtm(
    model_data_df,
    num_docs,
    num_topics,
    voc_size,
    num_silulations,
    doc_topic_prior,
    decoder_type,
    model_args=None,
    is_output=True,
):
    # TODO setting the default args
    default_model_args_dict = {
        "update_every": 1,
    }
    if model_args is None:
        model_args = default_model_args_dict

    df_doc_topic_list, df_topic_word_list = [], []
    for i in tqdm(range(num_silulations)):
        train_dataset = GTMCorpus(
            model_data_df, embeddings_type=None, prevalence="~ cov", content="~ cov"
        )
        tm = GTM(
            train_dataset,
            doc_topic_prior=doc_topic_prior,
            update_prior=True,
            decoder_type=decoder_type,
            num_epochs=10,
        )
        df_doc_topic = pd.DataFrame(
            tm.get_doc_topic_distribution(train_dataset),
            index=["Doc" + str(i) for i in range(num_docs)],
            columns=["Topic" + str(i) for i in range(num_topics)],
        )
        df_doc_topic_list.append(df_doc_topic)
        df_topic_word = pd.DataFrame(
            tm.get_topic_word_distribution(),
            columns=["word_{}".format(i) for i in range(voc_size)],
            index=["Topic" + str(i) for i in range(num_topics)],
        )
        df_topic_word_list.append(df_topic_word)

        if is_output:
            p = pathlib.Path()
            current_dir = p.cwd()
            if not current_dir.joinpath("data", "gtm").exists():
                current_dir.joinpath("data", "gtm").mkdir()
            name_df_doc_topic = "df_doc_topic_" + str(i) + ".pickle"
            df_doc_topic_path = (
                current_dir.joinpath("data", "gtm", name_df_doc_topic)
                .resolve()
                .as_posix()
            )
            name_df_topic_word = "df_topic_word_" + str(i) + ".pickle"
            df_topic_word_path = (
                current_dir.joinpath("data", "gtm", name_df_topic_word)
                .resolve()
                .as_posix()
            )
            with open(df_doc_topic_path, "wb") as f:
                pickle.dump(df_doc_topic, f)
            with open(df_topic_word_path, "wb") as f:
                pickle.dump(df_topic_word, f)

    return df_doc_topic_list, df_topic_word_list
