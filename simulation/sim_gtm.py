# Standard Library
import pathlib
import pickle

# Third Party Library
import numpy as np
import pandas as pd
from corpus import GTMCorpus
from patsy import dmatrix
from tqdm import tqdm

# First Party Library
from gtm import GTM


def generate_docs_by_gtm(
    num_topics,
    doc_topic_prior,
    decoder_type,
    seed,
    update_prior=False,
    lambda_=None,
    sigma=None,
    doc_args=None,
    is_output=True,
    decoder_estimate_interactions=False,  # for SAGE
):
    np.random.seed(seed)

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
        "num_content_covs": 2,
        "num_prev_covs": 2,
        "min_words": 50,
        "max_words": 100,
        "num_docs": 5000,
        "voc_size": 1000,
    }
    if doc_args is None:
        doc_args = default_data_args_dict

    # def _create_categorical_data(cov_nums, num_docs):
    #     categories = [i for i in range(cov_nums)]
    #     categories_of_each_doc = np.random.choice(
    #         categories, size=num_docs, replace=True
    #     ).tolist()
    #     df_categ = pd.DataFrame({"cov": categories_of_each_doc})
    #     categorical_covariates = dmatrix("~cov", data=df_categ)
    #     return categories_of_each_doc, categorical_covariates

    def _create_categorical_data(cov_nums, num_docs):
        categories = [i for i in range(cov_nums)]
        categories_of_each_doc = np.random.choice(
            categories, size=num_docs, replace=True
        ).tolist()
        onehot_covariates = np.eye(len(categories))[categories_of_each_doc]
        return categories_of_each_doc, onehot_covariates

    if update_prior:
        (
            prevalence_categories_of_each_doc,
            prevalence_covariates,
        ) = _create_categorical_data(doc_args["num_prev_covs"], doc_args["num_docs"])
        content_categories_of_each_doc, content_covariates = _create_categorical_data(
            doc_args["num_content_covs"], doc_args["num_docs"]
        )
        if lambda_ is None:
            # lambda_ = 2 * np.random.rand(doc_args["num_content_covs"], num_topics) - 1
            lambda_ = np.random.rand(doc_args["num_content_covs"], num_topics)
            lambda_ = lambda_ - lambda_[:, 0][:, None]

    # for doc_topic matrix
    if doc_topic_prior == "logistic_normal":
        if sigma is None:
            sqrt_sigma = np.random.rand(num_topics, num_topics)
            sigma = sqrt_sigma * sqrt_sigma.T
        doc_topic_raw_list = []
        for i in range(doc_args["num_docs"]):
            if update_prior:
                mean = np.exp(np.dot(prevalence_covariates[i, :].T, lambda_))
            else:
                mean = np.array([0.1 for _ in range(num_topics)])
            doc_topic_raw_list.append(np.random.multivariate_normal(mean, sigma))
        doc_topic_pro_raw = np.array(doc_topic_raw_list)
        doc_topic_pro = np.exp(doc_topic_pro_raw) / np.sum(
            np.exp(doc_topic_pro_raw), axis=1, keepdims=True
        )
    else:  # dirichlet
        doc_topic_pro_list = []
        for i in range(doc_args["num_docs"]):
            if update_prior:
                diri_params = np.exp(np.dot(prevalence_covariates[i, :].T, lambda_))
            else:
                diri_params = np.array([0.1 for _ in range(num_topics)])
            doc_topic_pro_list.append(np.random.dirichlet(diri_params))
        doc_topic_pro = np.array(doc_topic_pro_list)

    # for baseline topic_word matrix
    topic_word_pro = np.random.dirichlet(
        [0.1 for _ in range(doc_args["voc_size"])], num_topics
    )

    # for doc_word matrix
    if decoder_type == "sage":
        if update_prior:
            if decoder_estimate_interactions:
                b = np.random.rand(doc_args["num_docs"], doc_args["voc_size"])
                cov_word_freq = np.random.randint(
                    low=0,
                    high=2,
                    size=(doc_args["num_content_covs"], doc_args["voc_size"]),
                )
                cov_top_word_dev = np.random.rand(
                    doc_args["voc_size"], doc_args["num_content_covs"] * num_topics
                )
                doc_word_pro_raw = (
                    b
                    + np.dot(doc_topic_pro, topic_word_pro)
                    + np.dot(content_covariates, cov_word_freq)
                    + np.dot(
                        np.array(
                            [
                                np.kron(row_dt, row_cc).reshape(-1)
                                for row_dt, row_cc in zip(
                                    doc_topic_pro, content_covariates
                                )
                            ]
                        ),
                        cov_top_word_dev.T,
                    )
                )
            else:
                doc_word_pro_raw = np.dot(doc_topic_pro, topic_word_pro)
        else:
            doc_word_pro_raw = np.dot(doc_topic_pro, topic_word_pro)
    else:  # MLP
        num_embeddings = doc_args.get("num_embeddings", 300)
        rho = np.array(np.random.rand(doc_args["voc_size"], num_embeddings))
        alpha = np.array(np.random.rand(num_topics, num_embeddings))
        if update_prior:
            phi = np.array(np.random.rand(doc_args["num_content_covs"], num_embeddings))
            doc_word_pro_raw = np.dot(
                (np.dot(doc_topic_pro, alpha) + np.dot(content_covariates, phi)), rho.T
            )
        else:
            doc_word_pro_raw = np.dot(np.dot(doc_topic_pro, alpha), rho.T)
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

    # docs = []
    # for docname in tqdm(docnames):
    #     doc = []
    #     num_words = np.random.randint(
    #         low=doc_args["min_words"],
    #         high=doc_args["max_words"] + 1,
    #     )
    #     top_pro = df_doc_topic.loc[docname, :]
    #     topic_list = [np.random.choice(topicnames, p=top_pro) for _ in range(num_words)]
    #     for topic in topic_list:
    #         word_pro = df_topic_word.loc[topic, :]
    #         word = np.random.choice(words, p=word_pro)
    #         doc.append(word)
    #     docs.append(" ".join(doc))

    docs = []
    for docname in tqdm(docnames):
        num_words = np.random.randint(
            low=doc_args["min_words"],
            high=doc_args["max_words"] + 1,
        )
        word_pro_per_doc = df_doc_word.loc[docname, :]
        docs.append(
            " ".join(
                [np.random.choice(words, p=word_pro_per_doc) for _ in range(num_words)]
            )
        )

    if update_prior:
        original_dataset_dict = {
            "doc": docs,
            "prevalence_covariates": prevalence_categories_of_each_doc,
            "content_covariates": content_categories_of_each_doc,
        }
    else:
        original_dataset_dict = {"doc": docs}
    if is_output:
        p = pathlib.Path()
        current_dir = p.cwd()
        if not current_dir.joinpath("..", "data").exists():
            current_dir.joinpath("..", "data").mkdir()
        if not current_dir.joinpath("..", "data", "gtm").exists():
            current_dir.joinpath("..", "data", "gtm").mkdir()
        # if decoder_estimate_interactions:
        #     file_name = "{}_{}_int".format(doc_topic_prior, decoder_type)
        # else:
        file_name = "{}_{}".format(doc_topic_prior, decoder_type)

        true_df_doc_topic_path = (
            current_dir.joinpath(
                "..",
                "data",
                "gtm",
                "true_df_doc_topic_{}.pickle".format(file_name),
            )
            .resolve()
            .as_posix()
        )
        true_df_topic_word_path = (
            current_dir.joinpath(
                "..",
                "data",
                "gtm",
                "true_df_topic_word_{}.pickle".format(file_name),
            )
            .resolve()
            .as_posix()
        )
        true_df_doc_word_path = (
            current_dir.joinpath(
                "..",
                "data",
                "gtm",
                "true_df_doc_word_{}.pickle".format(file_name),
            )
            .resolve()
            .as_posix()
        )
        original_dataset_path = (
            current_dir.joinpath(
                "..",
                "data",
                "gtm",
                "original_dataset_{}.pickle".format(file_name),
            )
            .resolve()
            .as_posix()
        )
        true_lambda_path = (
            current_dir.joinpath(
                "..",
                "data",
                "gtm",
                "true_lambda_{}.pickle".format(file_name),
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
        with open(original_dataset_path, "wb") as f:
            pickle.dump(original_dataset_dict, f)
        with open(true_lambda_path, "wb") as f:
            pickle.dump(lambda_, f)

    return df_true_dist_list, original_dataset_dict


def estimate_dist_by_gtm(
    original_dataset_dict,
    num_topics,
    num_docs,
    voc_size,
    embeddings_type=None,
    model_args=None,
    is_output=True,
):
    # TODO setting the default args
    default_model_args_dict = {
        "num_epochs": 10,
        "update_prior": True,
        "decoder_estimate_interactions": True,
        "doc_topic_prior": "dirichlet",
        "decoder_type": "mlp",
    }
    if model_args is None:
        model_args = default_model_args_dict

    df_doc_topic_list, df_topic_word_list = [], []
    model_data_df = pd.DataFrame(
        {
            "doc": original_dataset_dict["doc"],
            "doc_clean": original_dataset_dict["doc"],
            "prevalence": original_dataset_dict["prevalence_covariates"],
            "content": original_dataset_dict["content_covariates"],
        }
    )
    train_dataset = GTMCorpus(
        model_data_df,
        embeddings_type=embeddings_type,
        prevalence="~ prevalence",
        content="~ content",
    )
    tm = GTM(train_dataset, **model_args)
    df_doc_topic = pd.DataFrame(
        tm.get_doc_topic_distribution(train_dataset),
        index=["Doc{}".format(i) for i in range(num_docs)],
        columns=["Topic{}".format(i) for i in range(num_topics)],
    )
    df_doc_topic_list.append(df_doc_topic)
    df_topic_word = pd.DataFrame(
        tm.get_topic_word_distribution(voc_size),
        index=["Topic{}".format(i) for i in range(num_topics)],
        columns=["word_{}".format(i) for i in range(voc_size)],
    )
    df_topic_word_list.append(df_topic_word)

    if is_output:
        p = pathlib.Path()
        current_dir = p.cwd()
        if not current_dir.joinpath("..", "data", "gtm").exists():
            current_dir.joinpath("..", "data", "gtm").mkdir()
        # if model_args.get("decoder_estimate_interactions", False):
        #     file_name = "{}_{}_int".format(
        #         model_args["doc_topic_prior"], model_args["decoder_type"]
        #     )
        # else:
        file_name = "{}_{}".format(
            model_args["doc_topic_prior"], model_args["decoder_type"]
        )

        name_df_doc_topic = "df_doc_topic_{}.pickle".format(file_name)
        df_doc_topic_path = (
            current_dir.joinpath("..", "data", "gtm", name_df_doc_topic)
            .resolve()
            .as_posix()
        )
        name_df_topic_word = "df_topic_word_{}.pickle".format(file_name)
        df_topic_word_path = (
            current_dir.joinpath("..", "data", "gtm", name_df_topic_word)
            .resolve()
            .as_posix()
        )
        name_train_dataset = "train_dataset_{}.pickle".format(file_name)
        train_dataset_path = (
            current_dir.joinpath("..", "data", "gtm", name_train_dataset)
            .resolve()
            .as_posix()
        )
        estimated_lambda_path = (
            current_dir.joinpath(
                "..",
                "data",
                "gtm",
                "estimated_lambda_{}.pickle".format(file_name),
            )
            .resolve()
            .as_posix()
        )
        with open(df_doc_topic_path, "wb") as f:
            pickle.dump(df_doc_topic, f)
        with open(df_topic_word_path, "wb") as f:
            pickle.dump(df_topic_word, f)
        with open(train_dataset_path, "wb") as f:
            pickle.dump(train_dataset, f)
        with open(estimated_lambda_path, "wb") as f:
            pickle.dump(tm.prior.lambda_, f)

    return df_doc_topic_list, df_topic_word_list
