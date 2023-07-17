# Standard Library
import pathlib
import pickle

# Third Party Library
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def matching_topic(model_type, matching_by, **kwargs):
    """
    function for matching the true matrix and the estimated matrix \
    (by cosine similarity or correlation coefficient)
        1. column of doc-topic matrix
        2. the row of topic-word matrix

    output:

        corres_num_topic_dict
            {num_sim : {true_topic_idx: estimated_topic_name}}
    """

    # TODO check if this assignment is fine, especially top10 keywords approach
    if matching_by in ["correlation", "cossim", "dot_product"]:
        dist_type = "doc_topic"
    elif matching_by == "keyword":
        dist_type = "topic_word"
    else:
        raise ValueError(
            "Only four options for soring: correlation, cossim, and dot_product, keyword."
        )

    if model_type not in ["lda", "gtm"]:
        raise ValueError("Only two options for topic model: gtm, lda.")

    p = pathlib.Path()
    current_dir = p.cwd()

    corres_num_topic_dict = {}

    model_type = "{}".format(model_type)
    if model_type == "lda":
        true_df_dist_name = "true_df_{}.pickle".format(dist_type)

    else:
        true_df_dist_name = "true_df_{}_{}_{}.pickle".format(
            dist_type, kwargs["doc_topic_prior"], kwargs["decoder_type"]
        )
    true_path = (
        current_dir.joinpath(
            "..",
            "data",
            model_type,
            true_df_dist_name,
        )
        .resolve()
        .as_posix()
    )

    with open(true_path, "rb") as f:
        true_df = pickle.load(f)

    if model_type == "lda":
        estimated_df_dist_name = "df_{}.pickle".format(dist_type)

    else:
        estimated_df_dist_name = "df_{}_{}_{}.pickle".format(
            dist_type, kwargs["doc_topic_prior"], kwargs["decoder_type"]
        )
    estimated_path = (
        current_dir.joinpath(
            "..",
            "data",
            model_type,
            estimated_df_dist_name,
        )
        .resolve()
        .as_posix()
    )
    with open(estimated_path, "rb") as f:
        estimated_df = pickle.load(f)
    if dist_type == "doc_topic":
        score_list = []
        for true_col in true_df.columns:
            true_target_col = true_df.loc[:, true_col]
            score_list_per_row = []
            for col in estimated_df.columns:
                target_col = estimated_df.loc[:, col]
                if matching_by == "correlation":
                    score_list_per_row.append(
                        np.corrcoef(target_col, true_target_col)[0][1]
                    )
                elif matching_by == "cossim":
                    score_list_per_row.append(
                        np.dot(target_col.T, true_target_col)
                        / (np.linalg.norm(target_col) * np.linalg.norm(true_target_col))
                    )
                else:
                    score_list_per_row.append(np.dot(target_col, true_target_col))
            score_list.append(score_list_per_row)
    else:
        # TODO check if this logic is fine
        score_list = []
        true_topic_keywords_dict, estimated_topic_keywords_dict = {}, {}
        for index in true_df.index:
            true_word = true_df.loc[index, :]
            score_list_per_row = []
            true_topic_keywords_dict[index] = list(
                true_word.sort_values(ascending=False).index
            )[:10]
            estimated_word = estimated_df.loc[index, :]
            estimated_topic_keywords_dict[index] = list(
                estimated_word.sort_values(ascending=False).index
            )[:10]
            estimated_word = estimated_df.loc[index, :]
        for t_topic in true_topic_keywords_dict.keys():
            score_list_per_index = []
            true_words = true_topic_keywords_dict[t_topic]
            for e_topic in estimated_topic_keywords_dict.keys():
                words = estimated_topic_keywords_dict[e_topic]
                counter = 0
                for word in words:
                    if word in true_words:
                        counter += 1
                    score_list_per_index.append(counter / 10)
            score_list.append(score_list_per_index)

    score_matrix = pd.DataFrame(score_list)
    true_topics, estimated_topics = linear_sum_assignment(-score_matrix)

    for true_topic, estimated_topic in zip(true_topics, estimated_topics):
        corres_num_topic_dict["Topic{}".format(true_topic)] = "Topic{}".format(
            estimated_topic
        )

    return corres_num_topic_dict


def calculate_score(model_type, score_type, corres_num_topic_dict, **kwargs):
    """
    A function for calculating the scoring
    input
        score_type: "correlation", "euclid", "cossim", "keywords"

        corres_num_topic_dict (output of creating_dict_for_topic_correspondence)
            {estimated_topic_idx: true_topic}
    """
    if score_type in ["correlation", "euclid", "cossim"]:
        dist_type = "doc_topic"
    elif score_type == "keywords":
        dist_type = "topic_word"
    else:
        raise ValueError(
            "Only four options for scoring similarities: \
        correlation, euclid, cossim, and keywords."
        )

    def _rearange_estimated_df(
        df,
        corres_num_topic_dict,
        dist_type,
    ):
        """
        inner function for rearanging an estimated matrix referencing the true matrix
        """

        if dist_type == "doc_topic":
            reanged_df = df.loc[:, corres_num_topic_dict.values()]
            reanged_df.columns = corres_num_topic_dict.keys()
        else:
            reanged_df = df.loc[corres_num_topic_dict.values(), :]
            reanged_df.index = corres_num_topic_dict.keys()

        return reanged_df

    def _calc_score_from_two_df(score_type, df1, df2):
        """
        df1 should be true matrix
        df2 should be estimated matrix
        """
        if score_type == "correlation":
            res = df1.corrwith(df2, axis=0).to_list()
        elif score_type == "euclid":
            res = []
            for col in df1.columns:
                series_1 = df1.loc[:, col]
                series_2 = df2.loc[:, col]
                res.append(np.linalg.norm(series_1 - series_2))
        elif score_type == "cossim":
            res = []
            for col in df1.columns:
                series_1 = df1.loc[:, col]
                series_2 = df2.loc[:, col]
                res.append(
                    np.dot(series_1.T, series_2)
                    / (np.linalg.norm(series_1) * np.linalg.norm(series_2))
                )
        else:
            # top 10 keywords approach
            res = []
            df1_topic_keywords_dict, df2_topic_keywords_dict = {}, {}
            for index in df1.index:
                words_1 = df1.loc[index, :]
                df1_topic_keywords_dict[index] = list(
                    words_1.sort_values(ascending=False).index
                )[:10]
                words_2 = df2.loc[index, :]
                df2_topic_keywords_dict[index] = list(
                    words_2.sort_values(ascending=False).index
                )[:10]
            for t_topic in df1_topic_keywords_dict.keys():
                true_words = df1_topic_keywords_dict[t_topic]
                words = df2_topic_keywords_dict[t_topic]
                counter = 0
                for word in words:
                    if word in true_words:
                        counter += 1
                res.append(counter / 10)

        return res

    p = pathlib.Path()
    current_dir = p.cwd()

    if model_type == "lda":
        true_df_dist_name = "true_df_{}.pickle".format(dist_type)
    else:
        true_df_dist_name = "true_df_{}_{}_{}.pickle".format(
            dist_type, kwargs["doc_topic_prior"], kwargs["decoder_type"]
        )
    true_df_dist_path = (
        current_dir.joinpath("..", "data", model_type, true_df_dist_name)
        .resolve()
        .as_posix()
    )
    with open(true_df_dist_path, "rb") as f:
        true_df_dist = pickle.load(f)

    if model_type == "lda":
        target_df_dist_name = "df_{}.pickle".format(
            dist_type,
        )
    else:
        target_df_dist_name = "df_{}_{}_{}.pickle".format(
            dist_type, kwargs["doc_topic_prior"], kwargs["decoder_type"]
        )
    target_df_dist_path = (
        current_dir.joinpath(
            "..",
            "data",
            model_type,
            target_df_dist_name,
        )
        .resolve()
        .as_posix()
    )
    with open(target_df_dist_path, "rb") as f:
        estimated_df_dist = pickle.load(f)

    reanged_df = _rearange_estimated_df(
        df=estimated_df_dist,
        corres_num_topic_dict=corres_num_topic_dict,
        dist_type=dist_type,
    )
    res = _calc_score_from_two_df(
        score_type=score_type, df1=true_df_dist, df2=reanged_df
    )

    return res
