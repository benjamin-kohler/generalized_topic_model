# Standard Library
import pathlib
import pickle

# Third Party Library
import gensim
import gensim.corpora as corpora
import numpy as np
import pandas as pd
from tqdm import tqdm


def generate_docs_by_lda(
    num_topics,
    seed,
    alpha=None,
    beta=None,
    doc_args=None,
    is_output=True,
):
    np.random.seed(seed)
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
    if alpha is None:
        alpha = [0.1 for _ in range(num_topics)]
    if beta is None:
        beta = [0.1 for _ in range(doc_args["voc_size"])]

    topicnames = ["Topic" + str(i) for i in range(num_topics)]
    docnames = ["Doc" + str(i) for i in range(doc_args["num_docs"])]
    words = ["word_" + str(i) for i in range(doc_args["voc_size"])]

    doc_topic_pro = np.random.dirichlet(alpha, doc_args["num_docs"])
    df_doc_topic = pd.DataFrame(doc_topic_pro, columns=topicnames, index=docnames)
    topic_word_pro = np.random.dirichlet(beta, num_topics)
    df_topic_word = pd.DataFrame(topic_word_pro, index=topicnames, columns=words)

    df_true_dist_list = [df_doc_topic, df_topic_word]

    docs = []
    for docname in tqdm(docnames):
        sentence = []
        num_words = np.random.randint(
            default_data_args_dict["min_words"], default_data_args_dict["max_words"]
        )
        top_pro = df_doc_topic.loc[docname, :]
        topic_list = [np.random.choice(topicnames, p=top_pro) for _ in range(num_words)]
        for topic in topic_list:
            word_pro = df_topic_word.loc[topic, :]
            word = np.random.choice(words, p=word_pro)
            sentence.append(word)
        docs.append(" ".join(sentence))

    if is_output:
        p = pathlib.Path()
        current_dir = p.cwd()

        if not current_dir.joinpath("..", "data").exists():
            current_dir.joinpath("..", "data").mkdir()

        if not current_dir.joinpath("..", "data", "lda").exists():
            current_dir.joinpath("..", "data", "lda").mkdir()
        true_df_doc_topic_path = (
            current_dir.joinpath("..", "data", "lda", "true_df_doc_topic.pickle")
            .resolve()
            .as_posix()
        )
        true_df_topic_word_path = (
            current_dir.joinpath("..", "data", "lda", "true_df_topic_word.pickle")
            .resolve()
            .as_posix()
        )
        original_docs_path = (
            current_dir.joinpath("..", "data", "lda", "original_docs.pickle")
            .resolve()
            .as_posix()
        )
        with open(true_df_doc_topic_path, "wb") as f:
            pickle.dump(df_doc_topic, f)
        with open(true_df_topic_word_path, "wb") as f:
            pickle.dump(df_topic_word, f)
        with open(original_docs_path, "wb") as f:
            pickle.dump(docs, f)

    return df_true_dist_list, docs


def _create_input_for_lda_from_generated_docs(data):
    """
    input
        generated_new_docs[list]
    output
        generated_id2word, generated_corpus

    """
    split_data = [sentence.split(" ") for sentence in data]

    id2word = corpora.Dictionary(split_data)
    corpus = [id2word.doc2bow(text) for text in split_data]

    return id2word, corpus


def estimate_dist_by_lda(
    data,
    num_topics,
    voc_size,
    model_args=None,
    is_output=True,
):
    default_model_args_dict = {
        "update_every": 1,
        "chunksize": 100,
        "passes": 10,
        "alpha": 0.1,
        "eta": 0.1,
        "per_word_topics": True,
    }
    if model_args is None:
        model_args = default_model_args_dict
    df_doc_topic_list, df_topic_word_list = [], []

    def _estimate_doctopic_dist_by_lda(lda_model, corpus):
        dist_doc_topic = []
        for doc in corpus:
            temp_list = [0 for i in range(lda_model.num_topics)]
            dist = lda_model.get_document_topics(doc)
            for temp_set in dist:
                topic, val = int(temp_set[0]), float(temp_set[1])
                temp_list[topic] = val
            dist_doc_topic.append(temp_list)
        topicnames = ["Topic{}".format(i) for i in range(lda_model.num_topics)]
        docnames = ["Doc{}".format(i) for i in range(len(corpus))]
        df_doc_topic = pd.DataFrame(dist_doc_topic, columns=topicnames, index=docnames)

        return df_doc_topic

    def _estimate_topicword_dist_by_lda(lda_model, id2word, voc_size):
        df_topic_word = pd.DataFrame(
            lda_model.get_topics(),
            columns=[id2word[i] for i in id2word],
            index=["Topic{}".format(i) for i in range(lda_model.num_topics)],
        )
        col_words = ["word_{}".format(i) for i in range(voc_size)]

        df_topic_word_reordered = df_topic_word.reindex(columns=col_words).fillna(0)

        return df_topic_word_reordered

    id2word, corpus = _create_input_for_lda_from_generated_docs(data)
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        **model_args,
    )
    df_doc_topic = _estimate_doctopic_dist_by_lda(lda_model, corpus)
    df_doc_topic_list.append(df_doc_topic)
    df_topic_word = _estimate_topicword_dist_by_lda(lda_model, id2word, voc_size)
    df_topic_word_list.append(df_topic_word)
    if is_output:
        p = pathlib.Path()
        current_dir = p.cwd()
        if not current_dir.joinpath("..", "data", "lda").exists():
            current_dir.joinpath("..", "data", "lda").mkdir()
        name_df_doc_topic = "df_doc_topic.pickle"
        df_doc_topic_path = (
            current_dir.joinpath("..", "data", "lda", name_df_doc_topic)
            .resolve()
            .as_posix()
        )
        name_df_topic_word = "df_topic_word.pickle"
        df_topic_word_path = (
            current_dir.joinpath("..", "data", "lda", name_df_topic_word)
            .resolve()
            .as_posix()
        )

        with open(df_doc_topic_path, "wb") as f:
            pickle.dump(df_doc_topic, f)
        with open(df_topic_word_path, "wb") as f:
            pickle.dump(df_topic_word, f)

    return df_doc_topic_list, df_topic_word_list
