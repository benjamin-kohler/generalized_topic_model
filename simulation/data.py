#
# このファイルは不要な可能性が高い
#

# Standard Library
import pathlib
import pickle
import random

# Third Party Library
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

# Local Library
from .preprocessing import data_preprocessing

random.seed(123)


def _create_docs(
    num_min, num_max, num_doc, num_dictionaries=5000, is_output=False, output_path=None
):
    """
    Creating dataset from random artificial words


    """
    doc_list = []
    p = pathlib.Path()
    random_corpus = [
        "word_" + str(random.randint(0, num_dictionaries))
        for i in range(num_dictionaries)
    ]
    for _ in range(num_doc):
        num_words = random.randint(num_min, num_max)
        sentence_list = [random.choice(random_corpus) for _ in range(num_words)]
        sentence = " ".join(sentence_list)
        doc_list.append(sentence)
    if is_output:
        if not output_path:
            current_dir = p.cwd()
            if not current_dir.joinpath("..", "data").exists():
                current_dir.joinpath("..", "data").mkdir()
            file_path = current_dir.joinpath("original_doc.pickle").resolve()
            output_path = file_path.as_posix()
        with open(output_path, "wb") as f:
            pickle.dump(doc_list, f)

    return doc_list


def _preparing_20newsgroups_dataset(subset="train"):
    fetch_data = fetch_20newsgroups(subset=subset)
    # targret_names_list = fetch_data["target_names"]
    # ref_dict = {i: name for i, name in enumerate(targret_names_list)}
    # targret_names_col = [ref_dict[num] for num in fetch_data["target"]]
    # df = pd.DataFrame(
    #     {
    #         "data": fetch_data["data"],
    #         "target": fetch_data["target"],
    #         "target_names": targret_names_col,
    #     }
    # )
    df = pd.DataFrame(
        {
            "data": fetch_data["data"],
        }
    )
    data = df.data.values.tolist()

    return data


def create_data(data_type="artificial", data_args=None, is_preprocessing=False):
    if data_type == "artificial":
        if data_args is None:
            data_args = {"num_min": 50, "num_max": 100, "num_doc": 5000}
        data = _create_docs(**data_args)
    elif data_type == "news20":
        if data_args is None:
            data_args = {"subset": "train"}
        data = _preparing_20newsgroups_dataset(**data_args)
        # TODO data = _create_docs_from_news20(**data_args)
        if is_preprocessing:
            data = data_preprocessing(data)
    else:
        raise ValueError(
            "Only one option for topic model: news20, \
                artificial(create from scratch)."
        )
    return data
