# Standard Library
import re

# Third Party Library
import gensim

# import gensim.corpora as corpora
import nltk
import spacy
from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words = stopwords.words("english")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def data_preprocessing(data):
    """
    function for preprocessing the data
    input
        data [list]
    output
        data [list]

    1. removing specific characters
    2. removing stopwords
    3. making bigrams
    4. implementing lemmatizing
    """

    def _remove_email_newline_char(data):
        data = [re.sub(r"\S*@\S*\s?", "", sent) for sent in data]
        data = [re.sub(r"\s+", " ", sent) for sent in data]
        data = [re.sub(r"'", "", sent) for sent in data]

        return data

    def _sent_to_words(sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))

    def _remove_stopwords(texts):
        return [
            [
                word
                for word in gensim.utils.simple_preprocess(str(doc))
                if word not in stop_words
            ]
            for doc in texts
        ]

    def _make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def _lemmatization(texts, allowed_postags):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append(
                [token.lemma_ for token in doc if token.pos_ in allowed_postags]
            )
        return texts_out

    data = _remove_email_newline_char(data)
    data_words = list(_sent_to_words(data))

    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    data_words_nostops = _remove_stopwords(data_words)
    data_words_bigrams = _make_bigrams(data_words_nostops)

    data_lemmatized = _lemmatization(
        data_words_bigrams, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]
    )
    data_lemmatized = [" ".join(i) for i in data_lemmatized]
    # id2word = corpora.Dictionary(data_lemmatized)
    # corpus = [id2word.doc2bow(text) for text in data_lemmatized]

    return data_lemmatized


# def data_preprocessing_for_artificial_docs(data):
#     """
#     function for preprocessing the data
#     input
#         data [list]
#     output
#         corpus [list[list][tuple]]
#         id2word [gensim.corpora.dictionary.Dictionary]

#     """
#     input_for_creating_dict = [i.split(" ") for i in data]
#     id2word = corpora.Dictionary(input_for_creating_dict)
#     corpus = [id2word.doc2bow(text) for text in input_for_creating_dict]
#     return corpus, id2word
