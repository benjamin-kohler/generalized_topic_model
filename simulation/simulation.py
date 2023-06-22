# Local Library
from .lda import estimate_dist_by_lda, generate_docs_by_lda

# from .utils import calculate_score


class Simulator:
    """
    A general class for the simulation (document generating processing & \
    estimating the distributions)
    """

    def __init__(
        self,
        model_type,
        num_topics,
        num_iters,
        num_silulations=20,
    ):
        if model_type not in ["lda", "gtp"]:
            raise ValueError("Only two options for topic model: GTP, LDA.")

        self.model_type = model_type
        self.num_topics = num_topics
        self.num_iters = num_iters
        self.num_silulations = num_silulations
        self.docs = None
        self.true_df_doc_topic = None
        self.true_df_topic_word = None
        self.df_doc_topic_list = None
        self.df_topic_word_list = None
        # self.num_words_list = None

    def generate_docs(self, **kwargs):
        """
        input (kwargs)
            alpha=None,
            beta=None,
            doc_args=None,
            is_output=False
        """
        if self.model_type == "lda":
            # df_true_dist_list, docs, num_words_list = generate_docs_by_lda(
            #     num_topics=self.num_topics, num_iters=self.num_iters, **kwargs
            # )
            df_true_dist_list, docs = generate_docs_by_lda(
                num_topics=self.num_topics, num_iters=self.num_iters, **kwargs
            )
        self.docs = docs
        self.true_df_doc_topic = df_true_dist_list[0]
        self.true_df_topic_word = df_true_dist_list[1]
        # self.num_words_list = num_words_list
        self.voc_size = df_true_dist_list[1].shape[1]

    def estimate_distributions(self, **kwargs):
        """
        input (kwargs)
            data,
            num_topics,
            num_silulations,
            model_args=None,
            is_output=False,
        """
        if self.model_type == "lda":
            df_doc_topic_list, df_topic_word_list = estimate_dist_by_lda(
                data=self.docs,
                num_topics=self.num_topics,
                num_iters=self.num_iters,
                num_silulations=self.num_silulations,
                voc_size=self.voc_size,
                **kwargs
            )
        self.df_doc_topic_list = df_doc_topic_list
        self.df_topic_word_list = df_topic_word_list
