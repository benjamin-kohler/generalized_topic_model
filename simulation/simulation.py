# First Party Library
from sim_gtm import estimate_dist_by_gtm, generate_docs_by_gtm
from sim_lda import estimate_dist_by_lda, generate_docs_by_lda


class Simulator:
    """
    A general class for the simulation (document generating processing & \
    estimating the distributions)
    """

    def __init__(
        self,
        model_type,
        num_topics,
        num_covs,
        num_silulations=100,
    ):
        if model_type not in ["lda", "gtm"]:
            raise ValueError("Only two options for topic model: gtm, lda.")

        self.model_type = model_type
        self.num_topics = num_topics
        self.num_silulations = num_silulations
        self.num_covs = num_covs
        self.num_docs = None
        self.voc_size = None
        self.docs = None
        self.true_df_doc_topic = None
        self.true_df_topic_word = None
        self.df_doc_topic_list = None
        self.df_topic_word_list = None

    def generate_docs(self, **kwargs):
        """
        input (kwargs)
            alpha=None,
            beta=None,
            doc_args=None,
            is_output=False
        """
        if self.model_type == "lda":
            df_true_dist_list, docs = generate_docs_by_lda(
                num_topics=self.num_topics, **kwargs
            )
        else:
            df_true_dist_list, docs = generate_docs_by_gtm(
                num_topics=self.num_topics, num_covs=self.num_covs, **kwargs
            )
        self.docs = docs
        self.true_df_doc_topic = df_true_dist_list[0]
        self.true_df_topic_word = df_true_dist_list[1]
        self.num_docs = df_true_dist_list[0].shape[0]
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
                num_silulations=self.num_silulations,
                voc_size=self.voc_size,
                **kwargs,
            )
        else:
            df_doc_topic_list, df_topic_word_list = estimate_dist_by_gtm(
                data=self.docs,
                num_topics=self.num_topics,
                num_silulations=self.num_silulations,
                voc_size=self.voc_size,
                num_docs=self.num_docs,
                **kwargs,
            )
        self.df_doc_topic_list = df_doc_topic_list
        self.df_topic_word_list = df_topic_word_list
