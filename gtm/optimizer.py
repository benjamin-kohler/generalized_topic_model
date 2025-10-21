from itertools import product
from utils import topic_diversity, vect2gensim
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from gtm import GTM
import numpy as np
import pandas as pd
import os


class GTMOptimizer:
    """
    Wrapper class to choose the hyperparameters of the Generalized Topic Model via Grid Search.

    /!\ 
    NB: There are many possible hyperparameters to be optimized in the GTM. This can make the grid quickly intractable. 
    /!\
    """

    def __init__(
        self,
        n_topics=[20, 50, 100],
        evaluation_metrics=["diversity", "c_npmi", "c_v", "c_uci", "umass"],
        n_samples=1,
        doc_topic_priors=["logistic_normal"],
        alphas=[0.1],
        encoder_inputs=["bow"],
        encoder_hidden_layers=[[1024,512]],
        encoder_non_linear_activation=["relu"],
        encoder_biases=[True],
        decoder_hidden_layers=[[512,1024]],
        decoder_non_linear_activation=["relu"],
        decoder_biases=[True],
        predictor_hidden_layers=[[]],
        predictor_non_linear_activation=["relu"],
        w_priors=[10],
        w_pred_losses=[1],
        dropout=[0.2],
        gtm_model_args={
            "print_every_n_epochs": 1000000,
            "print_every_n_batches": 1000000,
            "log_every_n_epochs": 1000000,
        },
        topK=10,
        save_folder="gtm_models",
    ):
        """
        Args:
            n_topics: List of integers representing the number of topics to be tested.
            evaluation_metrics: List of strings representing the topic model quality metrics to be tested. Available metrics: "diversity".
            n_samples: Integer representing the number of samples to be drawn from each cell of the hyperparameter grid.
            doc_topic_priors: List of strings representing the prior distribution of the document-topic matrix. Available priors: "dirichlet", "logistic_normal".
            alphas: List of floats representing the alpha parameter of the Dirichlet prior.
            encoder_inputs: List of strings representing the input data to the encoder. Available inputs: "bow", "embeddings".
            encoder_hidden_layers: List of list of integers representing the number of hidden layers in the encoder.
            encoder_non_linear_activation: List of strings representing the non-linear activation functions to be tested in the encoder. Available activations: "relu", "sigmoid", None.
            encoder_biases: List of booleans representing whether to include a bias in the encoder.
            decoder_hidden_layers: List of list of integers representing the number of hidden layers in the decoder.
            decoder_biases: List of booleans representing whether to include a bias in the decoder.
            decoder_non_linear_activation: List of strings representing the non-linear activation functions to be tested in the decoder. Available activations: "relu", "sigmoid", None.
            predictor_hidden_layers: List of list of integers representing the number of hidden layers in the predictor.
            predictor_non_linear_activation: List of strings representing the non-linear activation functions to be tested in the predictor. Available activations: "relu", "sigmoid", None.
            w_priors: List of floats representing the weight of the prior loss.
            w_pred_losses: List of floats representing the weight of the prediction loss.
            gtm_model_args: Additional arguments for the GTM.
            topK: Integer representing the number of words per topic.
            save_folder: String representing the folder where to save the GTM models.
        """

        self.n_topics = n_topics
        self.n_samples = n_samples
        self.doc_topic_priors = doc_topic_priors
        self.alphas = alphas
        self.evaluation_metrics = evaluation_metrics
        self.encoder_inputs = encoder_inputs
        self.encoder_hidden_layers = encoder_hidden_layers
        self.encoder_non_linear_activation = encoder_non_linear_activation
        self.encoder_biases = encoder_biases
        self.decoder_hidden_layers = decoder_hidden_layers
        self.decoder_non_linear_activation = decoder_non_linear_activation
        self.decoder_biases = decoder_biases
        self.predictor_hidden_layers = predictor_hidden_layers
        self.predictor_non_linear_activation = predictor_non_linear_activation
        self.w_priors = w_priors
        self.w_pred_losses = w_pred_losses
        self.dropout = dropout
        self.grid = None
        self.gtm_model_args = gtm_model_args
        self.topK = topK
        self.save_folder = save_folder

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        self.corpus = None
        self.dictionary = None
        self.texts = []

    def create_gensim_objects(self, dataset):
        """
        Converts a GTMCorpus into Gensim Corpus, Dictionary, and Texts. Those are required to compute coherence metrics.
        """
        vectorizer = dataset.vectorizer
        dtmatrix = dataset.M_bow
        (self.corpus, self.dictionary) = vect2gensim(vectorizer, dtmatrix)
        vocab = list(vectorizer.vocabulary_.keys())
        docs = list(dataset.df['doc_clean'])
        docs = [doc.split() for doc in docs]
        for doc in docs: 
            text = [word for word in doc if word in vocab]
            self.texts.append(text)
    
    def optimize(self, train_dataset, test_dataset=None):
        """
        Optimizes the hyperparameters of the Generalized Topic Model via Grid Search.
        """

        if test_dataset is not None: 
            self.create_gensim_objects(test_dataset)
        else:
            self.create_gensim_objects(train_dataset)
        
        all_results = []

        i = 1
        for (
            n_topics,
            doc_topic_prior,
            alpha,
            encoder_input,
            encoder_hidden_layer,
            encoder_non_linear_activation,
            encoder_bias,
            decoder_hidden_layer,
            decoder_non_linear_activation,
            decoder_bias,
            predictor_hidden_layer,
            predictor_non_linear_activation,
            w_prior,  
            w_pred_loss,
            dropout  
        ) in product(
            self.n_topics,
            self.doc_topic_priors,
            self.alphas,
            self.encoder_inputs,
            self.encoder_hidden_layers,
            self.encoder_non_linear_activation,
            self.encoder_biases,
            self.decoder_hidden_layers,
            self.decoder_non_linear_activation,
            self.decoder_biases,
            self.predictor_hidden_layers,
            self.predictor_non_linear_activation,
            self.w_priors,
            self.w_pred_losses,
            self.dropout
        ):
            for _ in range(self.n_samples):
                seed = _
                hyperparameters = {
                    "n_topics": n_topics,
                    "doc_topic_prior": doc_topic_prior,
                    "alpha": alpha,
                    "encoder_input": encoder_input,
                    "encoder_hidden_layers": encoder_hidden_layer,
                    "encoder_non_linear_activation": encoder_non_linear_activation,
                    "encoder_bias": encoder_bias,
                    "decoder_hidden_layers": decoder_hidden_layer,
                    "decoder_non_linear_activation": decoder_non_linear_activation,
                    "decoder_bias": decoder_bias,
                    "predictor_hidden_layers": predictor_hidden_layer,
                    "predictor_non_linear_activation": predictor_non_linear_activation,
                    "w_prior": w_prior,  
                    "w_pred_loss": w_pred_loss,
                    "dropout": dropout 
                }
                print(f"Training GTM with hyperparameters: {hyperparameters}")
                gtm = GTM(
                    train_data=train_dataset,
                    test_data=test_dataset,
                    seed=seed,
                    **hyperparameters,
                    **self.gtm_model_args,
                )
                gtm.save_model(
                    os.path.join(
                        self.save_folder,
                        f"gtm_{n_topics}_{doc_topic_prior}_{alpha}_{encoder_input}_{encoder_hidden_layer}_{encoder_non_linear_activation}_{encoder_bias}_{decoder_hidden_layer}_{decoder_non_linear_activation}_{decoder_bias}_{predictor_hidden_layer}_{predictor_non_linear_activation}_w_prior_{w_prior}_w_pred_loss_{w_pred_loss}_seed_{seed}.ckpt",
                    )
                )       

                result = {
                    "n_topics": n_topics,
                    "doc_topic_prior": doc_topic_prior,
                    "alpha": alpha,
                    "encoder_input": encoder_input,
                    "encoder_hidden_layer": encoder_hidden_layer,
                    "encoder_non_linear_activation": encoder_non_linear_activation,
                    "encoder_bias": encoder_bias,
                    "decoder_hidden_layer": decoder_hidden_layer,
                    "decoder_non_linear_activation": decoder_non_linear_activation,
                    "decoder_bias": decoder_bias,
                    "predictor_hidden_layer": predictor_hidden_layer,
                    "predictor_non_linear_activation": predictor_non_linear_activation,
                    "w_prior":w_prior,
                    "w_pred_loss":w_pred_loss,
                    "seed": seed,
                    "config_id": i,
                    "dropout": dropout,
                    "ckpt_folder": self.save_folder,
                }
                for metric in self.evaluation_metrics:
                    result[metric] = self.evaluate(gtm, metric)
                all_results.append(result)
            i = i + 1

        self.grid = pd.DataFrame.from_records(all_results)

    def evaluate(self, gtm, metric="diversity"):
        """
        Evaluates the quality of the Generalized Topic Model using the specified evaluation metrics.

        Args:
            gtm (GTM): Instance of the Generalized Topic Model to be evaluated.
            metric (str): Evaluation metric.
        Returns:
            float: Evaluation score.
        """
        if metric not in self.evaluation_metrics:
            raise ValueError(f"Metric '{metric}' not found in evaluation metrics.")

        topics = [
            v
            for k, v in gtm.get_topic_words(
                l_content_covariates=[], topK=self.topK
            ).items()
        ]

        if metric == "diversity":
            score = topic_diversity(topics, topK=self.topK)
        elif metric == "c_npmi": 
            cm = CoherenceModel(topics=topics, texts=self.texts, dictionary=self.dictionary, coherence='c_npmi')
            score = cm.get_coherence()
        elif metric == "c_v":
            cm = CoherenceModel(topics=topics, texts=self.texts, dictionary=self.dictionary, coherence='c_v')
            score = cm.get_coherence()
        elif metric == "c_uci":
            cm = CoherenceModel(topics=topics, texts=self.texts, dictionary=self.dictionary, coherence='c_uci')
            score = cm.get_coherence()
        elif metric == "prediction_loss":
            score = -1.0*gtm.prediction_loss
        else:
            cm = CoherenceModel(topics=topics, corpus=self.corpus, dictionary=self.dictionary, coherence='u_mass')
            score = cm.get_coherence()    
        return score

    def plot_evaluation_metric(
        self, metric="diversity", display=True, output_path=None
    ):
        """
        Plots boxplots for the given evaluation metric across different hyperparameter configurations.
        """
        if metric not in self.evaluation_metrics:
            raise ValueError(f"Metric '{metric}' not found in evaluation metrics.")

        data_by_config = []
        for i in range(1, self.grid["config_id"].max() + 1):
            config_data = self.grid[self.grid["config_id"] == i]
            metrics_list = config_data[metric].tolist()
            data_by_config.append(metrics_list)

        max_metric_config = (
            np.argmax([np.mean(metrics) for metrics in data_by_config]) + 1
        )

        plt.figure(figsize=(12, 8))
        bp = plt.boxplot(
            data_by_config,
            labels=range(1, len(data_by_config) + 1),
            vert=True,
            patch_artist=True,
        )

        for patch in bp["boxes"]:
            patch.set_alpha(0.5)

        for box in bp["boxes"]:
            box.set(color="grey", linewidth=2)
            box.set(facecolor="grey")

        bp["boxes"][max_metric_config - 1].set(color="green", facecolor="green")

        metric_uppercase = metric[0].upper() + metric[1:]

        plt.title(
            f"Boxplots for {metric_uppercase} across Hyperparameter Configurations"
        )
        plt.xlabel("Configuration ID")
        plt.ylabel(metric_uppercase)
        plt.grid(True)
        if output_path:
            plt.savefig(output_path)
        if display:
            plt.show()

    def get_best_model(self, metric="coherence"):
        """
        Returns the best model according to the specified evaluation metric.

        Args:
            metric (str): Evaluation metric.

        Returns:
            dict: Best model hyperparameters and corresponding score.
        """

        best_model_params = self.grid[
            self.grid[metric] == self.grid[metric].max()
        ].to_dict("records")[0]
        gtm = GTM(
            ckpt=os.path.join(
                self.save_folder,
                f"gtm_{best_model_params['n_topics']}_{best_model_params['doc_topic_prior']}_{best_model_params['alpha']}_{best_model_params['encoder_input']}_{best_model_params['encoder_hidden_layer']}_{best_model_params['encoder_non_linear_activation']}_{best_model_params['encoder_bias']}_{best_model_params['decoder_hidden_layer']}_{best_model_params['decoder_non_linear_activation']}_{best_model_params['decoder_bias']}_{best_model_params['predictor_hidden_layer']}_{best_model_params['predictor_non_linear_activation']}_w_prior_{best_model_params['w_prior']}_w_pred_loss_{best_model_params['w_pred_loss']}_seed_{best_model_params['seed']}.ckpt",
            )
        )
        return best_model_params, gtm