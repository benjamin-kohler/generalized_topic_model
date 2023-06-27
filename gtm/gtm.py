#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# Standard Library
import time

# Third Party Library
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from autoencoders import AutoEncoderMLP, AutoEncoderSAGE
from priors import DirichletPrior, LogisticNormalPrior
from torch.utils.data import DataLoader

# First Party Library
from utils import compute_mmd_loss, compute_reconstruction_loss

# TO-DO:
# multilingual tests
# add method to update prior conditional on prevalence covariates
# run on simulations (Shimpei)
# run on standard evaluation datasets (Shimpei)
# integrate with OCTIS
# application to the U.S. congressional record


class GTM:
    """
    Wrapper class for the Generalized Topic Model.
    """

    def __init__(
        self,
        train_data,
        n_topics=20,
        doc_topic_prior="dirichlet",
        update_prior=False,
        alpha=0.1,
        prevalence_covariates_regularization=0,
        encoder_input="bow",
        encoder_hidden_layers=[1024, 512],
        encoder_non_linear_activation="relu",
        encoder_bias=True,
        decoder_type="mlp",
        decoder_hidden_layers=[300],
        decoder_non_linear_activation=None,
        decoder_bias=False,
        decoder_sparsity_regularization=0.1,
        num_epochs=10,
        batch_size=256,
        learning_rate=1e-3,
        dropout=0.2,
        log_every=5,
        tightness=1,
        ckpt=None,
        device=None,
        seed=42,
    ):
        """
        Args:
            train_data: a GTMCorpus object
            n_topics: number of topics
            doc_topic_prior: prior on the document-topic distribution. Either 'dirichlet' or 'logistic_normal'.
            update_prior: whether to update the prior at each epoch to account for prevalence covariates.
            alpha: parameter of the Dirichlet prior (only used if update_prior=False)
            prevalence_covariates_regularization: regularization parameter for the logistic normal prior (only used if update_prior=True)
            encoder_input: input to the encoder. Either 'bow' or 'embeddings'. 'bow' is a simple Bag-of-Words representation of the documents. 'embeddings' is the representation from a pre-trained embedding model (e.g. GPT, BERT, GloVe, etc.).
            encoder_hidden_layers: list with the size of the hidden layers for the encoder.
            encoder_non_linear_activation: non-linear activation function for the encoder.
            encoder_bias: whether to use bias in the encoder.
            decoder_type: type of decoder. Either 'mlp' or 'sage'. 'mlp' is an arbitrarily complex Multilayer Perceptron. 'sage' is a Sparse Additive Generative Model.
            decoder_hidden_layers: list with the size of the hidden layers for the decoder (only used with decoder_type='mlp').
            decoder_non_linear_activation: non-linear activation function for the decoder (only used with decoder_type='mlp').
            decoder_bias: whether to use bias in the decoder (only used with decoder_type='mlp').
            decoder_sparsity_regularization: regularization parameter for the decoder (only used with decoder_type='sage').
            num_epochs: number of epochs to train the model.
            batch_size: batch size for training.
            learning_rate: learning rate for training.
            dropout: dropout rate for training.
            log_every: number of epochs between each checkpoint.
            tightness: parameter to control the tightness of the encoder output with the document-topic prior.
            ckpt: checkpoint to load the model from.
            device: device to use for training.
            seed: random seed.

        References:
            - Eisenstein, J., Ahmed, A., & Xing, E. P. (2011). Sparse additive generative models of text. In Proceedings of the 28th international conference on machine learning (ICML-11) (pp. 1041-1048).
            - Nan, F., Ding, R., Nallapati, R., & Xiang, B. (2019). Topic modeling with wasserstein autoencoders. arXiv preprint arXiv:1907.12374.
        """

        self.n_topics = n_topics
        self.doc_topic_prior = doc_topic_prior
        self.update_prior = update_prior
        self.alpha = alpha
        self.prevalence_covariates_regularization = prevalence_covariates_regularization
        self.encoder_input = encoder_input
        self.encoder_hidden_layers = encoder_hidden_layers
        self.encoder_non_linear_activation = encoder_non_linear_activation
        self.encoder_bias = encoder_bias
        self.decoder_type = decoder_type
        self.decoder_hidden_layers = decoder_hidden_layers
        self.decoder_non_linear_activation = decoder_non_linear_activation
        self.decoder_bias = decoder_bias
        self.decoder_sparsity_regularization = decoder_sparsity_regularization
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.log_every = log_every
        self.tightness = tightness

        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            np.random.seed(seed)

        if device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        bow_size = train_data.M_bow.shape[1]
        self.bow_size = bow_size

        if train_data.prevalence is not None:
            prevalence_covariate_size = train_data.M_prevalence_covariates.shape[1]
        else:
            prevalence_covariate_size = 0

        if train_data.content is not None:
            content_covariate_size = train_data.M_content_covariates.shape[1]
        else:
            content_covariate_size = 0

        if train_data.labels is not None:
            labels_size = train_data.M_labels.shape[1]
        else:
            labels_size = 0

        if encoder_input == "bow":
            self.input_size = bow_size
        elif encoder_input == "embeddings":
            input_embeddings_size = train_data.M_embeddings.shape[1]
            self.input_size = input_embeddings_size

        self.content_covariate_size = content_covariate_size
        self.prevalence_covariate_size = prevalence_covariate_size
        self.labels_size = labels_size
        self.id2token = train_data.id2token

        encoder_dims = [self.input_size + prevalence_covariate_size + labels_size]
        encoder_dims.extend(encoder_hidden_layers)
        encoder_dims.extend([n_topics])

        decoder_dims = [n_topics + content_covariate_size]
        decoder_dims.extend(decoder_hidden_layers)
        decoder_dims.extend([bow_size])

        if decoder_type == "mlp":
            self.AutoEncoder = AutoEncoderMLP(
                encoder_dims=encoder_dims,
                encoder_non_linear_activation=encoder_non_linear_activation,
                encoder_bias=encoder_bias,
                decoder_dims=decoder_dims,
                decoder_non_linear_activation=decoder_non_linear_activation,
                decoder_bias=decoder_bias,
                dropout=dropout,
            ).to(self.device)
        elif decoder_type == "sage":
            self.AutoEncoder = AutoEncoderSAGE(
                encoder_dims=encoder_dims,
                encoder_non_linear_activation=encoder_non_linear_activation,
                encoder_bias=encoder_bias,
                dropout=dropout,
                bow_size=bow_size,
                content_covariate_size=content_covariate_size,
                log_word_frequencies=train_data.log_word_frequencies,
                l1_beta_reg=decoder_sparsity_regularization,
                l1_beta_c_reg=decoder_sparsity_regularization,
                l1_beta_ci_reg=decoder_sparsity_regularization,
            ).to(self.device)

        if doc_topic_prior == "dirichlet":
            self.prior = DirichletPrior(
                prevalence_covariate_size, n_topics, alpha, device=self.device
            )
        elif doc_topic_prior == "logistic_normal":
            self.prior = LogisticNormalPrior(
                prevalence_covariate_size,
                n_topics,
                prevalence_covariates_regularization,
                device=self.device,
            )

        if labels_size != 0:
            self.classifier = nn.Linear(n_topics, labels_size).to(device)

        self.train(
            train_data,
            batch_size,
            learning_rate,
            num_epochs,
            log_every,
            tightness,
            ckpt,
        )

    def train(
        self,
        train_data,
        batch_size,
        learning_rate,
        num_epochs,
        log_every,
        tightness,
        ckpt,
    ):
        """
        Train the model.
        """
        self.AutoEncoder.train()

        data_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=2
        )

        optimizer = torch.optim.Adam(self.AutoEncoder.parameters(), lr=learning_rate)

        if self.decoder_type == "sage":
            n_train = train_data.M_bow.shape[0]
            l1_beta = (
                0.5
                * np.ones([self.bow_size, self.n_topics], dtype=np.float32)
                / float(n_train)
            )
            if self.content_covariate_size != 0:
                l1_beta_c = (
                    0.5
                    * np.ones(
                        [self.bow_size, self.content_covariate_size], dtype=np.float32
                    )
                    / float(n_train)
                )
                l1_beta_ci = (
                    0.5
                    * np.ones(
                        [self.bow_size, self.n_topics * self.content_covariate_size],
                        dtype=np.float32,
                    )
                    / float(n_train)
                )
            else:
                l1_beta_c = None
                l1_beta_ci = None

        if ckpt:
            self.load_model(ckpt["net"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
        else:
            start_epoch = 0

        trainloss_lst = []
        for epoch in range(start_epoch, num_epochs):
            epochloss_lst = []
            for iter, data in enumerate(data_loader):
                optimizer.zero_grad()

                # Unpack data
                for key, value in data.items():
                    data[key] = value.to(self.device)
                bows = data.get("M_bow", None)
                bows = bows.reshape(bows.shape[0], -1)
                embeddings = data.get("M_embeddings", None)
                prevalence_covariates = data.get("M_prevalence_covariates", None)
                content_covariates = data.get("M_content_covariates", None)
                target_labels = data.get("M_labels", None)
                if self.encoder_input == "bow":
                    x_input = bows
                elif self.encoder_input == "embeddings":
                    x_input = embeddings

                # Get theta and compute reconstruction loss
                x_recon, theta_q = self.AutoEncoder(
                    x_input, prevalence_covariates, content_covariates, target_labels
                )
                rec_loss = compute_reconstruction_loss(x_input, x_recon)

                # Get prior on theta and compute regularization loss
                theta_prior = self.prior.sample(
                    N=x_input.shape[0], M_prevalence_covariates=prevalence_covariates
                ).to(self.device)
                mmd_loss = compute_mmd_loss(
                    theta_q, theta_prior, device=self.device, t=0.1
                )
                s = torch.sum(x_input) / len(x_input)
                lamb = (
                    5.0
                    * s
                    * torch.log(torch.tensor(1.0 * x_input.shape[-1]))
                    / torch.log(torch.tensor(2.0))
                )
                mmd_loss = mmd_loss * lamb

                # Add regularization to induce sparsity in the topic-word-covariate distributions
                if self.decoder_type == "sage":
                    sparsity_loss = self.AutoEncoder.sparsity_loss(
                        l1_beta, l1_beta_c, l1_beta_ci, self.device
                    )
                else:
                    sparsity_loss = 0

                # Predict labels and compute classification loss
                if target_labels is not None:
                    estimated_labels = self.classifier(theta_q)
                    classification_loss = torch.nn.CrossEntropyLoss()(
                        estimated_labels, target_labels
                    )
                else:
                    classification_loss = 0

                # Total loss
                loss = (
                    rec_loss
                    + mmd_loss * tightness
                    + classification_loss
                    + sparsity_loss
                )

                loss.backward()
                optimizer.step()

                trainloss_lst.append(loss.item() / len(x_input))
                epochloss_lst.append(loss.item() / len(x_input))
                if (iter + 1) % 10 == 0:
                    print(
                        f"Epoch {(epoch+1):>3d}\tIter {(iter+1):>4d}\tLoss:{loss.item()/len(x_input):<.7f}\tRec Loss:{rec_loss.item()/len(x_input):<.7f}\tMMD:{mmd_loss.item()/len(x_input):<.7f}\tSparsity_Loss:{sparsity_loss/len(x_input):<.7f}\tPred_Loss:{classification_loss/len(x_input):<.7f}"
                    )

            if (epoch + 1) % log_every == 0:
                save_name = f'../ckpt/WTM_tp{self.n_topics}_{self.doc_topic_prior}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}_{epoch+1}.ckpt'
                checkpoint = {
                    "net": self.AutoEncoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "param": {
                        "input_dim": self.input_size,
                        "n_topics": self.n_topics,
                        "doc_topic_prior": self.doc_topic_prior,
                        "dropout": self.dropout,
                    },
                }
                torch.save(checkpoint, save_name)
                print(
                    f"Epoch {(epoch+1):>3d}\tLoss:{sum(epochloss_lst)/len(epochloss_lst):<.7f}"
                )
                print(
                    "\n".join([str(lst) for lst in self.get_topic_word_distribution()])
                )

            if self.update_prior:
                posterior_theta = self.get_doc_topic_distribution(train_data)
                self.prior.update_parameters(
                    posterior_theta, train_data.M_prevalence_covariates
                )

            if self.decoder_type == "sage":
                (
                    l1_beta,
                    l1_beta_c,
                    l1_beta_ci,
                ) = self.AutoEncoder.update_jeffreys_priors(n_train)

    def get_doc_topic_distribution(self, dataset):
        """
        Get the topic distribution of each document in the corpus.
        """
        with torch.no_grad():
            data_loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
            )
            final_thetas = []
            for iter, data in enumerate(data_loader):
                for key, value in data.items():
                    data[key] = value.to(self.device)
                bows = data.get("M_bow", None)
                bows = bows.reshape(bows.shape[0], -1)
                embeddings = data.get("M_embeddings", None)
                prevalence_covariates = data.get("M_prevalence_covariates", None)
                content_covariates = data.get("M_content_covariates", None)
                target_labels = data.get("M_labels", None)
                if self.encoder_input == "bow":
                    x_input = bows
                elif self.encoder_input == "embeddings":
                    x_input = embeddings
                x_recon, thetas = self.AutoEncoder(
                    x_input, prevalence_covariates, content_covariates, target_labels
                )
                thetas = thetas.cpu().numpy()
                final_thetas.append(thetas)

        return np.concatenate(final_thetas, axis=0)

    def get_topic_word_distribution(self, formula=None, topK=8):
        """
        Get the word distribution of each topic, potentially influenced by content covariates.
        """
        self.AutoEncoder.eval()
        with torch.no_grad():
            if self.decoder_type == "mlp":
                # topic_words = []
                idxes = torch.eye(self.n_topics + self.content_covariate_size).to(
                    self.device
                )
                word_dist = self.AutoEncoder.decode(idxes)
                word_dist = F.softmax(word_dist, dim=1)
                # vals, _ = torch.topk(word_dist, topK, dim=1)
                # vals = vals.cpu().tolist()
                # indices = indices.cpu().tolist()
                # for topic_id in range(self.n_topics + self.content_covariate_size):
                #     topic_words.append(
                #         [self.id2token[idx] for idx in indices[topic_id]]
                #     )
            elif self.decoder_type == "sage":
                # topic_words = []
                word_dist = F.softmax(self.AutoEncoder.beta.weight.T, dim=1)
                # vals, _ = torch.topk(word_dist, topK, dim=1)
                # vals = vals.cpu().tolist()
                # indices = indices.cpu().tolist()
                # for i in range(self.n_topics):
                #     topic_words.append([self.id2token[idx] for idx in indices[i]])
        return word_dist

    def get_topic_correlations(self):
        """
        Plot correlations between topics for a logistic normal prior.
        """
        pass

    def get_ldavis_data_format(self):
        """
        Returns a data format that can be used in input to pyldavis to interpret the topics.
        """
        pass

    def load_model(self, model):
        """
        Helper function to load a GTM model.
        """
        self.AutoEncoder.load_state_dict(model)
