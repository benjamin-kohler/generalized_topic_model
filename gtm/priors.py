#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# Third Party Library
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import linear_model
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multivariate_normal import MultivariateNormal
from utils import compute_dirichlet_likelihood


class Prior:
    """
    Base template class for doc-topic priors.
    """

    def __init__(self):
        pass

    def update_parameters(self):
        """
        M-step after each epoch.
        """
        pass

    def sample(self):
        """
        Sample from the prior.
        """
        pass

    def simulate(self):
        """
        Simulate data to test the prior's updating rule.
        """
        pass


class LogisticNormalPrior(Prior):
    """
    Logistic Normal prior

    We draw from a multivariate gaussian and map it to the simplex.
    Does not induce sparsity, but may account for topic correlations.

    References:
        - Roberts, M. E., Stewart, B. M., & Airoldi, E. M. (2016). A model of text for experimentation in the social sciences. Journal of the American Statistical Association, 111(515), 988-1003.
    """

    def __init__(
        self,
        prevalence_covariate_size,
        n_topics,
        prevalence_covariates_regularization,
        device,
    ):
        self.prevalence_covariates_size = prevalence_covariate_size
        self.n_topics = n_topics
        self.prevalence_covariates_regularization = prevalence_covariates_regularization
        self.device = device
        if prevalence_covariate_size != 0:
            self.lambda_ = torch.zeros(prevalence_covariate_size, n_topics).to(
                self.device
            )
            self.sigma = torch.diag(torch.Tensor([1.0] * self.n_topics)).to(self.device)

    def update_parameters(self, posterior_theta, M_prevalence_covariates):
        """
        M-step after each epoch.
        """
        M_prevalence_covariates = M_prevalence_covariates
        posterior_mu = np.log(posterior_theta + 1e-6)

        reg = linear_model.Ridge(
            alpha=self.prevalence_covariates_regularization, fit_intercept=False
        )
        lambda_ = reg.fit(M_prevalence_covariates, posterior_mu).coef_
        self.lambda_ = torch.from_numpy(lambda_.T).to(self.device)

        posterior_mu = torch.from_numpy(posterior_mu).to(self.device)
        M_prevalence_covariates = torch.from_numpy(M_prevalence_covariates).to(
            self.device
        )
        difference_in_means = posterior_mu - torch.matmul(
            M_prevalence_covariates, self.lambda_
        )
        self.sigma = (
            torch.matmul(difference_in_means.T, difference_in_means)
            / posterior_mu.shape[0]
        )

        self.lambda_ = self.lambda_ - self.lambda_[:, 0][:, None]

    def sample(self, N, M_prevalence_covariates):
        """
        Sample from the prior.
        """
        if self.prevalence_covariates_size == 0:
            z_true = np.random.randn(N, self.n_topics)
            z_true = torch.softmax(torch.from_numpy(z_true), dim=1).float()
        else:
            means = torch.matmul(M_prevalence_covariates, self.lambda_)
            for i in range(means.shape[0]):
                if i == 0:
                    m = MultivariateNormal(means[i], self.sigma)
                    z_true = m.sample().unsqueeze(0)
                else:
                    m = MultivariateNormal(means[i], self.sigma)
                    z_temp = m.sample()
                    z_true = torch.cat([z_true, z_temp.unsqueeze(0)], 0)
            z_true = torch.softmax(z_true, dim=1).float()
        return z_true

    def simulate(self, M_prevalence_covariates, lambda_, sigma):
        """
        Simulate data to test the prior's updating rule.
        """
        means = torch.matmul(M_prevalence_covariates, lambda_)
        for i in range(means.shape[0]):
            if i == 0:
                m = MultivariateNormal(means[i], sigma)
                z_sim = m.sample().unsqueeze(0)
            else:
                m = MultivariateNormal(means[i], sigma)
                z_temp = m.sample()
                z_sim = torch.cat([z_sim, z_temp.unsqueeze(0)], 0)
        z_sim = torch.softmax(z_sim, dim=1).float()
        return z_sim


class LinearModel(torch.nn.Module):
    """
    Simple linear model for the priors.
    """

    def __init__(self, prevalence_covariates_size, n_topics):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(prevalence_covariates_size, n_topics)

    def forward(self, M_prevalence_covariates):
        linear_preds = self.linear(M_prevalence_covariates)
        return linear_preds


class DirichletPrior(Prior):
    """
    Dirichlet prior

    Induces sparsity, but does not account for topic correlations.

    References:
        - Mimno, D. M., & McCallum, A. (2008, July). Topic models conditioned on arbitrary features with Dirichlet-multinomial regression. In UAI (Vol. 24, pp. 411-418).
        - Maier, M. (2014). DirichletReg: Dirichlet regression for compositional data in R.
    """

    def __init__(self, prevalence_covariates_size, n_topics, alpha, device):
        self.prevalence_covariates_size = prevalence_covariates_size
        self.n_topics = n_topics
        self.alpha = alpha
        self.lambda_ = None
        self.device = device
        if prevalence_covariates_size != 0:
            self.linear_model = LinearModel(prevalence_covariates_size, n_topics).to(
                self.device
            )

    def update_parameters(
        self, posterior_theta, M_prevalence_covariates, num_epochs=1000
    ):
        """
        M-step after each epoch.
        """

        # Simple Conditional Means Estimation
        # (fast educated guess)
        y = np.log(posterior_theta + 1e-6)
        reg = linear_model.Ridge(alpha=0, fit_intercept=False)
        self.lambda_ = reg.fit(M_prevalence_covariates, y).coef_.T
        self.lambda_ = self.lambda_ - self.lambda_[:, 0][:, None]

        # Maximum Likelihood Estimation (MLE)
        # (pretty slow)
        if num_epochs != 0:
            self.lambda_ = torch.from_numpy(self.lambda_).float().to(self.device)
            posterior_theta = torch.from_numpy(posterior_theta).float().to(self.device)
            M_prevalence_covariates = (
                torch.from_numpy(M_prevalence_covariates).float().to(self.device)
            )
            with torch.no_grad():
                self.linear_model.linear.weight.copy_(self.lambda_.T)
            optimizer = torch.optim.Adam(self.linear_model.parameters(), lr=1e-3)
            for i in range(num_epochs):
                optimizer.zero_grad()
                linear_preds = self.linear_model(M_prevalence_covariates)
                alphas = torch.exp(linear_preds)
                loss = -compute_dirichlet_likelihood(alphas, posterior_theta)
                loss.backward()
                optimizer.step()

            self.lambda_ = self.linear_model.linear.weight.detach().T
            self.lambda_ = self.lambda_ - self.lambda_[:, 0][:, None]
            self.lambda_ = self.lambda_.cpu().numpy()

    def sample(self, N, M_prevalence_covariates):
        """
        Sample from the prior.
        """
        if self.prevalence_covariates_size == 0:
            z_true = np.random.dirichlet(np.ones(self.n_topics) * self.alpha, size=N)
            z_true = torch.from_numpy(z_true).float()
        else:
            with torch.no_grad():
                linear_preds = self.linear_model(M_prevalence_covariates)
                # alphas = F.softmax(linear_preds, dim=1)
                alphas = torch.exp(linear_preds)
                for i in range(alphas.shape[0]):
                    if i == 0:
                        d = Dirichlet(alphas[i])
                        z_true = d.sample().unsqueeze(0)
                    else:
                        d = Dirichlet(alphas[i])
                        z_temp = d.sample()
                        z_true = torch.cat([z_true, z_temp.unsqueeze(0)], 0)
        return z_true

    def simulate(self, M_prevalence_covariates, lambda_):
        """
        Simulate data to test the prior's updating rule.
        """
        alphas = torch.exp(torch.matmul(M_prevalence_covariates, lambda_))
        for i in range(alphas.shape[0]):
            if i == 0:
                d = Dirichlet(alphas[i])
                z_sim = d.sample().unsqueeze(0)
            else:
                d = Dirichlet(alphas[i])
                z_temp = d.sample()
                z_sim = torch.cat([z_sim, z_temp.unsqueeze(0)], 0)
        return z_sim
