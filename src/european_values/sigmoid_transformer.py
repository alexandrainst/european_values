"""Custom transformer used to transform log-likelihoods into scores."""

import logging
import typing as t
from functools import partial

import scipy.optimize as opt
from numpy import mean, quantile, zeros
from scipy.special import expit as sigmoid
from scipy.special import logit as inverse_sigmoid
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

if t.TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


class SigmoidTransformer(TransformerMixin):
    """Transformer to apply a sigmoid function to log-likelihoods."""

    def fit(self, X: "ndarray") -> "SigmoidTransformer":
        """Fit the transformer to the data.

        Args:
            X:
                The input array of log-likelihoods.

        Returns:
            The fitted transformer.
        """
        # We choose the alpha parameter to fit the range of the log-likelihoods. An
        # alpha of 0.1 has an effective range of 100, and scales inversely with the
        # range of the data: with alpha being 0.05 we get an effective range of 200,
        lower, upper = quantile(X, q=[0.01, 0.99])
        self.alpha_ = 0.1 / ((upper - lower) / 100)

        # Optimise the center of the sigmoid function to fit the target value
        result: opt.OptimizeResult = opt.minimize(
            fun=partial(self._loss, array=X, target=0.9, alpha=self.alpha_),
            x0=zeros(shape=(1,)),
        )
        self.center_ = result.x[0]
        logger.info(
            f"Fitted sigmoid transformer with alpha={self.alpha_:.2f} and "
            f"center={self.center_:.2f}."
        )
        return self

    def transform(self, X: "ndarray") -> "ndarray":
        """Transform the input data using the fitted sigmoid function.

        Args:
            X:
                The input array of log-likelihoods.

        Returns:
            The transformed values between 0 and 1.
        """
        check_is_fitted(estimator=self, attributes=["alpha_", "center_"])
        return sigmoid(self.alpha_ * (X - self.center_))

    @staticmethod
    def _loss(
        center: "ndarray", array: "ndarray", target: float, alpha: float
    ) -> float:
        """Calculate the loss for the sigmoid transformation.

        The loss aims to get the sigmoid values of the array as close to a given target
        value as possible.

        Args:
            center:
                The center of the sigmoid curve.
            array:
                The input array of log-likelihoods.
            target:
                The target value for the sigmoid transformation.
            alpha:
                The steepness of the sigmoid curve.

        Returns:
            The l2 loss between the transformed values and the target sigmoid values.
        """
        target = inverse_sigmoid(target)
        errors = (alpha * (array - center) - target) ** 2
        l2_loss = mean(errors).item()
        return l2_loss
