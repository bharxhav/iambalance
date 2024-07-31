"""Oversampler module for iambalance package."""

from typing import List
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from .purity import Purity


class Oversampler():
    """A class for performing multiple oversampling techniques.

    This class combines various oversampling methods:
        - random - Random oversampling
        - mutation - Random oversampling with mutation to neighboring columns.
        - smote - Synthetic Minority Over-sampling Technique.
        - adasyn - Adaptive Synthetic Sampling Approach for Imbalanced Learning.

    It allows for customized oversampling strategies and includes purity measures.

    Attributes:
        methods (List[str]): List of oversampling methods to use.
        oversample_contribution (List[float]): Proportion of oversampling for each method.
        oversample_count (int): Number of rows to oversample per iteration.
        iterations (int): Number of iterations for oversampling.
        purity_trend (List[Tuple]): List to store purity trends.

    """

    def __init__(
        self,
        methods: List[str] = None,
        oversample_contribution: List[float] = None,
        oversample_count: int = 100,
        iterations: int = 1,
    ):
        if methods is None:
            methods = ["random", "mutation", "smote", "adasyn"]
        if oversample_contribution is None:
            oversample_contribution = list(1/len(methods) for _ in methods)
        if not isinstance(oversample_contribution, list) or not all(isinstance(i, float) for i in oversample_contribution):
            raise ValueError("oversample_size must be a list of floats.")
        if len(oversample_contribution) != len(methods):
            raise ValueError(
                "Length of oversample_contributions must match length of methods.")

        self.methods = methods
        self.oversample_contribution = oversample_contribution
        self.oversample_count = oversample_count
        self.iterations = iterations

        # Purity object
        self.purity = Purity()

    def fit_resample(
        self, x: pd.DataFrame, y: str
    ) -> pd.DataFrame:
        """Fit the oversampler and resample the data.

        Args:
            X: Input features.
            y: Name of the target variable.

        Returns:
            Tuple of oversampled features and target.
        """
        pass

    def _fit_resample_random(
        self, x: pd.DataFrame, y: str
    ) -> pd.DataFrame:
        """Perform random oversampling.

        Args:
            x: Input features.
            y: Name of the target variable.

        Returns:
            Random oversampled features.
        """
        minority_class = x[x[y].value_counts().idxmin()]
        oversampled = minority_class.sample(
            n=self.oversample_count, replace=True)
        return pd.concat([x, oversampled], ignore_index=True)

    def _fit_resample_mutation(
        self, x: pd.DataFrame, y: str
    ) -> pd.DataFrame:
        """Perform mutation oversampling.

        Args:
            x: Input features.
            y: Name of the target variable.

        Returns:
            Mutation oversampled features.
        """
        minority_class = x[x[y].value_counts().idxmin()]
        oversampled = minority_class.sample(
            n=self.oversample_count, replace=True)

        # Perform mutation on the oversampled data
        for column in oversampled.columns:
            if column != y and oversampled[column].dtype in ['int64', 'float64']:
                mutation = oversampled[column].mean() * 0.1  # 10% mutation
                oversampled[column] += np.random.normal(
                    0, mutation, oversampled.shape[0])

        return pd.concat([x, oversampled], ignore_index=True)

    def _fit_resample_smote(
        self, x: pd.DataFrame, y: str
    ) -> pd.DataFrame:
        """Perform SMOTE oversampling.

        Args:
            x: Input features.
            y: Name of the target variable.

        Returns:
            SMOTE oversampled features.
        """
        smote = SMOTE()
        target = x[y]
        features = x.drop(columns=[y])
        features_resampled, target_resampled = smote.fit_resample(
            features, target)
        x_resampled = features_resampled.copy()
        x_resampled[y] = target_resampled
        return x_resampled

    def _fit_resample_adasyn(
        self, x: pd.DataFrame, y: str
    ) -> pd.DataFrame:
        """Perform ADASYN oversampling.

        Args:
            x: Input features.
            y: Name of the target variable.

        Returns:
            ADASYN oversampled features.
        """
        adasyn = ADASYN()
        target = x[y]
        features = x.drop(columns=[y])
        features_resampled, target_resampled = adasyn.fit_resample(
            features, target)
        x_resampled = features_resampled.copy()
        x_resampled[y] = target_resampled
        return x_resampled

    def get_oversampled_fingerprint(self) -> List[int]:
        """Get indices of oversampled rows.

        Returns:
            Dataframe with:
                - index: Index of the oversampled row.
                - method: Method used for oversampling.
                - iteration: Iteration number.
        """
        pass

    def _generate_purity(self):
        pass

    def get_purity_object(self) -> Purity:
        """Return the purity object.

        Returns:
            Purity object.
        """
        return self.purity
