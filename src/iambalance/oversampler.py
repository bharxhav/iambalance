"""Oversampler module for imbalance package."""

from typing import Any, List
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from .purity import Purity


class Oversampler:
    """A class for performing multiple oversampling techniques.

    This class combines various oversampling methods:
        - random: Random oversampling
        - mutation: Random oversampling with mutation to neighboring columns
        - smote: Synthetic Minority Over-sampling Technique
        - adasyn: Adaptive Synthetic Sampling Approach for Imbalanced Learning

    It allows for customized oversampling strategies and includes purity measures.

    Attributes:
        methods (List[str]): List of oversampling methods to use.
        method_contributions (List[float]): Proportion of oversampling for each method.
        iterations (int): Number of iterations for oversampling.

    Raises:
        ValueError: If method_contributions is not a list of floats or if its length
                    doesn't match the length of methods.
    """

    def __init__(
        self,
        methods: List[str] = None,
        method_contributions: List[float] = None,
        iterations: int = 1,
        purity_range: int = 0,
    ):
        if not isinstance(method_contributions, list) or not all(isinstance(i, float) for i in method_contributions):
            raise ValueError("method_contributions must be a list of floats.")
        if len(method_contributions) != len(methods):
            raise ValueError(
                "Length of method_contributions must match length of methods.")
        if purity_range not in [0, 1, 2]:
            raise ValueError("purity_range must be 0, 1, or 2.")

        self.methods = methods or ["random", "mutation", "smote", "adasyn"]
        self.method_contributions = method_contributions or [
            1/len(self.methods) for _ in self.methods]
        self.iterations = iterations

        # Purity object
        self.purity = Purity(purity_range)

        self.fingerprint_df = pd.DataFrame(
            columns=["index", "method", "iteration"])

    def fit_resample(self, x: pd.DataFrame, y: str) -> pd.DataFrame:
        """Fit the oversampler and resample the data.

        This method applies the specified oversampling techniques to the input data.

        Args:
            x (pd.DataFrame): Input features.
            y (str): Name of the target variable.

        Returns:
            pd.DataFrame: Dataframe with oversampled features.
        """
        pass

    def _fit_resample_random(
        self, x: pd.DataFrame, y: str, nrows: int, target_class: Any
    ) -> pd.DataFrame:
        """Perform random oversampling.

        This method randomly duplicates samples from the minority class.

        Args:
            x (pd.DataFrame): Input features.
            y (str): Name of the target variable.
            nrows (int): Number of new rows to be generated.
            target_class (Any): The class that must be generated.

        Returns:
            pd.DataFrame: Newly generated rows using random oversampling.
        """
        minority_class = x[x[y] == target_class]
        return minority_class.sample(n=nrows, replace=True)

    def _fit_resample_smote(
        self, x: pd.DataFrame, y: str, nrows: int, target_class: Any
    ) -> pd.DataFrame:
        """Perform SMOTE oversampling.

        This method applies the Synthetic Minority Over-sampling Technique (SMOTE)
        to generate new samples for the minority class.

        Args:
            x (pd.DataFrame): Input features.
            y (str): Name of the target variable.
            nrows (int): Number of new rows to be generated.
            target_class (Any): The class that must be generated.

        Returns:
            pd.DataFrame: Newly generated rows using SMOTE oversampling.
        """
        smote = SMOTE(sampling_strategy={target_class: x.shape[0] + nrows})
        target = x[y]
        features = x.drop(columns=[y])
        features_resampled, target_resampled = smote.fit_resample(
            features, target)
        x_resampled = features_resampled.copy()
        x_resampled[y] = target_resampled

        # Return only the newly generated rows
        return x_resampled.iloc[-nrows:]

    def _fit_resample_adasyn(
        self, x: pd.DataFrame, y: str, nrows: int, target_class: Any
    ) -> pd.DataFrame:
        """Perform ADASYN oversampling.

        This method applies the Adaptive Synthetic (ADASYN) sampling approach
        to generate new samples for the minority class.

        Args:
            x (pd.DataFrame): Input features.
            y (str): Name of the target variable.
            nrows (int): Number of new rows to be generated.
            target_class (Any): The class that must be generated.

        Returns:
            pd.DataFrame: Newly generated rows using ADASYN oversampling.
        """
        adasyn = ADASYN(sampling_strategy={target_class: x.shape[0] + nrows})
        target = x[y]
        features = x.drop(columns=[y])
        features_resampled, target_resampled = adasyn.fit_resample(
            features, target)
        x_resampled = features_resampled.copy()
        x_resampled[y] = target_resampled

        # Return only the newly generated rows
        return x_resampled.iloc[-nrows:]

    def get_oversampled_fingerprint(self) -> pd.DataFrame:
        """Get indices of oversampled rows.

        This method returns information about the oversampled rows, including
        their indices, the method used for oversampling, and the iteration number.

        Returns:
            pd.DataFrame: Dataframe with columns:
                - index: Index of the oversampled row.
                - method: Method used for oversampling.
                - iteration: Iteration number.
        """
        pass

    def _generate_purity(self):
        """Generate purity measures for the oversampled data.

        This method calculates and updates the purity measures for the
        oversampled dataset.
        """
        pass

    def get_purity_object(self) -> Purity:
        """Return the purity object.

        This method provides access to the Purity object, which contains
        measures of data quality for the oversampled dataset.

        Returns:
            Purity: Purity object containing data quality measures.
        """
        return self.purity
