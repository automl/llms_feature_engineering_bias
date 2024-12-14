from dataclasses import dataclass
from pandas import DataFrame, Series


@dataclass
class Dataset:
    """
    A simple class to represent a dataset.
    """
    X: DataFrame
    y: Series
    dataset_name: str
    dataset_description: str
    feature_names: list[str]
    target_name: str

    def __init__(self,
                 X: DataFrame,
                 y: Series,
                 name: str,
                 dataset_description: str) -> None:
        """
        Initialize the dataset object. This object holds all relevant (meta-)
        data representing the dataset.

        Parameters
        ----------
        X : DataFrame
            The input features of the dataset.
        y: Series
            The target values of the dataset.
        name : str
            The name of the dataset.
        dataset_description : str
            A short description of the dataset.
        """
        self.X = X
        self.y = y
        self.dataset_name = name
        self.dataset_description = dataset_description