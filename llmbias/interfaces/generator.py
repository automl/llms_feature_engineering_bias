from abc import ABC, abstractmethod
from llmbias.util.dataset import Dataset


class AbstractGenerator(ABC):
    """
    An abstract interface class providing functionalities for generating new
    features using a given method.
    """

    def __init__(self,
                 train_ds: Dataset,
                 test_ds: Dataset) -> None:
        self._train_ds = train_ds
        self._test_ds = test_ds

    @property
    def train_ds(self) -> Dataset:
        return self._train_ds

    @property
    def test_ds(self) -> Dataset:
        return self._test_ds

    @abstractmethod
    def ask(self,
            n_features: int,
            return_operators: bool,
            n_jobs: int,
            seed: int):
        """
        Ask the generator to generate a new sample.
        """
        raise NotImplementedError

    @abstractmethod
    def tell(self):
        """
        Update the states of the generator based on the feedback.
        """
        raise NotImplementedError

    @abstractmethod
    def _generate(self):
        """
        Generate a new features by interfering the generator.
        """
        raise NotImplementedError
