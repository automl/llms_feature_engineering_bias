import pandas as pd
import numpy as np
import logging
from typing import Any, Union
import traceback

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from tabpfn.scripts.tabular_metrics import auc_metric

from llmbias.util.preprocessing import preprocess

logger = logging.getLogger(__name__)


def run_stratified_cv(X_train: pd.DataFrame,
                      y_train: pd.Series,
                      model: Any,
                      n_splits: int = 10,
                      shuffle: bool = True,
                      random_state: int = 42,
                      ) -> list[float]:

    cv = StratifiedKFold(n_splits=n_splits,
                         shuffle=shuffle,
                         random_state=random_state)
    scores = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X_train,
                                                       y_train)):
        try:
            logger.debug(f"Running {i}th fold for \
{n_splits}-fold-stratified cv")
            X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]

            # preprocessing for current split
            X_train_fold, X_test_fold, y_train_fold, y_test_fold = \
                preprocess(X_train_fold,
                           X_test_fold,
                           y_train_fold,
                           y_test_fold)

            model.fit(X_train_fold, y_train_fold)

            y_fold_predict = model.predict_proba(X_test_fold)

            score = auc_metric(y_test_fold, y_fold_predict)

            # score is type torch.tensor
            scores.append(score.item())
        except Exception as e:
            logger.error(f"Failed to run cross validation because an error \
occured: {str(e)}, {traceback.format_exc()}")
            scores.append(0.0)
            break

    return scores
