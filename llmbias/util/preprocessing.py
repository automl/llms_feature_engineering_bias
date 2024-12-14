import pandas as pd
import numpy as np
import logging
from typing import Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder

pd.options.mode.chained_assignment = None

logger = logging.getLogger(__name__)


def _encode_categorical(X: pd.Series,
                        mapping: dict[str, float] | None = None,
                        ) -> tuple[pd.Series, dict[str, float]]:
    if mapping is None:
        mapping = {unique: i for i, unique in enumerate(X.unique())}
    return X.apply(lambda x: mapping[x] if x in mapping else -1), mapping


def preprocess(X_train: pd.DataFrame,
               X_test: pd.DataFrame,
               y_train: Optional[pd.Series] = None,
               y_test: Optional[pd.Series] = None
               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    encodes and scales

    Returns:
    --------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    X_train, X_test, y_train, y_test
    """

    # encode
    l_enc = LabelEncoder()

    # TODO: logging.info(f"Missing {X.isna().sum()}")
    logger.debug("Encoding categorical values")
    for col in X_train.columns:
        if X_train[col].dtype == "object" or X_train[col].dtype == "category":
            encoded_col_train, mapping = _encode_categorical(X_train[col])

            X_train[col] = encoded_col_train

            # encode test data with labels from test data, handle unseen lables
            encoded_col_test, _ = _encode_categorical(X_test[col], mapping)
            X_test[col] = encoded_col_test
        else:
            # assume numerical col
            if X_train[col].isna().any():
                # fill missing with mean
                filler = X_train[col].mean()
                X_train[col] = X_train[col].fillna(filler)

                # fill missing in test with mean from train
                X_train[col] = X_test[col].fillna(filler)

    if y_train is not None and y_test is not None:
        # relace possible missing target values
        y_train = y_train.replace([np.inf, -np.inf, np.nan], -1)
        y_test = y_test.replace([np.inf, -np.inf, np.nan], -1)

        l_enc.fit(y_train)

        y_train = l_enc.transform(y_train)

        # transform test target, handle unseen labels
        tmp = []
        for item in y_test:
            if item in l_enc.classes_:
                tmp.append(l_enc.transform([item])[0])
            else:
                tmp.append(-1)
        y_test = tmp
    else:
        y_train, y_test = None, None

    # scale
    scaler = StandardScaler(with_mean=False)
    logger.debug("Scaling on training data")
    scaler.fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    return (X_train, X_test, y_train, y_test)