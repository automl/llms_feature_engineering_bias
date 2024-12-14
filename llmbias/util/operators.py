import pandas as pd
import numpy as np


def frequency_encode(
        feature_one: pd.Series,
        mapping: pd.Series | None
        ) -> tuple[pd.Series, pd.Series]:
    if mapping is None:
        mapping = feature_one.value_counts()
    return feature_one.map(mapping), mapping


def absolute(feature_one: pd.Series) -> pd.Series:
    return feature_one.abs()


def log(feature_one: pd.Series) -> pd.Series:
    return feature_one.apply(lambda x: np.log(x) if x > 0 else 0)


def squareroot(feature_one: pd.Series) -> pd.Series:
    return feature_one.apply(lambda x: np.sqrt(x) if x > 0 else 0)


def sigmoid(feature_one: pd.Series) -> pd.Series:
    return feature_one.apply(lambda x: 1 / (1 + np.exp(-x)))


def round(feature_one: pd.Series) -> pd.Series:
    return feature_one.round()


def residual(feature_one: pd.Series) -> pd.Series:
    return feature_one.apply(lambda x: x - np.floor(x))


def min(feature_one: pd.Series, feature_two: pd.Series) -> pd.Series:
    return np.minimum(feature_one, feature_two)


def max(feature_one: pd.Series, feature_two: pd.Series) -> pd.Series:
    return np.maximum(feature_one, feature_two)


def add(feature_one: pd.Series, feature_two: pd.Series) -> pd.Series:
    return feature_one + feature_two


def subtract(feature_one: pd.Series, feature_two: pd.Series) -> pd.Series:
    return feature_one - feature_two


def multiply(feature_one: pd.Series, feature_two: pd.Series) -> pd.Series:
    return feature_one * feature_two


def divide(feature_one: pd.Series, feature_two: pd.Series) -> pd.Series:
    return feature_one / feature_two.replace(0, np.nan)


def combine(
        feature_one: pd.Series,
        feature_two: pd.Series,
        mapping: dict[str, int] | None
        ) -> tuple[pd.Series, pd.Series]:
    combined_feature = feature_one.astype(str) + "_" + feature_two.astype(str)
    if mapping is None:
        codes, uniques = pd.factorize(combined_feature)
        mapping = dict(zip(uniques, codes))
    return combined_feature.map(mapping).astype("float64"), mapping


def combine_then_frequency_encode(
        feature_one: pd.Series,
        feature_two: pd.Series,
        mapping: dict[str, int] | None
        ) -> tuple[pd.Series, pd.Series]:
    combined_feature = feature_one.astype(str) + "_" + feature_two.astype(str)
    if mapping is None:
        codes, uniques = pd.factorize(combined_feature)
        combine_mapping = dict(zip(uniques, codes))
        frequency_mapping = combined_feature.map(combine_mapping).value_counts().to_dict()
        # reverse combine mapping to map combine keys to frequency counts
        mapping = {key: frequency_mapping[combine_mapping[key]] for key in combine_mapping}
    return combined_feature.map(mapping), mapping


def group_by_then_min(
        feature_one: pd.Series,
        feature_two: pd.Series,
        groupby: pd.Series | None
        ) -> tuple[pd.Series, pd.Series]:
    if groupby is None:
        groupby = feature_one.groupby(feature_two).min()
    return feature_one.apply(lambda x: groupby.loc[x] if x in groupby else np.nan), groupby


def group_by_then_max(
        feature_one: pd.Series,
        feature_two: pd.Series,
        groupby: pd.Series | None
        ) -> tuple[pd.Series, pd.Series]:
    if groupby is None:
        groupby = feature_one.groupby(feature_two).max()
    return feature_one.apply(lambda x: groupby.loc[x] if x in groupby else np.nan), groupby


def group_by_then_mean(
        feature_one: pd.Series,
        feature_two: pd.Series,
        groupby: pd.Series | None
        ) -> tuple[pd.Series, pd.Series]:
    if groupby is None:
        groupby = feature_one.groupby(feature_two).mean()
    return feature_one.apply(lambda x: groupby.loc[x] if x in groupby else np.nan), groupby


def group_by_then_median(
        feature_one: pd.Series,
        feature_two: pd.Series,
        groupby: pd.Series | None
        ) -> tuple[pd.Series, pd.Series]:
    if groupby is None:
        groupby = feature_one.groupby(feature_two).median()
    return feature_one.apply(lambda x: groupby.loc[x] if x in groupby else np.nan), groupby


def group_by_then_std(
        feature_one: pd.Series,
        feature_two: pd.Series,
        groupby: pd.Series | None
        ) -> tuple[pd.Series, pd.Series]:
    if groupby is None:
        groupby = feature_one.groupby(feature_two).std()
    return feature_one.apply(lambda x: groupby.loc[x] if x in groupby else np.nan), groupby


def group_by_then_rank(
        feature_one: pd.Series,
        feature_two: pd.Series,
        groupby: pd.Series | None
        ) -> tuple[pd.Series, pd.Series]:
    if groupby is None:
        groupby = feature_one.groupby(feature_two).rank(pct=True)
    return feature_one.apply(lambda x: groupby.loc[x] if x in groupby else np.nan), groupby
