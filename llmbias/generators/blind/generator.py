import logging
import re
import pandas as pd
import lightgbm as lgbm
from statistics import mean

from remoteinference.interfaces.llm import LLMInterface
from typing import Any

from llmbias.generators.blind.core import build_assistant_prompt, \
    build_system_prompt, build_user_prompt, build_feature_from_string
from llmbias.interfaces.generator import AbstractGenerator
from llmbias.util.dataset import Dataset
from llmbias.util.cv import run_stratified_cv


logger = logging.getLogger(__name__)


class BlindGenerator(AbstractGenerator):
    """
    A simple class to generate features from a language model given a dataset
    input. The generator will have as options a list of possible applicable
    operators and will provide us with a blueprint on which operators to apply
    to which features. We will then manually create these operators in the
    background. Compared to the blueprint generator this generator does not
    know which types of operators he can select and the operators will only
    be numbered.
    """

    model: LLMInterface

    def __init__(self,
                 train_ds: Dataset,
                 test_ds: Dataset,
                 model: LLMInterface,
                 max_retries: int = 5,
                 context_size: int = 8192):
        super().__init__(train_ds, test_ds)
        # TODO: Implement the constructor
        self.model = model
        self.context_size = context_size
        self.max_retries = max_retries

    def _fit(self,
             temperature: float,
             max_tokens: int,
             feedback: dict[str, str],
             seed: int,
             ):
        """
        Fit the LLM on the currently given training dataset and prompt a new
        feature. Note that this method should only be fitted on the training
        data. The transformation for the test dataset at a later point has to
        be done seperately.

        Returns
            A string containing the instructions on how to generate the new
            feature.
        """
        if feedback:
            assistant_prompt = build_assistant_prompt(feedback)
            logger.debug(f"Assistant prompt: {assistant_prompt}")
        else:
            logger.debug("No feedback provided, not using assistant prompt")
            assistant_prompt = None

        system_prompt = build_system_prompt()
        logger.debug(f"System prompt: {system_prompt}")

        user_prompt = build_user_prompt(self.train_ds,
                                        feedback,
                                        seed)

        logger.debug(f"User prompt: {user_prompt}")

        if assistant_prompt:
            messages = [system_prompt, user_prompt, assistant_prompt]
        else:
            messages = [system_prompt, user_prompt]

        response = self.model.chat_completion(messages,
                                              temperature,
                                              max_tokens)

        # quit when we have an empty response
        if not response:
            logger.error("Received an empty response from the model.")
            return None

        logger.debug(f"Response: {response}")
        logger.info(f"total request tokens: \
{response['usage']['total_tokens']}")

        # TODO: try to generalize this by reading out of the model what the
        # effective context window is
        if int(response["usage"]["prompt_tokens"]) > self.context_size:
            logger.warning(f"Prompt tokens exceeded maximum context window of \
{self.context_size} with a total of {response['usage']['prompt_tokens']} \
tokens. Results are probably useless.")

        # tranform the string representation that the LLM proposed into a real
        # feature
        feature_str = str(response["choices"][0]["message"]["content"])
        return feature_str

    def _build_feature(self,
                       dataset: Dataset,
                       operator: str,
                       mapping: Any,
                       features: list[str]) -> pd.Series:
        """
        Build a new feature from the exsiting features of a given dataset and
        the proposed instructions.

        :param operator: the operator to apply
        :param feature: the list of currently available features

        Returns
            The transformed dataset containing the new feature, the mapping
            applied to the new feature
        """
        new_feature, mapping = build_feature_from_string(dataset.X,
                                                         operator,
                                                         mapping,
                                                         features)

        logger.debug(f"Len of new feature: {len(new_feature)}")
        logger.debug(f"New feature has mapping: \
                     {True if mapping is not None else False}")
        if new_feature.empty:
            logger.error("Feature could not be generated")
            raise Exception("Feature could not be generated")

        return new_feature, mapping

    def _validate_feature(self,
                          new_feature: pd.Series,
                          new_feature_name: str,
                          base_mean: float,
                          model: Any) -> tuple[bool, float]:
        """
        Validate the new feature by training a model on the training dataset
        containg the new feature.
        """
        keep_feature = True

        # required to prevent wrong labeling
        new_feature.name = new_feature_name

        train_X_with_new_feature = pd.concat([self.train_ds.X, new_feature],
                                             axis=1)

        cv_scores = run_stratified_cv(train_X_with_new_feature,
                                      self.train_ds.y,
                                      model)
        new_mean = mean(cv_scores)

        # calc the diff between the score without the feature and the score
        # with the new feature.
        diff = new_mean - base_mean
        if diff <= 0:
            logger.info("Feature does not improve the model, discarding.")
            keep_feature = False

        return keep_feature, diff

    def ask(self,
            n_features: int = 1,
            return_operators: bool = False,
            n_jobs: int = 1,
            seed: int = 1
            ) -> tuple[pd.DataFrame, pd.DataFrame] | \
            tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """
        Ask the generator to generate a new sample.

        :param n_features: int, The number of features to generate.

        Returns
        -------
            tuple[pd.DataFrame, pd.DataFrame] |
            tuple[pd.DataFrame, pd.DataFrame, list[str]]

            A pandas dataframe containing the new feature(s), one for the
            training, one for the test set. If additionally the
            'return_operartors' flag is set to True, a list of operators is
            returned as well.
        """

        # get the base mean score
        # TODO: generalize
        params = {"n_estimators": 100,
                  "learning_rate": 0.1,
                  "n_jobs": n_jobs,
                  "verbosity": -1}

        lgb = lgbm.LGBMClassifier(**params)
        base_cv_scores = run_stratified_cv(self.train_ds.X,
                                           self.train_ds.y,
                                           lgb)
        base_mean = mean(base_cv_scores)

        new_features_train = pd.DataFrame()
        new_features_test = pd.DataFrame()
        operators = []
        feedback = {}

        for i in range(n_features):
            logger.info(f"Generating feature {i + 1} of {n_features}")
            name, features_combination, reasoning, operator, train_feature, test_feature = \
                self._generate(feedback=feedback,
                               seed=seed)
            operators.append(operator)

            if train_feature is None or test_feature is None:
                logger.error("Feature could not be generated.")
                if return_operators:
                    return None, None, None
                return None, None

            # NOTE: for every feature we individually test if it will perform
            # better then the baseline, ref ms_1 for more info
            keep_feature, difference = self._validate_feature(train_feature,
                                                              name,
                                                              base_mean,
                                                              lgb)

            # update feedback
            feedback["feature"] = features_combination
            feedback["reasoning"] = reasoning
            feedback["difference"] = difference

            if keep_feature:
                new_features_train[name] = train_feature
                new_features_test[name] = test_feature

        if return_operators:
            return new_features_train, new_features_test, operators
        return new_features_train, new_features_test

    def tell(self):
        """
        Update the states of the generator based on the feedback.
        """
        raise NotImplementedError

    def _generate(self,
                  temperature: float = 0.5,
                  max_tokens: int = 300,
                  feedback: dict[str, str] = {},
                  seed: int = 1,
                  ) -> tuple[str, str, str, str, pd.Series, pd.Series]:
        """
        Generate a singular feature from a dataset.

        :param train_dataset: The training dataset to generate features from.
            Note that the generator is only fitted on this dataset and
            afterwards just transforms the test dataset.
        :param max_retries: The maximum number of retries to generate a
            feature, whenever the LLM generates a non-parseable output.
            Default is 5.
        :param temperature: The temperature of the model.
        :param max_tokens: The maximum number of tokens the model is allowed to
            generate.
        Returns
            [name, description, reasoning, operator, train_feature,
            test_feature]
            A tuple containing:
                name: the name of the new feature,
                description: the description of the new feature,
                reasoning: the reasoning behind the new feature,
                operator: the operator that was applied to generate the new
                    feature,
                train_feature: the build feature for the training dataset,
                test_feature: the build feature for the testing dataset
        """
        retries = 0

        train_feature, test_feature = pd.Series(), pd.Series()
        name, operator, reasoning, features_combination = "", "", "", ""
        while retries < self.max_retries:
            feature_str = self._fit(temperature=temperature,
                                    max_tokens=max_tokens,
                                    feedback=feedback,
                                    seed=seed)

            if feature_str is None:
                logger.error("Received no instructions to generate the new \
    feature, quitting")
                return None, None, None, None, None, None

            # extract the feature, order is Feature, Name, Description, Reasoning
            pattern = r'([A-Z]+):\s*(.*?)\s*(?=(?:[A-Z]+:)|(?:\n\n)|$)'
            matches = re.findall(pattern, feature_str, re.DOTALL)
            infos = {key: value.strip() for key, value in matches}
            try:
                reasoning = infos["REASONING"]
                operator = re.findall(r'[A-Za-z]+', infos["FEATURE"])[0]

                # get all features in every possible form
                features = re.findall(r'\((.*?)\)', infos["FEATURE"])
                if features:
                    features = [item.strip() for item in features[0].split(",")]

                    # check whether features are non numbers
                    for i, feature in enumerate(features):
                        if not feature.isdigit():
                            # try to match the feature position from the dataset
                            # to the feature name
                            feature_in_df = False
                            for idx, feature_name in enumerate(list(self.train_ds.X.columns)):
                                if feature_name.lower() == str(feature).lower():
                                    features[i] = idx + 1  # feature position is 1-based
                                    feature_in_df = True
                                    break
                            if not feature_in_df:
                                del features[i]
                else:
                    features = []


                features_combination = str(infos["FEATURE"])
                name = str(infos["NAME"]).strip(';')  # remove trailing ;
                description = str(infos["DESCRIPTION"])

            except Exception as e:
                logger.error(f"Received an error trying to extract the \
feature from the response: {str(e)}. Retrying for {retries +1} of \
{self.max_retries} retries.")
                retries += 1
                # TODO: return empty feedback for now if generated feature
                # breaks
                feedback = {}
                continue
            try:
                # transform the training dataset
                train_feature, mapping = \
                    self._build_feature(dataset=self.train_ds,
                                        operator=operator,
                                        mapping=None,
                                        features=features)
            except Exception as e:
                logger.warn(f"Received an error trying to transform the training \
    dataset: {str(e)}. Retrying for {retries +1} of {self.max_retries} retries.")
                retries += 1
                # TODO: return empty feedback for now if generated feature breaks
                feedback = {}
                continue

            try:
                # transform the testing dataset
                test_feature, _ = \
                    self._build_feature(dataset=self.test_ds,
                                        operator=operator,
                                        mapping=mapping,
                                        features=features)
            except Exception as e:
                logger.warn(f"Received an error trying to transform the \
testing dataset: {str(e)}. Retrying for {retries +1} of {self.max_retries} \
retries.")
                retries += 1
                # TODO: return empty feedback for now if generated feature breaks
                feedback = {}
                continue

            try:
                logger.info(f"Generated feature: {name}")
                logger.info(f"Operator: {operator}")
                logger.info(f"Features: {features}")
                logger.info(f"Reasoning: {reasoning}")
            except Exception as e:
                logger.error(f"Received an error trying to log the generated \
    feature: {str(e)}")

            break

        if name and operator and reasoning and \
                features_combination:
            return name, features_combination, reasoning, operator, \
                train_feature, test_feature
        else:
            return None, None, None, None, None, None