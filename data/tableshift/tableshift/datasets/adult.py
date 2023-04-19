"""Utilities and constants for the Adult dataset."""
import pandas as pd

from tableshift.core.features import Feature, FeatureList, cat_dtype

ADULT_RESOURCES = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
]

# Names to use for the features in the Adult dataset. These correspond to
# (human-readable) column names in the order of the columns in adult.data file.
ADULT_FEATURE_NAMES = ["Age", "Workclass", "fnlwgt", "Education",
                       "Education-Num",
                       "Marital Status",
                       "Occupation", "Relationship", "Race", "Sex",
                       "Capital Gain",
                       "Capital Loss",
                       "Hours per week", "Country", "Target"]

ADULT_FEATURES = FeatureList(features=[
    Feature("Age", float, "Age"),
    Feature("Workclass", cat_dtype, """Private, Self-emp-not-inc, 
    Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, 
    Never-worked."""),
    Feature("Education-Num", cat_dtype, "No documentation provided."),
    Feature("Marital Status", cat_dtype, """Married-civ-spouse, Divorced, 
    Never-married, Separated, Widowed, Married-spouse-absent, 
    Married-AF-spouse."""),
    Feature("Occupation", cat_dtype, """Tech-support, Craft-repair, 
    Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, 
    Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, 
    Priv-house-serv, Protective-serv, Armed-Forces."""),
    Feature("Relationship", cat_dtype, """Wife, Own-child, Husband, 
    Not-in-family, Other-relative, Unmarried."""),
    Feature("Race", cat_dtype, """White, Asian-Pac-Islander, 
    Amer-Indian-Eskimo, Other, Black."""),
    Feature("Sex", cat_dtype, "Female, Male."),
    Feature("Capital Gain", float, "No documentation provided."),
    Feature("Capital Loss", float, "No documentation provided."),
    Feature("Hours per week", float, "No documentation provided."),
    Feature("Country", cat_dtype, """United-States, Cambodia, England, 
    Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, 
    Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, 
    Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, 
    Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, 
    Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, 
    Holand-Netherlands."""),
    Feature("Target", int,
            description="Binary indicator for whether income >= 50k. See "
                        "preprocess_adult() below.",
            is_target=True),
], documentation="https://archive.ics.uci.edu/ml/datasets/Adult")


def preprocess_adult(df: pd.DataFrame):
    """Process a raw adult dataset."""
    df['Target'] = df['Target'].replace(
        {'<=50K': 0,
         '<=50K.': 0,
         '>50K': 1,
         '>50K.': 1})
    del df['Education']
    return df
