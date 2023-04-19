import pandas as pd
from scipy.stats import percentileofscore

from tableshift.core.features import Feature, FeatureList, cat_dtype

HELOC_FEATURES = FeatureList(features=[
    Feature('RiskPerformance', int, "Paid as negotiated flag (12-36 Months). "
                                    "String of Good and Bad", is_target=True),
    Feature('ExternalRiskEstimateLow', int,
            "Consolidated version of risk markers"),
    Feature('MSinceOldestTradeOpen', int, 'Months Since Oldest Trade Open'),
    Feature('MSinceMostRecentTradeOpen', int,
            'Months Since Most Recent Trade Open'),
    Feature('AverageMInFile', int, 'Average Months in File'),
    Feature('NumSatisfactoryTrades', int, 'Number Satisfactory Trades'),
    Feature('NumTrades60Ever2DerogPubRec', int, 'Number Trades 60+ Ever'),
    Feature('NumTrades90Ever2DerogPubRec', int, 'Number Trades 90+ Ever'),
    Feature('PercentTradesNeverDelq', int, 'Percent Trades Never Delinquent'),
    Feature('MSinceMostRecentDelq', int,
            'Months Since Most Recent Delinquency'),
    Feature('MaxDelq2PublicRecLast12M', cat_dtype,
            """Max Delq/Public Records Last 12 Months. Values: 0 derogatory 
            comment 1 120+ days delinquent 2 90 days delinquent 3 60 days 
            delinquent 4 30 days delinquent 5, 6 unknown delinquency 7 
            current and never delinquent 8, 9 all other"""),
    Feature('MaxDelqEver', cat_dtype,
            """Max Delinquency Ever. Values: 1 No such value 2 derogatory 
            comment 3 120+ days delinquent 4 90 days delinquent 5 60 days 
            delinquent 6 30 days delinquent 7 unknown delinquency 8 current 
            and never delinquent 9 all other"""),
    Feature('NumTotalTrades', int,
            'Number of Total Trades (total number of credit accounts)'),
    Feature('NumTradesOpeninLast12M', int,
            'Number of Trades Open in Last 12 Months'),
    Feature('PercentInstallTrades', int, 'Percent Installment Trades'),
    Feature('MSinceMostRecentInqexcl7days', int,
            'Months Since Most Recent Inq excl 7 days'),
    Feature('NumInqLast6M', int, 'Number of Inq Last 6 Months'),
    Feature('NumInqLast6Mexcl7days', int, """Number of Inq Last 6 Months excl 
    7days. Excluding the last 7 days removes inquiries that are likely due to 
    price comparision shopping."""),
    # Feature('NetFractionRevolvingBurden', int, """Net Fraction Revolving
    # Burden. This is revolving balance divided by credit limit"""),
    Feature('NetFractionRevolvingBurden', int),
    Feature('NetFractionInstallBurden', int, """Net Fraction Installment 
    Burden. This is installment balance divided by original loan amount"""),
    Feature('NumRevolvingTradesWBalance', int,
            'Number Revolving Trades with Balance'),
    Feature('NumInstallTradesWBalance', int,
            'Number Installment Trades with Balance'),
    Feature('NumBank2NatlTradesWHighUtilization', int,
            'Number Bank/Natl Trades w high utilization ratio'),
    Feature('PercentTradesWBalance', int, 'Percent Trades with Balance'),
], documentation="""Data dictionary .xslx file can be accessed after filling 
out the data agreement at https://community.fico.com/s/explainable-machine
-learning-challenge """)


def preprocess_heloc(df: pd.DataFrame) -> pd.DataFrame:
    # Transform target to integer
    target = HELOC_FEATURES.target
    df[target] = (df[target] == "Good").astype(int)

    df['ExternalRiskEstimateLow'] = (df['ExternalRiskEstimate'] <= 63)
    df.drop(columns=['ExternalRiskEstimate'], inplace=True)
    return df
