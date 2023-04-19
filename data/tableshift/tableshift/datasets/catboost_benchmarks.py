"""
CatBoost quality benchmarks; adapted from
https://github.com/catboost/benchmarks/tree/master/quality_benchmarks
"""
import re

import pandas as pd
from pandas import DataFrame
from tableshift.core.features import Feature, FeatureList, cat_dtype

AMAZON_FEATURES = FeatureList(features=[
    Feature('ACTION', float,
            "ACTION is 1 if the resource was approved, 0 if the resource was not",
            name_extended="access to resource was approved",
            is_target=True),
    Feature('RESOURCE', int, "An ID for each resource",
            name_extended="resource ID"),
    Feature('MGR_ID', int,
            "The EMPLOYEE ID of the manager of the current EMPLOYEE ID record; an employee may have only one manager at a time",
            name_extended="manager ID"),
    Feature('ROLE_ROLLUP_1', int,
            "Company role grouping category id 1 (e.g. US Engineering)",
            name_extended="company role grouping category 1"),
    Feature('ROLE_ROLLUP_2', int,
            "Company role grouping category id 2 (e.g. US Retail)",
            name_extended="company role grouping category 2"),
    Feature('ROLE_DEPTNAME', int,
            'Company role department description (e.g. Retail)',
            name_extended='company role department description'),
    Feature('ROLE_TITLE', int,
            'Company role business title description (e.g. Senior Engineering Retail Manager)',
            name_extended='company role business title description'),
    Feature('ROLE_FAMILY_DESC', int,
            'Company role family extended description (e.g. Retail Manager, Software Engineering)',
            name_extended='company role family extended description'),
    Feature('ROLE_FAMILY', int,
            'Company role family description (e.g. Retail Manager)',
            name_extended='company role family description'),
    Feature('ROLE_CODE', int,
            'Company role code; this code is unique to each role (e.g. Manager)',
            name_extended='company role code'),
], documentation="https://www.kaggle.com/c/amazon-employee-access-challenge")

APPETENCY_FEATURES = FeatureList(features=[
    Feature('Var6', float),
    Feature('Var7', float),
    Feature('Var13', float),
    Feature('Var21', float),
    Feature('Var22', float),
    Feature('Var24', float),
    Feature('Var25', float),
    Feature('Var28', float),
    Feature('Var35', float),
    Feature('Var38', float),
    Feature('Var44', float),
    Feature('Var51', float),
    Feature('Var57', float),
    Feature('Var65', float),
    Feature('Var72', float),
    Feature('Var73', float),
    Feature('Var74', float),
    Feature('Var76', float),
    Feature('Var78', float),
    Feature('Var81', float),
    Feature('Var83', float),
    Feature('Var85', float),
    Feature('Var94', float),
    Feature('Var109', float),
    Feature('Var112', float),
    Feature('Var113', float),
    Feature('Var119', float),
    Feature('Var123', float),
    Feature('Var125', float),
    Feature('Var126', float),
    Feature('Var132', float),
    Feature('Var133', float),
    Feature('Var134', float),
    Feature('Var140', float),
    Feature('Var143', float),
    Feature('Var144', float),
    Feature('Var149', float),
    Feature('Var153', float),
    Feature('Var160', float),
    Feature('Var163', float),
    Feature('Var173', float),
    Feature('Var181', float),
    Feature('Var189', float),
    Feature('Var191', cat_dtype),
    Feature('Var192', cat_dtype),
    Feature('Var193', cat_dtype),
    Feature('Var194', cat_dtype),
    Feature('Var195', cat_dtype),
    Feature('Var196', cat_dtype),
    Feature('Var197', cat_dtype),
    Feature('Var198', cat_dtype),
    Feature('Var199', cat_dtype),
    Feature('Var200', cat_dtype),
    Feature('Var201', cat_dtype),
    Feature('Var202', cat_dtype),
    Feature('Var203', cat_dtype),
    Feature('Var204', cat_dtype),
    Feature('Var205', cat_dtype),
    Feature('Var206', cat_dtype),
    Feature('Var207', cat_dtype),
    Feature('Var208', cat_dtype),
    Feature('Var210', cat_dtype),
    Feature('Var211', cat_dtype),
    Feature('Var212', cat_dtype),
    Feature('Var213', cat_dtype),
    Feature('Var214', cat_dtype),
    Feature('Var215', cat_dtype),
    Feature('Var216', cat_dtype),
    Feature('Var217', cat_dtype),
    Feature('Var218', cat_dtype),
    Feature('Var219', cat_dtype),
    Feature('Var220', cat_dtype),
    Feature('Var221', cat_dtype),
    Feature('Var222', cat_dtype),
    Feature('Var223', cat_dtype),
    Feature('Var224', cat_dtype),
    Feature('Var225', cat_dtype),
    Feature('Var226', cat_dtype),
    Feature('Var227', cat_dtype),
    Feature('Var228', cat_dtype),
    Feature('Var229', cat_dtype),
    Feature('Var149_imputed', float),
    Feature('Var83_imputed', float),
    Feature('Var7_imputed', float),
    Feature('Var181_imputed', float),
    Feature('Var119_imputed', float),
    Feature('Var76_imputed', float),
    Feature('Var173_imputed', float),
    Feature('Var21_imputed', float),
    Feature('Var143_imputed', float),
    Feature('Var125_imputed', float),
    Feature('Var13_imputed', float),
    Feature('Var189_imputed', float),
    Feature('Var28_imputed', float),
    Feature('Var35_imputed', float),
    Feature('Var133_imputed', float),
    Feature('Var22_imputed', float),
    Feature('Var126_imputed', float),
    Feature('Var6_imputed', float),
    Feature('Var78_imputed', float),
    Feature('Var163_imputed', float),
    Feature('Var140_imputed', float),
    Feature('Var134_imputed', float),
    Feature('Var153_imputed', float),
    Feature('Var81_imputed', float),
    Feature('Var38_imputed', float),
    Feature('Var94_imputed', float),
    Feature('Var85_imputed', float),
    Feature('Var51_imputed', float),
    Feature('Var132_imputed', float),
    Feature('Var160_imputed', float),
    Feature('Var112_imputed', float),
    Feature('Var74_imputed', float),
    Feature('Var123_imputed', float),
    Feature('Var44_imputed', float),
    Feature('Var109_imputed', float),
    Feature('Var72_imputed', float),
    Feature('Var24_imputed', float),
    Feature('Var144_imputed', float),
    Feature('Var25_imputed', float),
    Feature('Var65_imputed', float),
    Feature('label', float, name_extended='class label', is_target=True),
], documentation='https://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data')

CLICK_FEATURES = FeatureList(features=[
    Feature('click', float, is_target=True,
            name_extended='user clicked ad at least once'),
    Feature('impression', cat_dtype),
    Feature('url_hash', cat_dtype, name_extended='URL hash'),
    Feature('ad_id', cat_dtype, name_extended='ad ID'),
    Feature('advertiser_id', float, name_extended='advertiser ID'),
    Feature('depth', float,
            name_extended='number of ads impressed in this session'),
    Feature('position', cat_dtype,
            name_extended='order of this ad in the impression list'),
    Feature('query_id', cat_dtype, name_extended='query ID'),
    Feature('keyword_id', cat_dtype, name_extended='keyword ID'),
    Feature('title_id', cat_dtype, name_extended='title ID'),
    Feature('description_id', cat_dtype, name_extended='description ID'),
    Feature('user_id', float, name_extended='user ID'),
], documentation='https://www.kaggle.com/competitions/kddcup2012-track2/data ,'
                 'http://www.kdd.org/kdd-cup/view/kdd-cup-2012-track-2')

KICK_FEATURES = FeatureList(features=[
    Feature('RefId', cat_dtype,
            name_extended='Unique (sequential) number assigned to vehicles'),
    Feature('IsBadBuy', int, is_target=True,
            name_extended='indicator for whether the kicked vehicle was an avoidable purchase'),
    Feature('PurchDate', cat_dtype,
            name_extended='date the vehicle was purchased at auction'),
    Feature('Auction', cat_dtype,
            name_extended="Auction provider at which the  vehicle was purchased"),
    Feature('VehYear', int, name_extended="manufacture year of the vehicle"),
    Feature('VehicleAge', int,
            name_extended="Years elapsed since the manufacture year"),
    Feature('Make', cat_dtype),
    Feature('Model', cat_dtype),
    Feature('Trim', cat_dtype, name_extended='Trim Level'),
    Feature('SubModel', cat_dtype),
    Feature('Color', cat_dtype),
    Feature('Transmission', cat_dtype,
            name_extended='Vehicle transmission type'),
    Feature('WheelTypeID', cat_dtype,
            name_extended='type id of the vehicle wheel'),
    Feature('WheelType', cat_dtype, name_extended='wheel type description'),
    Feature('VehOdo', int, name_extended='odometer reading'),
    Feature('Nationality', cat_dtype, name_extended="manufacturer's country"),
    Feature('Size', cat_dtype, name_extended='size category of the vehicle'),
    Feature('TopThreeAmericanName', cat_dtype,
            name_extended='manufacturer is one of the top three American manufacturers'),
    Feature('MMRAcquisitionAuctionAveragePrice', float,
            name_extended='average acquisition price at auction for this vehicle in average condition at time of purchase'),
    Feature('MMRAcquisitionAuctionCleanPrice', float,
            name_extended='average acquisition price at auction for this vehicle in the above average condition at time of purchase'),
    Feature('MMRAcquisitionRetailAveragePrice', float,
            name_extended='average retail price for this vehicle in average condition at time of purchase'),
    Feature('MMRAcquisitonRetailCleanPrice', float,
            name_extended='average retail price for this vehicle in above average condition at time of purchase'),
    Feature('MMRCurrentAuctionAveragePrice', float,
            name_extended='average acquisition price at auction for this vehicle in average condition as of current day'),
    Feature('MMRCurrentAuctionCleanPrice', float,
            name_extended='average acquisition price at auction for this vehicle in above average condition as of current day'),
    Feature('MMRCurrentRetailAveragePrice', float,
            name_extended='average retail price for this vehicle in average condition as of current day'),
    Feature('MMRCurrentRetailCleanPrice', float,
            name_extended='average retail price for this vehicle in above average condition as of current day'),
    Feature('PRIMEUNIT', cat_dtype,
            name_extended='vehicle would have a higher demand than a standard purchase'),
    Feature('AUCGUART', cat_dtype,
            name_extended='acquisition method of vehicle'),
    Feature('BYRNO', int,
            name_extended='level guarantee provided by auction for the vehicle'),
    Feature('VNZIP1', cat_dtype, 'ZIP code where the car was purchased'),
    Feature('VNST', cat_dtype,
            name_extended='State where the the car was purchased'),
    Feature('VehBCost', float,
            name_extended='acquisition cost paid for the vehicle at time of purchase'),
    Feature('IsOnlineSale', int,
            name_extended='vehicle was originally purchased online'),
    Feature('WarrantyCost', int,
            name_extended='Warranty price (with term=36 month and mileage=36K)'),
], documentation="https://www.kaggle.com/competitions/DontGetKicked/data")


def preprocess_kick(df: DataFrame) -> DataFrame:
    return df


def preprocess_click(data: DataFrame) -> DataFrame:
    categorical_features = {1, 2, 3, 6, 7, 8, 9, 10}

    def clean_string(s):
        return "v_" + re.sub('[^A-Za-z0-9]+', "_", str(s))

    for i in categorical_features:
        data[data.columns[i]] = data[data.columns[i]].apply(clean_string)

    data["click"] = data["click"].apply(lambda x: 1 if x != 0 else -1)

    return data


def preprocess_appetency(data: DataFrame) -> DataFrame:
    """Adapted from https://github.com/catboost/benchmarks/blob/master
    /quality_benchmarks/prepare_appetency_churn_upselling
    /prepare_appetency_churn_upselling.ipynb """

    # preparing categorical features

    categorical_features = {190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
                            200, 201, 202, 203, 204, 205, 206, 207, 209, 210,
                            211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
                            221, 222, 223, 224, 225, 226, 227, 228}

    numeric_colnames = {column for i, column in enumerate(data.columns) if
                        i not in categorical_features}

    # Note: we do not need to explicitly cast categorical features to string;
    # these are already of dtype object.

    for i in categorical_features:
        data[data.columns[i]] = data[data.columns[i]].fillna("MISSING").apply(
            str).astype("category")

    # prepare numerical features

    # drop any numeric column that is >= 95% missing
    all_missing = data.columns[(pd.isnull(data).sum() >= 0.95 * len(data))]
    data.drop(columns=all_missing, inplace=True)
    numeric_colnames -= set(all_missing)

    columns_to_impute = []
    for column in numeric_colnames:
        if pd.isnull(data[column]).any():
            columns_to_impute.append(column)

    for column_name in columns_to_impute:
        data[column_name + "_imputed"] = pd.isnull(data[column_name]).astype(
            float)
        data[column_name].fillna(0, inplace=True)

    for column in numeric_colnames:
        data[column] = data[column].astype(float)

    return data
