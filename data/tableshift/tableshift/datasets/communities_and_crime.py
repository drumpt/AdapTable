import numpy as np
import pandas as pd

from tableshift.core.features import Feature, FeatureList, cat_dtype

CANDC_RESOURCES = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/communities/communities.data"]

# Column names of the raw input features from UCI.
CANDC_INPUT_FEATURES = [
    'state', 'county', 'community', 'communityname', 'fold', 'population',
    'householdsize', 'racepctblack', 'racePctWhite', 'racePctAsian',
    'racePctHisp', 'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up',
    'numbUrban', 'pctUrban', 'medIncome', 'pctWWage', 'pctWFarmSelf',
    'pctWInvInc', 'pctWSocSec', 'pctWPubAsst', 'pctWRetire', 'medFamInc',
    'perCapInc', 'whitePerCap', 'blackPerCap', 'indianPerCap', 'AsianPerCap',
    'OtherPerCap', 'HispPerCap', 'NumUnderPov', 'PctPopUnderPov',
    'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore', 'PctUnemployed',
    'PctEmploy', 'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu',
    'PctOccupMgmtProf', 'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv',
    'TotalPctDiv', 'PersPerFam', 'PctFam2Par', 'PctKids2Par',
    'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids', 'PctWorkMom',
    'NumIlleg', 'PctIlleg', 'NumImmig', 'PctImmigRecent', 'PctImmigRec5',
    'PctImmigRec8', 'PctImmigRec10', 'PctRecentImmig', 'PctRecImmig5',
    'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell',
    'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerOccupHous',
    'PersPerOwnOccHous', 'PersPerRentOccHous', 'PctPersOwnOccup',
    'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant',
    'PctHousOccup', 'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos',
    'MedYrHousBuilt', 'PctHousNoPhone', 'PctWOFullPlumb', 'OwnOccLowQuart',
    'OwnOccMedVal', 'OwnOccHiQuart', 'RentLowQ', 'RentMedian', 'RentHighQ',
    'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc',
    'MedOwnCostPctIncNoMtg', 'NumInShelters', 'NumStreet', 'PctForeignBorn',
    'PctBornSameState', 'PctSameHouse85', 'PctSameCity85', 'PctSameState85',
    'LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps',
    'LemasSwFTFieldPerPop', 'LemasTotalReq', 'LemasTotReqPerPop',
    'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol', 'PctPolicWhite',
    'PctPolicBlack', 'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor',
    'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz', 'PolicAveOTWorked', 'LandArea',
    'PopDens', 'PctUsePubTrans', 'PolicCars', 'PolicOperBudg',
    'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', 'LemasPctOfficDrugUn',
    'PolicBudgPerPop', 'ViolentCrimesPerPop'
]

CANDC_STATE_LIST = ['1', '10', '11', '12', '13', '16', '18', '19', '2', '20',
                    '21', '22', '23', '24', '25', '27', '28', '29', '32', '33',
                    '34', '35', '36', '37', '38', '39', '4', '40', '41', '42',
                    '44', '45', '46', '47', '48', '49', '5', '50', '51', '53',
                    '54', '55', '56', '6', '8', '9']

CANDC_FEATURES = FeatureList([
    Feature('state', float,
            """US state (by number) - not counted as predictive above, but if 
            considered, should be consided nominal (nominal)"""),
    Feature('county', float,
            """numeric code for county - not predictive, and many missing 
            values (numeric)"""),
    Feature('community', float,
            """numeric code for community - not predictive and many missing 
            values (numeric)"""),
    Feature('communityname', float,
            """community name - not predictive - for information only (string)"""),

    Feature('population', float,
            """population for community', float, (numeric - decimal)"""),
    Feature('householdsize', float,
            """mean people per household (numeric - decimal)"""),
    Feature('racepctblack', float,
            """percentage of population that is african american (numeric - 
            decimal)"""),
    Feature('racePctWhite', float,
            """percentage of population that is caucasian (numeric - 
            decimal)"""),
    Feature('racePctAsian', float,
            """percentage of population that is of asian heritage (numeric - 
            decimal)"""),
    Feature('racePctHisp', float,
            """percentage of population that is of hispanic heritage (numeric 
            - decimal)"""),
    Feature('agePct12t21', float,
            """percentage of population that is 12-21 in age (numeric - 
            decimal)"""),
    Feature('agePct12t29', float,
            """percentage of population that is 12-29 in age (numeric - 
            decimal)"""),
    Feature('agePct16t24', float,
            """percentage of population that is 16-24 in age (numeric - 
            decimal)"""),
    Feature('agePct65up', float,
            """percentage of population that is 65 and over in age (numeric - 
            decimal)"""),
    Feature('numbUrban', float,
            """number of people living in areas classified as urban (numeric 
            - decimal)"""),
    Feature('pctUrban', float,
            """percentage of people living in areas classified as urban (
            numeric - decimal)"""),
    Feature('medIncome', float,
            """median household income (numeric - decimal)"""),
    Feature('pctWWage', float,
            """percentage of households with wage or salary income in 1989 (
            numeric - decimal)"""),
    Feature('pctWFarmSelf', float,
            """percentage of households with farm or self employment income 
            in 1989 (numeric - decimal)"""),
    Feature('pctWInvInc', float,
            """percentage of households with investment / rent income in 1989 
            (numeric - decimal)"""),
    Feature('pctWSocSec', float,
            """percentage of households with social security income in 1989 (
            numeric - decimal)"""),
    Feature('pctWPubAsst', float,
            """percentage of households with public assistance income in 1989 
            (numeric - decimal)"""),
    Feature('pctWRetire', float,
            """percentage of households with retirement income in 1989 (
            numeric - decimal)"""),
    Feature('medFamInc', float,
            """median family income (differs from household income for 
            non-family households) (numeric - decimal)"""),
    Feature('perCapInc', float, """per capita income (numeric - decimal)"""),
    Feature('whitePerCap', float,
            """per capita income for caucasians (numeric - decimal)"""),
    Feature('blackPerCap', float,
            """per capita income for african americans (numeric - decimal)"""),
    Feature('indianPerCap', float,
            """per capita income for native americans (numeric - decimal)"""),
    Feature('AsianPerCap', float,
            """per capita income for people with asian heritage (numeric - 
            decimal)"""),
    Feature('OtherPerCap', float,
            """per capita income for people with 'other' heritage (numeric - 
            decimal)"""),
    Feature('HispPerCap', float,
            """per capita income for people with hispanic heritage (numeric - 
            decimal)"""),
    Feature('NumUnderPov', float,
            """number of people under the poverty level (numeric - decimal)"""),
    Feature('PctPopUnderPov', float,
            """percentage of people under the poverty level (numeric - 
            decimal)"""),
    Feature('PctLess9thGrade', float,
            """percentage of people 25 and over with less than a 9th grade 
            education (numeric - decimal)"""),
    Feature('PctNotHSGrad', float,
            """percentage of people 25 and over that are not high school 
            graduates (numeric - decimal)"""),
    Feature('PctBSorMore', float,
            """percentage of people 25 and over with a bachelors degree or 
            higher education (numeric - decimal)"""),
    Feature('PctUnemployed', float,
            """percentage of people 16 and over, in the labor force, 
            and unemployed (numeric - decimal)"""),
    Feature('PctEmploy', float,
            """percentage of people 16 and over who are employed (numeric - 
            decimal)"""),
    Feature('PctEmplManu', float,
            """percentage of people 16 and over who are employed in 
            manufacturing (numeric - decimal)"""),
    Feature('PctEmplProfServ', float,
            """percentage of people 16 and over who are employed in 
            professional services (numeric - decimal)"""),
    Feature('PctOccupManu', float,
            """percentage of people 16 and over who are employed in 
            manufacturing (numeric - decimal) ########"""),
    Feature('PctOccupMgmtProf', float,
            """percentage of people 16 and over who are employed in 
            management or professional occupations (numeric - decimal)"""),
    Feature('MalePctDivorce', float,
            """percentage of males who are divorced (numeric - decimal)"""),
    Feature('MalePctNevMarr', float,
            """percentage of males who have never married (numeric - 
            decimal)"""),
    Feature('FemalePctDiv', float,
            """percentage of females who are divorced (numeric - decimal)"""),
    Feature('TotalPctDiv', float,
            """percentage of population who are divorced (numeric - 
            decimal)"""),
    Feature('PersPerFam', float,
            """mean number of people per family (numeric - decimal)"""),
    Feature('PctFam2Par', float,
            """percentage of families (with kids) that are headed by two 
            parents (numeric - decimal)"""),
    Feature('PctKids2Par', float,
            """percentage of kids in family housing with two parents (numeric 
            - decimal)"""),
    Feature('PctYoungKids2Par', float,
            """percent of kids 4 and under in two parent households (numeric 
            - decimal)"""),
    Feature('PctTeen2Par', float,
            """percent of kids age 12-17 in two parent households (numeric - 
            decimal)"""),
    Feature('PctWorkMomYoungKids', float,
            """percentage of moms of kids 6 and under in labor force (numeric 
            - decimal)"""),
    Feature('PctWorkMom', float,
            """percentage of moms of kids under 18 in labor force (numeric - 
            decimal)"""),
    Feature('NumIlleg', float,
            """number of kids born to never married (numeric - decimal)"""),
    Feature('PctIlleg', float,
            """percentage of kids born to never married (numeric - decimal)"""),
    Feature('NumImmig', float,
            """total number of people known to be foreign born (numeric - 
            decimal)"""),
    Feature('PctImmigRecent', float,
            """percentage of _immigrants_ who immigated within last 3 years (
            numeric - decimal)"""),
    Feature('PctImmigRec5', float,
            """percentage of _immigrants_ who immigated within last 5 years (
            numeric - decimal)"""),
    Feature('PctImmigRec8', float,
            """percentage of _immigrants_ who immigated within last 8 years (
            numeric - decimal)"""),
    Feature('PctImmigRec10', float,
            """percentage of _immigrants_ who immigated within last 10 years 
            (numeric - decimal)"""),
    Feature('PctRecentImmig', float,
            """percent of _population_ who have immigrated within the last 3 
            years (numeric - decimal)"""),
    Feature('PctRecImmig5', float,
            """percent of _population_ who have immigrated within the last 5 
            years (numeric - decimal)"""),
    Feature('PctRecImmig8', float,
            """percent of _population_ who have immigrated within the last 8 
            years (numeric - decimal)"""),
    Feature('PctRecImmig10', float,
            """percent of _population_ who have immigrated within the last 10 
            years (numeric - decimal)"""),
    Feature('PctSpeakEnglOnly', float,
            """percent of people who speak only English (numeric - decimal)"""),
    Feature('PctNotSpeakEnglWell', float,
            """percent of people who do not speak English well (numeric - 
            decimal)"""),
    Feature('PctLargHouseFam', float,
            """percent of family households that are large (6 or more) (
            numeric - decimal)"""),
    Feature('PctLargHouseOccup', float,
            """percent of all occupied households that are large (6 or more 
            people) (numeric - decimal)"""),
    Feature('PersPerOccupHous', float,
            """mean persons per household (numeric - decimal)"""),
    Feature('PersPerOwnOccHous', float,
            """mean persons per owner occupied household (numeric - 
            decimal)"""),
    Feature('PersPerRentOccHous', float,
            """mean persons per rental household (numeric - decimal)"""),
    Feature('PctPersOwnOccup', float,
            """percent of people in owner occupied households (numeric - 
            decimal)"""),
    Feature('PctPersDenseHous', float,
            """percent of persons in dense housing (more than 1 person per 
            room) (numeric - decimal)"""),
    Feature('PctHousLess3BR', float,
            """percent of housing units with less than 3 bedrooms (numeric - 
            decimal)"""),
    Feature('MedNumBR', float,
            """median number of bedrooms (numeric - decimal)"""),
    Feature('HousVacant', float,
            """number of vacant households (numeric - decimal)"""),
    Feature('PctHousOccup', float,
            """percent of housing occupied (numeric - decimal)"""),
    Feature('PctHousOwnOcc', float,
            """percent of households owner occupied (numeric - decimal)"""),
    Feature('PctVacantBoarded', float,
            """percent of vacant housing that is boarded up (numeric - 
            decimal)"""),
    Feature('PctVacMore6Mos', float,
            """percent of vacant housing that has been vacant more than 6 
            months (numeric - decimal)"""),
    Feature('MedYrHousBuilt', float,
            """median year housing units built (numeric - decimal)"""),
    Feature('PctHousNoPhone', float,
            """percent of occupied housing units without phone (in 1990, 
            this was rare!) (numeric - decimal)"""),
    Feature('PctWOFullPlumb', float,
            """percent of housing without complete plumbing facilities (
            numeric - decimal)"""),
    Feature('OwnOccLowQuart', float,
            """owner occupied housing - lower quartile value (numeric - 
            decimal)"""),
    Feature('OwnOccMedVal', float,
            """owner occupied housing - median value (numeric - decimal)"""),
    Feature('OwnOccHiQuart', float,
            """owner occupied housing - upper quartile value (numeric - 
            decimal)"""),
    Feature('RentLowQ', float,
            """rental housing - lower quartile rent (numeric - decimal)"""),
    Feature('RentMedian', float,
            """rental housing - median rent (Census variable H32B from file 
            STF1A) (numeric - decimal)"""),
    Feature('RentHighQ', float,
            """rental housing - upper quartile rent (numeric - decimal)"""),
    Feature('MedRent', float,
            """median gross rent (Census variable H43A from file STF3A - 
            includes utilities) (numeric - decimal)"""),
    Feature('MedRentPctHousInc', float,
            """median gross rent as a percentage of household income (numeric 
            - decimal)"""),
    Feature('MedOwnCostPctInc', float,
            """median owners cost as a percentage of household income - for 
            owners with a mortgage (numeric - decimal)"""),
    Feature('MedOwnCostPctIncNoMtg', float,
            """median owners cost as a percentage of household income - for 
            owners without a mortgage (numeric - decimal)"""),
    Feature('NumInShelters', float,
            """number of people in homeless shelters (numeric - decimal)"""),
    Feature('NumStreet', float,
            """number of homeless people counted in the street (numeric - 
            decimal)"""),
    Feature('PctForeignBorn', float,
            """percent of people foreign born (numeric - decimal)"""),
    Feature('PctBornSameState', float,
            """percent of people born in the same state as currently living (
            numeric - decimal)"""),
    Feature('PctSameHouse85', float,
            """percent of people living in the same house as in 1985 (5 years 
            before) (numeric - decimal)"""),
    Feature('PctSameCity85', float,
            """percent of people living in the same city as in 1985 (5 years 
            before) (numeric - decimal)"""),
    Feature('PctSameState85', float,
            """percent of people living in the same state as in 1985 (5 years 
            before) (numeric - decimal)"""),
    Feature('LemasSwornFT', float,
            """number of sworn full time police officers (numeric - decimal)"""),
    Feature('LemasSwFTPerPop', float,
            """sworn full time police officers per 100K population (numeric - 
            decimal)"""),
    Feature('LemasSwFTFieldOps', float,
            """number of sworn full time police officers in field operations 
            (on the street as opposed to administrative etc) (numeric - 
            decimal)"""),
    Feature('LemasSwFTFieldPerPop', float,
            """sworn full time police officers in field operations (on the 
            street as opposed to administrative etc) per 100K population (
            numeric - decimal)"""),
    Feature('LemasTotalReq', float,
            """total requests for police (numeric - decimal)"""),
    Feature('LemasTotReqPerPop', float,
            """total requests for police per 100K popuation (numeric - 
            decimal)"""),
    Feature('PolicReqPerOffic', float,
            """total requests for police per police officer (numeric - 
            decimal)"""),
    Feature('PolicPerPop', float,
            """police officers per 100K population (numeric - decimal)"""),
    Feature('RacialMatchCommPol', float,
            """a measure of the racial match between the community and the 
            police force. High values indicate proportions in community and 
            police force are similar (numeric - decimal)"""),
    Feature('PctPolicWhite', float,
            """percent of police that are caucasian (numeric - decimal)"""),
    Feature('PctPolicBlack', float,
            """percent of police that are african american (numeric - 
            decimal)"""),
    Feature('PctPolicHisp', float,
            """percent of police that are hispanic (numeric - decimal)"""),
    Feature('PctPolicAsian', float,
            """percent of police that are asian (numeric - decimal)"""),
    Feature('PctPolicMinor', float,
            """percent of police that are minority of any kind (numeric - 
            decimal)"""),
    Feature('OfficAssgnDrugUnits', float,
            """number of officers assigned to special drug units (numeric - 
            decimal)"""),
    Feature('NumKindsDrugsSeiz', float,
            """number of different kinds of drugs seized (numeric - 
            decimal)"""),
    Feature('PolicAveOTWorked', float,
            """police average overtime worked (numeric - decimal)"""),
    Feature('LandArea', float,
            """land area in square miles (numeric - decimal)"""),
    Feature('PopDens', float,
            """population density in persons per square mile (numeric - 
            decimal)"""),
    Feature('PctUsePubTrans', float,
            """percent of people using public transit for commuting (numeric 
            - decimal)"""),
    Feature('PolicCars', float,
            """number of police cars (numeric - decimal)"""),
    Feature('PolicOperBudg', float,
            """police operating budget (numeric - decimal)"""),
    Feature('LemasPctPolicOnPatr', float,
            """percent of sworn full time police officers on patrol (numeric 
            - decimal)"""),
    Feature('LemasGangUnitDeploy', float,
            """gang unit deployed (numeric - decimal - but really ordinal - 0 
            means NO, 1 means YES, 0.5 means Part Time)"""),
    Feature('LemasPctOfficDrugUn', float,
            """percent of officers assigned to drug units (numeric - decimal)"""),
    Feature('PolicBudgPerPop', float,
            """police operating budget per population (numeric - decimal)"""),

    Feature('Target', int, """Binary indicator for whether total number of 
    violent crimes per 100K popuation exceeds threshold.""", is_target=True),
    Feature('Race', int),
    Feature('income_level_above_median', int)
])


def preprocess_candc(df: pd.DataFrame,
                     target_threshold: float = 0.08) -> pd.DataFrame:
    df = df.replace('?', np.nan).dropna(axis=1)
    df["Race"] = df['racePctWhite'].apply(
        lambda x: 1 if x >= 0.85 else 0)
    income_thresh = df["medIncome"].median()
    df["income_level_above_median"] = df["medIncome"].apply(
        lambda x: 1 if x > income_thresh else 0)

    df = df.drop(columns=['racePctAsian', 'racePctHisp',
                          'racepctblack', 'whitePerCap',
                          'blackPerCap', 'indianPerCap',
                          'AsianPerCap',  # 'OtherPerCap',
                          'HispPerCap',
                          'racePctWhite', 'medIncome',
                          'fold',
                          ], axis=1).rename(
        columns={'ViolentCrimesPerPop': "Target"})

    # The label of a community is 1 if that community is among the
    # 70% of communities with the highest crime rate and 0 otherwise,
    # following Khani et al. and Kearns et al. 2018.
    df["Target"] = (df["Target"] >= target_threshold).astype(int)
    df["state"] = df["state"].apply(str).astype("category")
    return df
