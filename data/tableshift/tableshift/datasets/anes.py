"""
Utilities for ANES Time Series Cumulative Data File.

List of variables: https://electionstudies.org/wp-content/uploads/2019/09/anes_timeseries_cdf_codebook_Varlist.pdf
Codebook: https://electionstudies.org/wp-content/uploads/2022/09/anes_timeseries_cdf_codebook_var_20220916.pdf
"""
import pandas as pd

from tableshift.core.features import Feature, FeatureList, cat_dtype

# Note that "state" feature is named as VCF0901b; see below. Note that '99' is
# also a valid value, but it contains all missing targets .
ANES_STATES = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL',
               'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD',
               'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ',
               'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN',
               'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

# Note that "year" feature is named as VCF0004; see below.
ANES_YEARS = [1948, 1952, 1954, 1956, 1958, 1960, 1962, 1964, 1966, 1968, 1970,
              1972, 1974, 1976, 1978, 1980, 1982, 1984, 1986, 1988, 1990, 1992,
              1994, 1996, 1998, 2000, 2002, 2004, 2008, 2012, 2016, 2020]

# U.S. Census Regions; see 'VCF0112' feature below.
ANES_REGIONS = ['MISSING', '1.0', '2.0', '4.0', '3.0']

# This is a very preliminary feature list. We should probably
# try to find a good reference/principled heuristics for selecting these.
# I generally tried to select a few from each category, with an emphasis on
# questions that would be asked/relevant every year (i.e. not
# questions about Kennedy, Vietnam, Cold War, etc.).
# Only pre-election questions. Also dropped questions that were
# asked in only 3 or fewer years.

# We give the actual coding values for potential sensitive variables; for others
# we mostly give the question title; see the documentation linked above for
# details.

ANES_FEATURES = FeatureList(features=[
    Feature("VCF0004", int, "Year of study"),
    Feature("VCF0901b", cat_dtype, """State of interview - state postal 
    abbreviation, 99. NA; wrong district identified (2000) INAP. question not 
    used"""),

    # PARTISANSHIP AND ATTITUDES TOWARDS PARTIES
    Feature('VCF0218', float, "The Democratic Party – feeling thermometer"),
    Feature('VCF0224', float, "The Republican Party – feeling thermometer"),
    Feature('VCF0301', cat_dtype,
            """Generally speaking, do you usually think of yourself as a 
            Republican, a Democrat, an Independent, or what? (IF REPUBLICAN 
            OR DEMOCRAT) you call yourself a strong (REP/DEM) or a not very 
            strong (REP/DEM)? (IF INDEPENDENT, OTHER [1966 AND LATER:] OR NO 
            PREFERENCE; 2008: OR DK) Do you think of yourself as closer to 
            the Republican or Democratic party?"""),
    Feature('VCF0302', cat_dtype,
            "Generally speaking, do you usually think of yourself as a "
            "Republican, a Democrat, an Independent, or what?"),
    Feature('VCF9008', cat_dtype,
            """Which party do you think would do a better job of handling the 
            problem of pollution and (1990,1994: protection of) the 
            environment?"""),
    Feature('VCF9010', cat_dtype,
            """Do you think inflation would be handled better by the 
            Democrats, by the Republicans, or about the same by both?"""),
    Feature('VCF9011', cat_dtype,
            """Do you think the problems of unemployment would be handled 
            better by the Democrats, by the Republicans, or about the same by 
            both?"""),
    Feature('VCF9201', cat_dtype,
            """(I’d like to know what you think about each of our political 
            parties. After I read the name of a political party, please rate 
            it on a scale from 0 to 10, where 0 means you strongly dislike 
            that party and 10 means that you strongly like that party. If I 
            come to a party you haven’t heard of or you feel you do not know 
            enough about, just say so.) [The first party is: / Using the same 
            scale where would you place:] the Democratic party {INTERVIEWER: 
            DO NOT PROBE DON’T KNOW}"""),
    Feature('VCF9202', cat_dtype,
            """(I’d like to know what you think about each of our political 
            parties. After I read the name of a political party, please rate 
            it on a scale from 0 to 10, where 0 means you strongly dislike 
            that party and 10 means that you strongly like that party. If I 
            come to a party you haven’t heard of or you feel you do not know 
            enough about, just say so.) [The first party is: / Using the same 
            scale where would you place:] the Republican party {INTERVIEWER: 
            DO NOT PROBE DON’T KNOW}"""),
    Feature('VCF9203', cat_dtype,
            """Would you say that any of the parties in the United States 
            represents your views reasonably well? {INTERVIEWER: DO NOT PROBE 
            DON’T KNOW}"""),
    Feature('VCF9204', cat_dtype,
            """(Would you say that any of the parties in the United States 
            represents your views reasonably well?) Which party represents 
            your views best? {INTERVIEWER: DO NOT PROBE DON’T KNOW}"""),
    Feature('VCF9205', cat_dtype,
            """Which party do you think would do a better job of handling the 
            nation’s economy, the Democrats, the Republicans, or wouldn’t 
            there be much difference between them? {1996: IF ‘NO DIFFERENCE’ 
            AND ‘NEITHER PARTY’ ARE VOLUNTEERED, DO NOT PROBE RESPONSES. 
            2000-later: IF ‘DK’ OR ‘NEITHER PARTY’ IS VOLUNTEERED, DO NOT 
            PROBE]}"""),
    Feature('VCF9206', cat_dtype,
            """Do you think it is better when one party controls both the 
            presidency and Congress; better when control is split between the 
            Democrats and Republicans, or doesn’t it matter?"""),

    # PERCEIVED POSITIONS OF PARTIES
    Feature('VCF0521', cat_dtype,
            """Which party do you think is more likely to favor a stronger [
            1978,1980, 1984: more powerful; 1988,1992: a powerful] government 
            in Washington – the Democrats, the Republicans, or wouldn’t there 
            be any difference between them on this?"""),
    Feature('VCF0523', cat_dtype,
            """Which political party do you think is more in favor of cutting 
            military spending - the Democrats, the Republicans, or wouldn’t 
            there be much difference between them?"""),

    # CANDIDATE AND INCUMBENT EVALUATIONS
    Feature('VCF0428', float, "President thermometer."),
    Feature('VCF0429', float, "Vice-president thermometer."),

    # CANDIDATE/INCUMBENT PERFORMANCE EVALUATIONS
    Feature('VCF0875', cat_dtype, "MENTION 1: WHAT IS THE MOST IMPORTANT "
                                  "NATIONAL PROBLEM"),
    Feature('VCF9052', cat_dtype,
            """Let’s talk about the country as a whole. Would you say that 
            things in the country are generally going very well, fairly well, 
            not too well or not well at all?"""),
    # ISSUES
    Feature('VCF0809', cat_dtype, "GUARANTEED JOBS AND INCOME SCALE."),
    Feature('VCF0839', cat_dtype, "GOVERNMENT SERVICES-SPENDING SCALE"),
    Feature('VCF0822', cat_dtype,
            """As to the economic policy of the government – I mean steps 
            taken to fight inflation or unemployment – would you say the 
            government is doing a good job, only fair, or a poor job?"""),
    Feature('VCF0870', cat_dtype, "BETTER OR WORSE ECONOMY IN PAST YEAR"),
    Feature('VCF0843', cat_dtype, "DEFENSE SPENDING SCALE"),
    Feature('VCF9045', cat_dtype,
            "POSITION OF THE U.S. WEAKER/STRONGER IN THE PAST YEAR"),
    Feature('VCF0838', cat_dtype, "BY LAW, WHEN SHOULD ABORTION BE ALLOWED"),
    Feature('VCF9239', cat_dtype, "HOW IMPORTANT IS GUN CONTROL ISSUE TO R"),
    # IDEOLOGY AND VALUES
    Feature('VCF0803', cat_dtype, "LIBERAL-CONSERVATIVE SCALE"),
    Feature('VCF0846', cat_dtype, "IS RELIGION IMPORTANT TO RESPONDENT"),
    # SYSTEM SUPPORT
    Feature('VCF0601', cat_dtype, "APPROVE PARTICIPATION IN PROTESTS"),
    Feature('VCF0606', cat_dtype, "HOW MUCH DOES THE FEDERAL GOVERNMENT WASTE "
                                  "TAX MONEY"),
    Feature('VCF0612', cat_dtype, "VOTING IS THE ONLY WAY TO HAVE A SAY IN "
                                  "GOVERNMENT"),
    Feature('VCF0615', cat_dtype, "MATTER WHETHER RESPONDENT VOTES OR NOT"),
    Feature('VCF0616', cat_dtype, "SHOULD THOSE WHO DON’T CARE ABOUT ELECTION "
                                  "OUTCOME VOTE"),
    Feature('VCF0617', cat_dtype, "SHOULD SOMEONE VOTE IF THEIR PARTY CAN’T "
                                  "WIN"),
    Feature('VCF0310', cat_dtype, "INTEREST IN THE ELECTIONS"),
    Feature('VCF0743', cat_dtype, "DOES R BELONG TO POLITICAL ORGANIZATION OR "
                                  "CLUB"),
    Feature('VCF0717', cat_dtype, "RESPONDENT TRY TO INFLUENCE THE VOTE OF "
                                  "OTHERS DURING THE CAMPAIGN"),
    Feature('VCF0718', cat_dtype, "RESPONDENT ATTEND POLITICAL "
                                  "MEETINGS/RALLIES DURING THE CAMPAIGN"),
    Feature('VCF0720', cat_dtype, "RESPONDENT DISPLAY CANDIDATE "
                                  "BUTTON/STICKER DURING THE CAMPAIGN"),
    Feature('VCF0721', cat_dtype, "RESPONDENT DONATE MONEY TO PARTY OR "
                                  "CANDIDATE DURING THE CAMPAIGN"),

    # REGISTRATION, TURNOUT, AND VOTE CHOICE
    Feature('VCF0701', cat_dtype, "REGISTERED TO VOTE PRE-ELECTION"),
    Feature('VCF0702', int, "DID RESPONDENT VOTE IN THE NATIONAL "
                                  "ELECTIONS 1. No, did not vote 2. Yes, "
                                  "voted 0. DK; NA; no Post IW; refused to "
                                  "say if voted; Washington D.C. ("
                                  "presidential years only)",
            is_target=True),
    # MEDIA
    Feature('VCF0675', cat_dtype,
            "HOW MUCH OF THE TIME DOES RESPONDENT TRUST THE "
            "MEDIA TO REPORT FAIRLY"),
    Feature('VCF0724', cat_dtype, "WATCH TV PROGRAMS ABOUT THE ELECTION "
                                  "CAMPAIGNS"),
    Feature('VCF0725', cat_dtype, "HEAR PROGRAMS ABOUT CAMPAIGNS ON THE RADIO "
                                  "2- CATEGORY"),
    Feature('VCF0726', cat_dtype, "ARTICLES ABOUT ELECTION CAMPAIGNS IN "
                                  "MAGAZINES"),
    Feature('VCF0745', cat_dtype, "SAW ELECTION CAMPAIGN INFORMATION ON THE "
                                  "INTERNET"),

    # PERSONAL AND DEMOGRAPHIC
    Feature('VCF0101', float, "RESPONDENT - AGE"),
    Feature('VCF0104', cat_dtype, """RESPONDENT - GENDER 1. Male 2. Female 3. 
    Other (2016)"""),
    Feature('VCF0105a', cat_dtype, """RACE-ETHNICITY SUMMARY, 7 CATEGORIES 1.0 
    White non-Hispanic (1948-2012) 2.0 Black non-Hispanic (1948-2012) 3.0 Asian 
    or Pacific Islander, non-Hispanic (1966-2012) 4.0 American Indian or 
    Alaska Native non-Hispanic (1966-2012) 5.0 Hispanic (1966-2012) 6.0 Other 
    or multiple races, non-Hispanic (1968-2012) 7.0 Non-white and non-black (
    1948-1964)"""),
    Feature('VCF0115', cat_dtype, """RESPONDENT - OCCUPATION GROUP 6-CATEGORY 
    1. Professional and managerial 2. Clerical and sales workers 3. Skilled, 
    semi-skilled and service workers 4. Laborers, except farm 5. Farmers, 
    farm managers, farm laborers and foremen; forestry and fishermen 6. 
    Homemakers (1972-1992: 7 IN VCF0116, 4 in VCF0118; 1952-1970: 4 in 
    VCF0118)"""),
    Feature('VCF0140a', cat_dtype, """RESPONDENT - EDUCATION 7-CATEGORY 1. 8 
    grades or less (‘grade school’) 2. 9-12 grades (‘high school’), 
    no diploma/equivalency; less than high school credential (2020) 3. 12 
    grades, diploma or equivalency 4. 12 grades, diploma or equivalency plus 
    non-academic training 5. Some college, no degree; junior/community 
    college level degree (AA degree) 6. BA level degrees 7. Advanced degrees 
    incl. LLB"""),
    Feature('VCF0112', cat_dtype, """Region - U.S. Census 1. Northeast (CT, 
    ME, MA, NH, NJ, NY, PA, RI, VT) 2. North Central (IL, IN, IA, KS, MI, MN, 
    MO, NE, ND, OH, SD, WI) 3. South (AL, AR, DE, D.C., FL, GA, KY, LA, MD, 
    MS, NC, OK, SC,TN, TX, VA, WV) 4. West (AK, AZ, CA, CO, HI, ID, MT, NV, 
    NM, OR, UT, WA, WY)"""),
],
    documentation="https://electionstudies.org/data-center/anes-time-series"
                  "-cumulative-data-file/")


def preprocess_anes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ANES_FEATURES.names]
    df = df.dropna(subset=[ANES_FEATURES.target])
    df[ANES_FEATURES.target] = (
            df[ANES_FEATURES.target].astype(float) == 2.0).astype(int)
    for f in ANES_FEATURES.features:
        if f.kind == cat_dtype:
            df[f.name] = df[f.name].fillna("MISSING").apply(str) \
                .astype("category")
    return df
