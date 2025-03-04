args = None
AbaloneOpt = {
    'nominal_columns':[
        'Sex'
    ]
}

CholesterolOpt = {
    'path': './data/OpenML_arff/dataset_2190_cholesterol.arff',
    'nominal_columns':[
        'sex',
        'cp',
        'fbs',
        'restecg',
        'exang',
        'slope',
        'thal'
    ]
}

SarcosOpt = {
    'path': './data/OpenML_arff/sarcos.arff',
    'nominal_columns':[]
}

BostonOpt = {
    'path': './data/OpenML_arff/boston.arff',
    'nominal_columns':['CHAS', 'RAD']
}

NewsOpt = {
    'path': './data/OpenML_arff/colleges_usnews.arff',
    'nominal_columns':['College_name', 'State']
}

YpropOpt = {
    'nominal_columns':['oz40', 'oz42', 'oz46', 'oz50', 'oz69', 'oz71', 'oz73', 'oz79', 'oz96', 'oz100', 'oz107', 'oz108', 'oz111', 'oz112', 'oz113', 'oz115', 'oz135', 'oz206', 'oz222', 'oz234']
}

BlackfridayOpt = {
    'nominal_columns':['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']
}

BrazilianhousesOpt = {
    'nominal_columns':['city', 'animal', 'furniture']
}

DiamondsOpt = {
    'nominal_columns':['cut', 'color', 'clarity']
}

SeattlecrimeOpt = {
     'nominal_columns':['Precinct', 'Sector']
}

TopoOpt = {
    'nominal_columns':['oz256', 'oz260', 'oz265']
}

HouseOpt = {
    'nominal_columns':['waterfront', 'date_year']
}

UkairOpt = {
    'nominal_columns':['Month', 'DayofWeek', 'Environment.Type']
}

AnalcatOpt = {
    'nominal_columns':['Liberal', 'Unconstitutional', 'Precedent_alteration', 'Unanimous']
}

DelayOpt = {
    'nominal_columns':['vehicle_type', 'direction', 'weekday']
}

BikeOpt = {
    'nominal_columns':['season', 'year', 'holiday', 'workingday', 'weather']
}

TaxiOpt = {
    'nominal_columns':['VendorID', 'store_and_fwd_flag', 'RatecodeID', 'PULocationID', 'DOLocationID', 'extra', 'mta_tax', 'improvement_surcharge', 'trip_type']
}

SoilOpt = {
    'nominal_columns':['isns']
}

GpuOpt = {
    'nominal_columns':['KWG', 'KWI', 'STRM', 'STRN', 'SA', 'SB']
}