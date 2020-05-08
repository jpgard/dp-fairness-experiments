# Names of the columns; last column is the label.
ADULT_COLUMNS = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'salary'
]

CATEGORICAL_COLUMNS_TRAIN_ADULT = [
    'workclass',
    # 'fnlwgt',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    # 'salary'
]

NUMERIC_COLUMNS_TRAIN_ADULT = [
    'age',
    'education-num',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
]
