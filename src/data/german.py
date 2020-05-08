
# Names of the columns; last column is the label.
GERMAN_COLUMNS = ['checking_account',
                  'duration',
                  'credit_history',
                  'purpose',
                  'credit_amount',
                  'savings',
                  'employment_since',
                  'installment_rate',
                  'status_and_sex',
                  'debators',
                  'residence_since',
                  'property',
                  'age',
                  'installment_plans',
                  'housing',
                  'credits',
                  'job',
                  'num_liable',
                  'telephone',
                  'foreign_worker',
                  'assignment']

# The categorical columns to use during training (thus label column is excluded)
CATEGORICAL_COLUMNS_TRAIN_GERMAN = [
    'checking_account',
    'credit_history',
    'purpose',
    'savings',
    'employment_since',
    'status_and_sex',
    'debators',
    'property',
    'installment_plans',
    'housing',
    'job',
    'telephone',
    'foreign_worker',
]

# The numeric columns to use during training (thus label column is excluded)
NUMERIC_COLUMNS_TRAIN_GERMAN = [
    'duration',
    'credit_amount',
    'installment_rate',
    'residence_since',
    'age',
    'credits',
    'num_liable',
    # 'assignment'
]


def _sex_category(x):
    """A helper function to create the sex categories."""
    if x == 'A92' or x == 'A94':
        return 'female'
    else:
        return 'male'

def add_sex_category(df):
    """Mimics logic in scripts from
        https://worksheets.codalab.org/rest/bundles/0x2074cd3a10934e81accd6db433430ce8
        /contents/blob/utils/data.py"""
    df['sex'] = df['status_and_sex'].apply(_sex_category)
    return df