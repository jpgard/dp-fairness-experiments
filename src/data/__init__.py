def _age_category(age):
    """A helper function to create the age categories."""
    if age > 45:
        return 'Greater than 45'
    elif age < 25:
        return 'Less than 25'
    else:
        return '25-45'

def add_age_category(df):
    """Mimics logic in scripts from
        https://worksheets.codalab.org/rest/bundles/0x2074cd3a10934e81accd6db433430ce8
        /contents/blob/utils/data.py"""
    df['age_cat'] = df['age'].apply(_age_category)
    return df