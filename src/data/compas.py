COMPAS_COLUMNS = [
    'id', 'name', 'first', 'last',
    'compas_screening_date', 'sex', 'dob',
    'age', 'age_cat', 'race', 'juv_fel_count',
    'decile_score',
    'juv_misd_count', 'juv_other_count', 'priors_count',
    'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
    'c_case_number',
    'c_offense_date', 'c_arrest_date',
    'c_days_from_compas',
    'c_charge_degree', 'c_charge_desc', 'is_recid',
    'r_case_number',
    'r_charge_degree', 'r_days_from_arrest',
    'r_offense_date',
    'r_charge_desc', 'r_jail_in', 'r_jail_out',
    'violent_recid',
    'is_violent_recid', 'vr_case_number',
    'vr_charge_degree',
    'vr_offense_date', 'vr_charge_desc',
    'type_of_assessment',
    'decile_score.1', 'score_text', 'screening_date',
    'v_type_of_assessment', 'v_decile_score',
    'v_score_text',
    'v_screening_date', 'in_custody', 'out_custody',
    'priors_count.1',
    'start', 'end', 'event', 'two_year_recid']

CATEGORICAL_COLUMNS_TRAIN_COMPAS = [
    "age_cat",
    "race",
    "sex",
    "c_charge_degree"]

NUMERIC_COLUMNS_TRAIN_COMPAS = [
    "priors_count",
    # "two_year_recid"
]
