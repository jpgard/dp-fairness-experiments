"""
Usage:
python3 train_l2lr.py --learning_rate 0.01 --epochs 1000 --dataset german --niters 50 \
    --sensitive_attribute sex \
    --majority_attribute_label male \
    --minority_attribute_label female

python3 train_l2lr.py --learning_rate 0.01 --epochs 1000 --dataset adult \
    --niters 10 \
    --sensitive_attribute sex \
    --majority_attribute_label Male \
    --minority_attribute_label Female \
    --batch_size 512

python3 train_l2lr.py --learning_rate 0.01 --epochs 1000 --dataset adult \
    --niters 10 \
    --sensitive_attribute race \
    --majority_attribute_label White \
    --minority_attribute_label Black \
    --batch_size 512

python3 train_l2lr.py
    --learning_rate 0.01 \
    --epochs 1000 \
    --dataset compas \
    --niters 20 \
    --sensitive_attribute race \
    --majority_attribute_label Caucasian \
    --minority_attribute_label African-American \
    --batch_size 64
"""

import os
import shutil

import tensorflow as tf
import pandas as pd
from absl import app
from absl import flags
from sklearn.model_selection import train_test_split

from src.data import add_age_category
from src.data.german import GERMAN_COLUMNS, CATEGORICAL_COLUMNS_TRAIN_GERMAN, \
    NUMERIC_COLUMNS_TRAIN_GERMAN, add_sex_category
from src.data.adult import ADULT_COLUMNS, CATEGORICAL_COLUMNS_TRAIN_ADULT, \
    NUMERIC_COLUMNS_TRAIN_ADULT
from src.data.compas import COMPAS_COLUMNS, CATEGORICAL_COLUMNS_TRAIN_COMPAS, \
    NUMERIC_COLUMNS_TRAIN_COMPAS
from src.utils import keys
from src.utils.keys import ALL, MIN, MAJ, VALID_SUBSETS

FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rate", 0.01, "The learning rate to use during training.")
flags.DEFINE_float("l2_lambda", 1.0, "The L2 regularization coefficient.")
flags.DEFINE_integer("epochs", 5000, "The number of training epochs.")
flags.DEFINE_integer("batch_size", 64, "The batch size to use.")
flags.DEFINE_integer("n_classes", 2, "The number of classes of the outcome variable.")
flags.DEFINE_integer("niters", 50, "Number of experimental replicates to run")
flags.DEFINE_string("logdir", "./tmp", "The basic directory to save the model in.")
flags.DEFINE_string("sensitive_attribute", "sex", "The sensitive attribute to use.")
flags.DEFINE_string("majority_attribute_label", "male",
                    "The value of the sensitive attribute for the majority group.")
flags.DEFINE_string("minority_attribute_label", "female",
                    "The value of the sensitive attribute for the minority group.")
flags.DEFINE_enum("dataset", None, [keys.GERMAN, keys.ADULT, keys.COMPAS],
                  "String identifier for the dataset.")


def get_train_path(dataset: str):
    """Helper function to get the training dataset path."""
    if dataset == keys.GERMAN:
        train_path = "./datasets/german.data"
    elif dataset == keys.ADULT:
        train_path = "./datasets/adult.data"
    elif dataset == keys.COMPAS:
        train_path = "./datasets/compas-scores-two-years.csv"
    else:
        raise ValueError("train path not available for this dataset")
    return train_path


def get_colnames(dataset: str):
    """Helper function to get a 3-tuple of (all columns, categorical columns, numeric
    columns), where each element of the tuple is a list of column names."""
    if dataset == keys.GERMAN:
        return GERMAN_COLUMNS, CATEGORICAL_COLUMNS_TRAIN_GERMAN, \
               NUMERIC_COLUMNS_TRAIN_GERMAN
    elif dataset == keys.ADULT:
        return ADULT_COLUMNS, CATEGORICAL_COLUMNS_TRAIN_ADULT, NUMERIC_COLUMNS_TRAIN_ADULT
    elif dataset == keys.COMPAS:
        return COMPAS_COLUMNS, CATEGORICAL_COLUMNS_TRAIN_COMPAS, \
               NUMERIC_COLUMNS_TRAIN_COMPAS
    else:
        raise ValueError("train path not available for this dataset")


def get_df(dataset: str, train_path: str, flags):
    """Helper function to fetch the pd.DataFrame for the specified dataset and apply
    any necessary preprocessing."""
    if dataset == keys.GERMAN:
        input_df = pd.read_csv(train_path, header=None, names=GERMAN_COLUMNS, sep=' ')
        input_y = input_df.pop(GERMAN_COLUMNS[-1])
        # Shift the 1/2 coding of the label to a 0/1 coding
        input_y = (input_y.astype('int32') - 1)
        # make the age and sex category columns
        input_x = add_sex_category(input_df)
        input_x = add_age_category(input_x)
    elif dataset == keys.ADULT:
        # When reading csv, set skipinitialspace=True to trim extra spaces in the data
        # file.
        input_df = pd.read_csv(train_path, header=None, names=ADULT_COLUMNS,
                               skipinitialspace=True)
        input_y = input_df.pop(ADULT_COLUMNS[-1])
        input_x = add_age_category(input_df)
        input_x.drop(columns=['fnlwgt', ], inplace=True)
        # Recode the label from "<=50K" / ">50K" to 0/1
        input_y = (input_y == ">50K").astype('int32')
    elif dataset == keys.COMPAS:
        # The preprocessing here mimics preprocessing from 
        # https://worksheets.codalab.org/rest/bundles
        # /0x2074cd3a10934e81accd6db433430ce8/contents/blob/utils/data.py
        input_df = pd.read_csv(train_path)
        input_df = input_df.dropna(
            subset=["days_b_screening_arrest"])  # dropping missing vals
        # These filters are the same as propublica (refer to
        # https://github.com/propublica/compas-analysis)
        # If the charge date of a defendants Compas scored crime was not within 30 days
        # from when the person was arrested, we assume that because of data quality
        # reasons, that we do not have the right offense.
        input_df = input_df[(input_df["days_b_screening_arrest"] <= 30) & (
                input_df["days_b_screening_arrest"] >= -30)]
        # We coded the recidivist flag -- is_recid -- to be -1 if we could not find a
        # compas case at all.
        input_df = input_df[input_df["is_recid"] != -1]

        # In a similar vein, ordinary traffic offenses -- those with a c_charge_degree
        # of 'O' -- will not result in Jail time are removed (only two of them).
        input_df = input_df[
            input_df["c_charge_degree"] != "O"]  # F: felony, M: misconduct

        # We filtered the underlying data from Broward county to include only those
        # rows representing people who had either recidivated in two years, or had at
        # least two years outside of a correctional facility.
        input_df = input_df[input_df["score_text"] != "NA"]
        # Only consider blacks and whites for this analysis
        input_df = input_df[
            (input_df["race"] == "African-American") | (input_df["race"] == "Caucasian")]
        input_y = input_df.pop('two_year_recid')
        compas_train_cols = CATEGORICAL_COLUMNS_TRAIN_COMPAS + \
                            NUMERIC_COLUMNS_TRAIN_COMPAS
        input_x = input_df[compas_train_cols]
    else:
        raise ValueError
    # Filter the dataframe to keep only majority and minority groups (currently do not
    # look at non-binary sensitive attribute groups).
    input_x, input_y = filter_xy_df(input_x, input_y,
                                    colname=flags.sensitive_attribute,
                                    value=[flags.majority_attribute_label,
                                           flags.minority_attribute_label])
    return input_x, input_y


def make_model_uid(dataset: str, sensitive_attr: str, batch_size: int, epochs: int,
                   l2lambda: float):
    # TODO: add model parameters such as batch size, lr, epochs, etc here once
    #  experimentation gets more serious (currently we allow this to overwrite the
    #  models to avoid wasting disk space).
    uid = dataset
    uid += "-" + sensitive_attr
    uid += "bs{batch_size}e{epochs}l{l2lambda}".format(
        batch_size=batch_size,
        epochs=epochs,
        l2lambda=l2lambda
    )
    return uid


def filter_xy_df(df_x, df_y, colname, value):
    """Return a copy of df equivalent to the SQL statement SELECT * FROM df WHERE
    colname == value, and apply the same filtering to y. Value can be single value or a
    list of values."""
    if isinstance(value, list):
        filter_ix = df_x.index[df_x[colname].isin(value)]
    else:
        filter_ix = df_x.index[df_x[colname] == value]
    filtered_df_x = df_x.loc[filter_ix]
    filtered_df_y = df_y.loc[filter_ix]
    if len(filtered_df_x) == 0:
        raise ValueError("No rows matched condition {} == {}".format(colname, value))
    return filtered_df_x, filtered_df_y


def main(argv):
    def make_input_fn(data_df, label_df, shuffle=True, batch_size=FLAGS.batch_size,
                      num_epochs=FLAGS.epochs):
        """Make an input function based on the experiment parameters."""

        def input_function():
            ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
            if shuffle:
                ds = ds.shuffle(1000)
            ds = ds.batch(batch_size).repeat(num_epochs)
            return ds

        return input_function

    def build_l2lr_estimator(uid, subset):
        """Helper function to build an estimator."""

        logdir = os.path.join(FLAGS.logdir, uid, subset)
        shutil.rmtree(logdir, ignore_errors=True)
        # Build the estimator.
        est = tf.estimator.LinearClassifier(
            feature_columns=feature_columns,
            n_classes=FLAGS.n_classes,
            optimizer=lambda: tf.keras.optimizers.Ftrl(
                learning_rate=tf.compat.v1.train.exponential_decay(
                    learning_rate=FLAGS.learning_rate,
                    global_step=tf.compat.v1.train.get_global_step(),
                    decay_steps=10000,
                    decay_rate=0.96,
                    staircase=True),
                l2_regularization_strength=FLAGS.l2_lambda,
            ),
            model_dir=logdir
        )
        return est

    uid = make_model_uid(FLAGS.dataset, FLAGS.sensitive_attribute,
                         batch_size=FLAGS.batch_size, epochs=FLAGS.epochs,
                         l2lambda=FLAGS.l2_lambda)
    train_path = get_train_path(FLAGS.dataset)
    input_x, input_y = get_df(FLAGS.dataset, train_path, FLAGS)

    # Build the list of feature columns; these are required by the estimator API.
    all_columns, categorical_columns, numeric_columns = \
        get_colnames(
            FLAGS.dataset)
    feature_columns = []
    for feature_name in categorical_columns:
        vocabulary = input_x[feature_name].unique()
        feature_columns.append(
            tf.feature_column.categorical_column_with_vocabulary_list(feature_name,
                                                                      vocabulary))
    for feature_name in numeric_columns:
        feature_columns.append(
            tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

    # Print a summary of the sensitive attribute
    print("Sensitive attribute values:")
    print(input_x[FLAGS.sensitive_attribute].value_counts())
    results = list()  # List to hold the results dictionary for each iteration

    # Note that while the results are stored for every iteration, the models are
    # overwritten for every iteration to avoid wasting disk space. If changing this
    # behavior is desired, simply add the iternum to the result of make_model_model_uid.

    for iternum in range(FLAGS.niters):

        # Create the train-test split
        x_train, x_test, y_train, y_test = train_test_split(input_x, input_y,
                                                            train_size=0.9)

        x_train_majority, y_train_majority = filter_xy_df(
            x_train, y_train,
            colname=FLAGS.sensitive_attribute,
            value=FLAGS.majority_attribute_label)
        x_train_minority, y_train_minority = filter_xy_df(
            x_train, y_train,
            colname=FLAGS.sensitive_attribute,
            value=FLAGS.minority_attribute_label
        )
        x_test_majority, y_test_majority = filter_xy_df(
            x_test, y_test,
            colname=FLAGS.sensitive_attribute,
            value=FLAGS.majority_attribute_label)
        x_test_minority, y_test_minority = filter_xy_df(
            x_test, y_test,
            colname=FLAGS.sensitive_attribute,
            value=FLAGS.minority_attribute_label
        )
        # dictionary of {subset: (x: tf.train.Dataset, y: tf.train.Dataset)} for 
        # training/testing.
        train_datasets = dict()
        train_datasets[ALL] = (x_train, y_train)
        train_datasets[MAJ] = (x_train_majority, y_train_majority)
        train_datasets[MIN] = (x_train_minority, y_train_minority)
        test_datasets = dict()
        test_datasets[ALL] = (x_test, y_test)
        test_datasets[MAJ] = (x_test_majority, y_test_majority)
        test_datasets[MIN] = (x_test_minority, y_test_minority)

        # Train the models
        estimators = dict()
        for subset in VALID_SUBSETS:
            print("Training model for subset %s" % subset)
            estimator = build_l2lr_estimator(uid, subset)
            x_subset_train, y_subset_train = train_datasets[subset]
            train_input_fn = make_input_fn(x_subset_train, y_subset_train)
            estimator.train(input_fn=train_input_fn)
            x_subset_test, y_subset_test = test_datasets[subset]
            eval_input_fn = make_input_fn(x_subset_test, y_subset_test)
            result = estimator.evaluate(input_fn=eval_input_fn)
            result["train_subset"] = subset  # the data the model was trained on
            result["eval_subset"] = subset  # the data the model was evaluated on
            result["iteration"] = iternum
            print("Eval results for model trained/evaluated on subset {}:")
            print(result)
            estimators[subset] = estimator
            results.append(result)
        # Evaluate the base model on the majority and minority groups separately
        for subset in (MIN, MAJ):
            estimator = estimators[ALL]
            x_subset_test, y_subset_test = test_datasets[subset]
            eval_input_fn = make_input_fn(x_subset_test, y_subset_test)
            result = estimator.evaluate(input_fn=eval_input_fn)
            result["train_subset"] = ALL  # the data the model was trained on
            result["eval_subset"] = subset  # the data the model was evaluated on
            result["iteration"] = iternum
            results.append(result)

    csv_fp = "{}-{}-results.csv".format(uid, FLAGS.sensitive_attribute)
    pd.DataFrame(results).to_csv(csv_fp, index=False)

    return


if __name__ == "__main__":
    app.run(main)
