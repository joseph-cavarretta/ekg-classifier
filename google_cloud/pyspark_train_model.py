from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

from pyspark.sql.functions import isnull
from pyspark.sql.functions import col

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier

from sklearn.metrics import classification_report


sc = SparkContext()
spark = SparkSession(sc)
sc.setLogLevel("ERROR")

PROJECT = 'joe-test-project-373803'
GCS_BUCKET_PATH = 'gs://joe-test-bucket-373803/electrocardiograms/model.pkl'
BQ_TRAIN_TABLE = 'electrocardiograms.processed_train_data'
BQ_TEST_TABLE = 'electrocardiograms.processed_test_data'

def main():
    data = load_data()
    train_set = data[0].cache()
    test_set = data[1].cache()
    check_nulls(train_set)
    check_nulls(test_set)
    mlpModel = train_model(train_set)
    eval_model(mlpModel, test_set)


def load_data():
    train_set = spark.read.format("bigquery").option(
        "table", f"{PROJECT}.{BQ_TRAIN_TABLE}").load()

    test_set = spark.read.format("bigquery").option(
        "table", f"{PROJECT}.{BQ_TEST_TABLE}").load()

    return (train_set, test_set)


def check_nulls(dataframe):
    df = dataframe
    assert len([x for x in df.columns if df.filter(col(x).isNull()).count() > 0]) == 0
    assert len([x for x in df.columns if df.filter(col(x).isNull()).count() > 0]) == 0


def train_model(train_data):
    feature_cols = train_data.select(train_data.columns[:-1])

    indexer = StringIndexer(
        inputCol='double_field_187', 
        outputCol="label"
    )
    assembler = VectorAssembler(
        inputCols=feature_cols.columns,
        outputCol="features"
    )

    layers = [187, 75, 75, 75, 5]
    mlp = MultilayerPerceptronClassifier(
        maxIter=300, 
        layers=layers, 
        blockSize=128, 
        seed=42
    )
    pipeline = Pipeline(stages=[indexer, assembler, mlp])
    mlpModel = pipeline.fit(train_data)
    mlpModel.save(GCS_BUCKET_PATH)
    return mlpModel


def eval_model(model, test_data):
    predictions = model.transform(test_data)
    y_true = predictions.select(['label']).collect()
    y_pred = predictions.select(['prediction']).collect()
    print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    main()