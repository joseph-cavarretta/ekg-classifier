gcloud dataproc jobs submit pyspark \
--cluster=joe-test-cluster --region=us-central1 \
--jars gs://spark-lib/bigquery/spark-bigquery-with-dependencies_2.12-0.28.0.jar \
gs://joe-test-bucket-373803/electrocardiograms/pyspark_train_model.py