from google.cloud import bigquery


PROJECT_ID = "joe-test-project-373803"
BUCKET_ID = "joe-test-bucket-373803"
TRAIN_FILE_GCS_URI = f"gs://{BUCKET_ID}/electrocardiograms/data/mitbih_train.csv"
TEST_FILE_GCS_URI = f"gs://{BUCKET_ID}/electrocardiograms/data/mitbih_test.csv"
TRAIN_TABLE_ID = f"{PROJECT_ID}.electrocardiograms.processed_train_data"
TEST_TABLE_ID = f"{PROJECT_ID}.electrocardiograms.processed_test_data"


def load_bigquery(GCS_URI, TABLE_ID):
    client = bigquery.Client()
    job_config = bigquery.LoadJobConfig(
        autodetect = True,
        source_format = bigquery.SourceFormat.CSV,
        create_disposition = 'CREATE_IF_NEEDED',
        write_disposition = 'WRITE_TRUNCATE',
        skip_leading_rows = 1
        )

    load_job = client.load_table_from_uri(
        GCS_URI, TABLE_ID, job_config=job_config
    )
    load_job.result()
    table = client.get_table(TABLE_ID)

    print(f"Loaded {table.num_rows} rows to table {TABLE_ID}")


if __name__ == '__main__':
    load_bigquery(TRAIN_FILE_GCS_URI, TRAIN_TABLE_ID)
    load_bigquery(TEST_FILE_GCS_URI, TEST_TABLE_ID)