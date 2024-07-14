from google.cloud import bigquery

PROJECT_ID = 'joe-test-project-373803'
client = bigquery.Client()
dataset_id = f"{PROJECT_ID}.electrocardiograms"
dataset = bigquery.Dataset(dataset_id)
dataset.location = "US"

# raises google.api_core.exceptions.Conflict if the Dataset already exists within the project.
dataset = client.create_dataset(dataset, timeout=30)
print(f"Created dataset {client.project}.{ dataset.dataset_id}")