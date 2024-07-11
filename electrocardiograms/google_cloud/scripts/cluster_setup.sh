gcloud beta dataproc clusters create joe-test-cluster \
--enable-component-gateway --region us-central1 \
--master-machine-type n1-standard-2 --master-boot-disk-size 30 \
--num-workers 2 --worker-machine-type n1-standard-2 --worker-boot-disk-size 30 \
--image-version 2.0-debian10 --optional-components JUPYTER
