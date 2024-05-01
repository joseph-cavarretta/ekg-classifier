# Cardiac Arrythmia Classification

This project contains a deep neural network clssifiction model to classify electrocardiogram signals  
as one of these 5 categories:  

• N: Normal beat  
• S: Supraventricular premature beat  
• V: Premature ventricular contraction  
• F: Fusion of ventricular and normal beat  
• Q: Unclassifiable beat  

The best performing model implementation currently achieves an F-1 score of 0.97.  
The weakest classification performance is in predicting premature ventricular contractions currently.  
These types of arrythmias present a unique challenge in which the signal may look very normal,  
but is slightly compressed as the beat occurs prematurely.  

## Implementations
**Local:**  
• EKG_nn.ipynb - implements the model as a multi-layer perceptron using Sci-kit Learn  
• EKG_nn_pyspark.ipynb - implements the model using a local instance of Spark and the Spark ML Library  

**Cloud:**  
• pyspark_train_model.py - implements the model in a Google Cloud Platform instance of Spark  
running on a Google Dataproc cluster. Data is stored in Cloud Storage and BigQuery.  
