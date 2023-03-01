# MRI-Retrieval
## Retrieval of Brain MRI image

## augmentation.ipynb -->  Functions to augment training data.
## Preprocessing.ipynb --> Class including the following Functions :
![image](https://user-images.githubusercontent.com/62350791/222165148-f16f8a55-c7b0-4da9-9716-95341c8aba4a.png)

##                           Reorientation
<img width="390" alt="image" src="https://user-images.githubusercontent.com/62350791/222166127-b7d9ed88-79c7-4a9d-8f08-7f274dc791f1.png">

##                           Registration
<img width="390" alt="image" src="https://user-images.githubusercontent.com/62350791/222166223-46c9002c-a08a-4779-ad02-d1f07748e3c5.png">

##                           Skull Stripping
<img width="390" alt="image" src="https://user-images.githubusercontent.com/62350791/222166320-333e0bd1-4384-449b-8aab-9211d577e3c7.png">

##                           Bias Field Correction and 
<img width="390" alt="image" src="https://user-images.githubusercontent.com/62350791/222166408-28c8434b-ccc4-460a-89af-410b91b675b5.png">

##                           Intensity Normalization
<img width="390" alt="image" src="https://user-images.githubusercontent.com/62350791/222166490-71035bff-358f-4b99-8224-9edd4a0f7a84.png">

# Overall work flow 
![image](https://user-images.githubusercontent.com/62350791/222165375-7cc828cb-0d5e-4580-a669-507c380aa162.png)
The Overall framework of our proposed approach. We feed a mini-batch of samples from the preprocessed image to the networks 𝑓𝑜 and 𝑓𝑡 , which are both based on the same architecture, and for the first network, weights and biases are calculated and binary hash based on these weights and biases calculated and stored in a mini-batch table (yellow shaded), and for the second network, creates an initial 𝑙 length weights to generate an initial binary hashes to be stored in the dictionary (blue shaded), later a new value is enqueued and an old value is dequeued; then the hash codes in the mini-batch and in the dictionary concatenated and form another dictionary with size of 𝑙 + 𝑞, for mini-batch size of 𝑞. At this stage, an online triplet mining is used to determine the triplet loss by mining a triplet samples. Cross-entropy loss also be calculated, summed with the triplet loss, and its gradient propagates back to the input layer of the network 𝑓𝑜 ; while for the network 𝑓𝑡 updated via a momentum strategy method. Upon completion of training, the retrieval set applied to the network and the output of the final layer of the network are stored in the database. Following this, the query image is applied to the same network and a hash code vector is generated; then a Hamming distance is computed between this hash code and the hash codes of retrieval sets that are saved in the database; finally, a list of the top N ranked images are returned as a retrieval result.

# Training the model
## Examples of input neuroimages, their corresponding extracted features and hash codes using handedness label for different 𝑘
![image](https://user-images.githubusercontent.com/62350791/222164765-0aea61f4-a10e-4b59-acd1-6494bbf79818.png)

# Example 
## Query image
![image](https://user-images.githubusercontent.com/62350791/222163851-5bceaff2-ab82-4f4c-a09d-660ba083e24c.png)

## Top nine retrieved output
 ![image](https://user-images.githubusercontent.com/62350791/222163971-4b7a6849-a7bd-4cec-aa6f-145ff2a6c519.png)
 ## Overall performance (Mean Average Precision - MAP)
 <img width="459" alt="image" src="https://user-images.githubusercontent.com/62350791/222164356-2f57b229-c55b-440d-bb85-a2902f3f3b0d.png">


