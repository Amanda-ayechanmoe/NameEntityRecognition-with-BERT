# NameEntityRecognition-with-BERT
Article extraction from tweets mentioning research articles

# Objective 

The aim is to identify and extract research article titles from the twitter data containing research related posts by utilizing the state-of-the-art BERT model. 

# Data Preprocessing Steps 
1. Preprocessing the article names with generate unigram data set

* Example of extracting unigram tokens from original article title

<img src="https://github.com/Amanda-ayechanmoe/NameEntityRecognition-with-BERT/blob/master/screenshots/example%20of%20extracting%20unigram%20tokens%20from%20original%20article%20title.PNG" width="500"/>

2. Preporcessing the tweets

3. Tagging each word in the tweet posts

* Example of tagging each word in the tweet post

<img src="https://github.com/Amanda-ayechanmoe/NameEntityRecognition-with-BERT/blob/master/screenshots/example%20of%20tagging%20each%20word%20in%20the%20tweet%20post.PNG" width="500"/>

4. Performing BIO tagging

* Example of performing BIO tagging 

<img src="https://github.com/Amanda-ayechanmoe/NameEntityRecognition-with-BERT/blob/master/screenshots/example%20of%20performing%20BIO%20tagging.PNG" width="500"/>

# Feature Engineering 

* The output dataset from the preprocessing stage would contain the attributes 

<img src="https://github.com/Amanda-ayechanmoe/NameEntityRecognition-with-BERT/blob/master/screenshots/extracted%20features.PNG" width="500"/>

These word wise tweet data are recombined using the Id values to give a new dataset of complete tweets and their corresponding tags, tweet wise. 
Then the Bert Tokenizer from pretrained bert-base-uncased model is used to tokenize the tweet data.
The tokenized words are converted to ids and are post padded with zeros to form a consistent input dataset shape for the model.
The tags are enumerated and converted to ids and are post padded with zeros for consistent input shape.
The feature of attention masking in the Bert model was used to mask the extra padded values so that the paddings are not considered and do not affect the performance of the model.
Finally, the dataset is split into training and validation sets and are converted to tensordataset in batches, ready to be fed into the Bert model.

# BERT Model Implementation
TensorFlow implementation of the Bert model was used to perform the custom NER task of identifying article titles.
This training data is fed in batches to calculate the loss, followed by a backward pass that recalculates the gradients and tweaks the weights. 
Then gradient clipping and Adam’s optimization is performed on the model parameters in order to avoid the problem of exploding gradients and speeding up gradient descent.
The model used almost entirely the default hyperparameter set except for minor changes, as it has already been tuned for high performance on NLP tasks. 
The Bert-Base variation of the Bert model was used.
The Bert base has a 12 encoder (also called transformer blocks) setup with a theoretical limit to process 512 tokens at a time.
The token limit was set to 100 as the processed tweets were within this length limit.
The batch sizes for processing was set to 32 as per the recommendations of the original Bert paper.

# Validation and Testing
With some basic hyperparameter tuning by using Adam’s optimizer with weight decay, the model was able to achieve a validation F1 score of 0.71 in 1 epoch.
By adding an additional max gradient clipping step to reduce the effects of gradient blowing up, the model was able to attain a validation F1 score of 0.76.
As the original Bert paper suggests an epoch range of 3-5 for high performance in the NLP tasks using that algorithm, by modifying the epoch to 3 and utilizing the newer implementation on TF2.0,
The model was able to achieve the highest final F1 score of 0.91 in 3 epochs on the testing data, using an Adam’s optimizer with weight decay setup, along with max grad clipping and a scheduler that automatically readjusts the learning rate of the algorithm based on the progress of the training phase.




