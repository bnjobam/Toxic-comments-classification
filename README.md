# Toxic-comments-classification
classification of toxic comments

This uses the nltk library for processing the comments.
-Removal of common words (stopwords)
-stemming to have the basic structure of a word
-an alternative to stopwords or could be used together is TF-IDF but I did not use it
The important words were words with frequency of more than 100 in the whole document (mycorpus, not included due to Kaggle rules) 
This mycorpus was then used to populate the feature matrix as columns, Bag of Words.
The model classifies comments into six non-mutually exclusive categories, so we use six models: modela, modelb, modelc, modeld, 
modele and modelf using extreme gradient boosted trees (xgboost) then combind them into a model called models.
The accuracy of the model was 0.969, that is the average of accuracies of the all the six models on the test set.
Drawbacks: The models took a considerable long time (about 5 hours) to train as compared to CNN which took less than 5minutes and RNN less than 3minutes
but the accuracies were almost the same, 0.963.
The CNN and RNN methods use the Glove pretrained word vectors from http://nlp.stanford.edu/data/glove.6B.zip.
