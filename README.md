# Comparative Analysis of CNN, RNN and HAN for Text Classification with GloVe Data Model
Capstone Project as part of the course CSC591: Algorithms for Data-Guided Business Intelligence

**Team Members:**

- Rachit Shah (rshah25)

- Sourabh Sandanshi (ssandan)

[You can find the screencast of the presentation here.](https://drive.google.com/file/d/1PzghKuV0vGEa602jNGYQil0UhcRVTn_q/view?usp=sharing)

## How to Run
You can run the IPython Notebooks in the folders "Dataset - 1" and "Dataset -2" by either downloading on your local machine or using the "Open in Colab" link to run the notebooks on Google Colab. To run the notebooks you will need the [dataset zip file](http://mlg.ucd.ie/datasets/bbc.html), the [glove embedding zip file](http://nlp.stanford.edu/data/glove.6B.zip) and the model weights too if you want to replicate our results. You can find all these files in the [Google Drive folder](https://drive.google.com/drive/folders/15A2b8uNEfak_1Gfh-c85dTtOHJrfrk7h?usp=sharing). Add all the files in this link to your google drive (root folder) in order to run the notebooks.

## Problem Description
We are using DNN models - CNN, RNN and HAN to classify News articles into different categories. The goal of our project is to predict/classify categories of news based on the content of news articles from the BBC website. We have used 2 datasets:
1. BBC News - http://mlg.ucd.ie/datasets/bbc.html (2225 news, 5 categories)
2. 20_newsgroup - https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html (18846 news posts, 20 categories)

| Set      | Train| Validation | Test | labels |
| ----------- | ----------- |----------- |----------- |----------- |
| Dataset 1 (BBC)  | 1424 |	356	| 445 | 	5       |
| Dataset 2 (20_newsgroup) | 9051	| 2263	| 7532	| 20|

## Solution Framework
We will use a standard solution framework typically used in NLP problems namely, inputting data, preprocessing, model architecture, training, hyperparameter tuning and prediction. The figure below shows an overview of the steps we have taken to build the model and comparing the output of different models on both of our datasets. As you can see, we have used Google Colaboratory to train our models on cloud using fast GPUs. We first load the data into pandas dataframes and apply preprocessing like removing punctuations, stopwords, lemmatization, etc. Afterwards we tokenize the data using word_index which is fit on the train data. For CNN and RNN, we build a 2D data of (articles, words) while for HAN we build a 3D data of (articles, sentences, words) using the tokenizer. We set hyperparameters like dropout, embedding dimensions of glove model, trainable parameter of embedding layer, bidirectional lstm or simple lstm, etc. We then use hyperparameter tuning to find the best parameters by comparing the validation loss of each model. We build the model architecture corresponding to each of our model and the set hyperparameters. We will expand on the hyperparameter tuning later. After training using early stopping and checkpointing, we predict on the test set to find our test accuracies. We then compare test accuracies for each of our 3 models on both datasets to compare them.
![Solution Framework](https://github.com/rachit-shah/adbi-project/blob/master/Solution%20Framework.png)

## Model Architectures and Hyperparameters
### CNN
![CNN](https://github.com/rachit-shah/adbi-project/blob/master/Dataset%20-%201/Plots/cnn_model_arch.png)
### RNN
![RNN](https://github.com/rachit-shah/adbi-project/blob/master/Dataset%20-%201/Plots/rnn_model_arch.png)
### HAN
![HAN](https://github.com/rachit-shah/adbi-project/blob/master/Dataset%20-%201/Plots/han_model.png)

## Hyperparameter Tuning
As you can see in the above tables, we have done hyperparameter tuning for the following parameters:
1)	**Embedding Layer Trainable**  – Since we are using the Glove embedding layer which has pre-trained weights on a different dataset, we have the option to either freeze the pre-trained weights or allow the weights to be retrained in the training process. From the results of our tuning, we can see that for all of our models, retraining the weights resulted in a better validation loss (lower). 
2)	**Embedding Dimensions** – We had different dimensions of the Glove embedding model pre-trained on the Wikipedia corpus namely 50d,100d,200d and 300d. Higher dimensions will lengthen the training time but also provide a complex model. From our tuning, we found that only RNN had better results with 200d, while CNN and HAN performed better with 300d. 
3)	**Dropout** – We used a variety of different dropout rates like 0.2,0.3,0.4 and 0.5 to reduce overfitting of our trained model. From our tuning process, we found CNN needed 0.4 dropout while RNN and HAN needed 0.3 dropout for best results. 
4)	**Bidirectional LSTM or Unidirectional LSTM (for RNN)** – While unidirectional LSTM only considers the past input, bidirectional layer considers both the past and future inputs. While this may help uncover the context better, it will also make the model complex. From our results we found that unidirectional LSTM performed better on RNN while bidirectional performed better on RNN. 
5)	**MAX_SEQUENCE_LENGTH (for CNN and RNN)** – This parameter is used while generating the final input for CNN and RNN using tokenizer word_index  and embedding index. It is essentially the input dimension (i.e, the number of words it will consider for each article). It will pad the articles which has less words and cut the articles which has more words. We generated a boxplot to visualize the outliers and the 97th percentile of sequence length for all articles. This was around 500.
6)	**MAX_SENTENCES (for HAN)** – This parameter along with MAX_SENT_LENGTH determines the final input dimension for HAN model using tokenizer word_index and embedding index. Using a similar method to find the 97th percentile before, we found that the optimal number of sentences for each article was 50 for dataset 1 and 143 for dataset 2.
7)	**MAX_SENT_LENGTH (for HAN)** – Similar to MAX_SENTENCES, this parameter considers the maximum word length to take for each sentence in each article. This was found to be 50 for dataset 1 and 15 for dataset 2.

### CNN
| Embedding Trainable | Embedding Dim | Dropout | Validation Loss |
|---------------------|---------------|---------|-----------------|
| TRUE                | 100D          | 0.2     | 0.04462         |
| FALSE               | 100D          | 0.2     | 0.06694         |
| TRUE                | 300D          | 0.2     | 0.03009         |
| **TRUE**               | **300D**          | **0.4**     | **0.01323**         |
| TRUE                | 300D          | 0.5     | 0.02558         |
### RNN
| Embedding Trainable | Embedding Dim | Dropout | Bidirectional | Validation Loss |
|---------------------|---------------|---------|---------------|-----------------|
| TRUE                | 100D          | 0.2     | FALSE         | 0.10813         |
| FALSE               | 100D          | 0.2     | FALSE         | 0.1482          |
| TRUE                | 300D          | 0.2     | FALSE         | 0.13059         |
| TRUE                | 200D          | 0.2     | FALSE         | 0.05499         |
| TRUE                | 200D          | 0.4     | FALSE         | 0.05803         |
| TRUE                | 200D          | 0.2     | TRUE          | 0.06209         |
| TRUE                | 200D          | 0.4     | TRUE          | 0.0731          |
| **TRUE**                | **200D**        | **0.3**     | **FALSE**         | **0.04299**         |
### HAN
| Embedding Trainable | Embedding Dim | Dropout | Validation Loss |
|---------------------|---------------|---------|-----------------|
| TRUE                | 100D          | 0.2     | 0.03365         |
| FALSE               | 100D          | 0.2     | 0.06856         |
| TRUE                | 300D          | 0.2     | 0.02858         |
| TRUE                | 300D          | 0.4     | 0.03002         |
| **TRUE**                | **300D**          | **0.3**     | **0.02612**         |

## Results: Training and Validation Loss and Accuracy Plots
### CNN (Dataset1 on first row and Dataset 2 on second row)
![](https://github.com/rachit-shah/adbi-project/blob/master/Dataset%20-%201/Plots/CNN-Accuracy.png) ![](https://github.com/rachit-shah/adbi-project/blob/master/Dataset%20-%201/Plots/CNN-Loss.png) ![](https://github.com/rachit-shah/adbi-project/blob/master/Dataset%20-%202/Plots/CNN-d2-Accuracy.png) ![](https://github.com/rachit-shah/adbi-project/blob/master/Dataset%20-%202/Plots/CNN-d2-Loss.png)
### RNN (Dataset1 on first row and Dataset 2 on second row)
![](https://github.com/rachit-shah/adbi-project/blob/master/Dataset%20-%201/Plots/RNN-Accuracy.png) ![](https://github.com/rachit-shah/adbi-project/blob/master/Dataset%20-%201/Plots/RNN-Loss.png)![](https://github.com/rachit-shah/adbi-project/blob/master/Dataset%20-%201/Plots/RNN-Accuracy.png)![](https://github.com/rachit-shah/adbi-project/blob/master/Dataset%20-%202/Plots/RNN-d2-Loss.png)
### HAN (Dataset1 on first row and Dataset 2 on second row)
![](https://github.com/rachit-shah/adbi-project/blob/master/Dataset%20-%201/Plots/HAN-Accuracy.png)![](https://github.com/rachit-shah/adbi-project/blob/master/Dataset%20-%201/Plots/HAN-Loss.png)![](https://github.com/rachit-shah/adbi-project/blob/master/Dataset%20-%202/Plots/HAN-d2-Accuracy.png)![](https://github.com/rachit-shah/adbi-project/blob/master/Dataset%20-%202/Plots/HAN-d2-Loss.png)

## Final Accuracy on Test Data
### Dataset 1
| Model           | CNN     | RNN     | HAN     |
|-----------------|---------|---------|---------|
| **Validation Loss** | 0.01323 | 0.04299 | 0.02612 |
| **Test Accuracy**   | 96.63   | 95.73   | 97.07   |
### Dataset 2
| Model           | CNN     | RNN     | HAN     |
|-----------------|---------|---------|---------|
| **Validation Loss** | 0.48017 | 0.40511 | 0.36618 |
| **Test Accuracy**   | 79.82   | 82.63   | 83.4    |
| **Dropout**         | 0.5     | 0.3     | 0.3     |
| **Trainable**       | TRUE    | TRUE    | TRUE    |
| **Embedding DIM**   | 300D    | 200D    | 300D    |
| **MAX_SEQ_LEN**     | 1000    | 1000    | N/A     |
| **MAX_SENT_LEN**    | N/A     | N/A     | 15      |
| **MAX_SENT**        | N/A     | N/A     | 143     |
## Time Per Epoch
![](https://github.com/rachit-shah/adbi-project/blob/master/Dataset%20-%201/Plots/timeperepoch.png)![](https://github.com/rachit-shah/adbi-project/blob/master/Dataset%20-%202/Plots/timeperepoch2.png)
## Conclusion
From our experiment, we found that while CNN had the lowest validation loss on dataset 1, the test accuracy of HAN was highest even though it had lower validation loss compared to CNN. We can see that the difference between the test accuracies is very less. Hence, if someone needs to train a model faster, they could choose CNN over HAN. 

For dataset 2, we had considered a data with higher number of records and more classes. HAN performed the best on both validation loss and test accuracy while CNN performed the worst on both. This may have been because of larger dataset; HAN was able to retrieve a deeper understanding/context of the data. 

Overall, HAN performed consistently better for both types of datasets and it also took average time to train compared to CNN and RNN.

## References
1.	http://mlg.ucd.ie/datasets/bbc.html
2.	https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html
3.	https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
4.	https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
5.	https://medium.com/jatana/report-on-text-classification-using-cnn-rnn-han-f0e887214d5f
6.	http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
7.	https://machinelearningmastery.com/cnn-long-short-term-memory-networks/
8.	https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
9.	https://arxiv.org/pdf/1506.01057v2.pdf
10.	http://colah.github.io/posts/2015-08-Understanding-LSTMs/
11.	https://arxiv.org/abs/1408.5882
