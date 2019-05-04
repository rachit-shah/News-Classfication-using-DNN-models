# Comparative Analysis of CNN, RNN and HAN for Text Classification with GloVe Data Model
Capstone Project as part of the course CSC591: Algorithms for Data-Guided Business Intelligence

Team Members:

- Rachit Shah (rshah25)

- Sourabh Sandanshi (ssandan)

## How to Run
You can run the IPython Notebooks in the folders "Dataset - 1" and "Dataset -2" by either downloading on your local machine or using the "Open in Colab" link to run the notebooks on Google Colab. To run the notebooks you will need the [dataset zip file](http://mlg.ucd.ie/datasets/bbc.html), the [glove embedding zip file](http://nlp.stanford.edu/data/glove.6B.zip) and the model weights too if you want to replicate our results. You can find all these files in the [Google Drive folder](https://drive.google.com/drive/folders/15A2b8uNEfak_1Gfh-c85dTtOHJrfrk7h?usp=sharing). Add all the files in this link to your google drive (root folder) in order to run the notebooks.

## Problem Description
We are using DNN models - CNN, RNN and HAN to classify News articles into different categories. The goal of our project is to predict/classify categories of news based on the content of news articles from the BBC website. We have used 2 datasets:
1. BBC News - http://mlg.ucd.ie/datasets/bbc.html (2225 news, 5 categories)
2. 20_newsgroup - https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html (18846 news posts, 20 categories)

There are six Ipython Notebooks (3 models for each dataset) in the submission. These can be found in the folder "Ipython Notebooks". They also have respective HTML files.

The dataset, GloVe Data model embeddings and final model weights are included in the following folder - "Files to be added to your drive" https://drive.google.com/drive/folders/15A2b8uNEfak_1Gfh-c85dTtOHJrfrk7h?usp=sharing
You have to add all these files to your drive in order to run the code. You select all and add at once. This is required to run the code on google colab.

Dataset can also be found on - http://mlg.ucd.ie/datasets/bbc.html

2nd Dataset was imported directly from scikit-learn's dataset library. It does not need to be uploaded/added

glove embeddings can also be found on - http://nlp.stanford.edu/data/glove.6B.zip

Next, open any of the Ipython Notebooks through google colab (you can use the open in colab link to do so)

Execute the cells as required. The colab environment has all the packages installed. The files also contain statements to add dependencies required.

You can find the video, report, presentation and contributions here - https://drive.google.com/drive/folders/1heicqQYsABXzKG3KLB8uVJ117qJ_jHkm?usp=sharing

We have also added a copy of weights, plots, code and architecture in separate folders for each dataset.
