# SliceCast
Podcasts transcripts are long form documents of conversational nature which have not yet been studied as topical text segmentation problems. This repository explores a neural network approach to segment podcasts based on topic of discussion.  We model the problem as a binary classification task where each sentence is either labeled as the first sentence of a new segment or a continuation of the current segment. We embed sentences using the Universal Sentence Encoder and use an LSTM-based classification network to obtain the cutoff probabilities. Our results indicate that neural network models are indeed suitable for topical segmentation on long, conversational texts, but larger datasets are needed for a truly viable product.

# Demo
![Results](imgs/labeledWiki.png)

# Table of Contents
[**Demo**](#Demo)

[**Data**](#Data)
  * [**Wiki-727k Dataset**](#Wiki-727k)
  * [**Small-scale podcast dataset**](#Small-scale-podcast-dataset)

[**Reproducing Results**](#Reproducing-results)

[**Built With**](#Built-With)

[**Authors**](#Authors)

[**License**](#License)

[**Acknowledgments**](#Acknowledgments)

# Repository Structure
The three Jupyter notebooks at the root directory of this repository perform the following functions:
* EDA.ipynb - Provides preliminary exploratory data analysis on the Wiki and Podcast datasets
* train.ipynb - Contains runnable code for training the neural network model
* inference.ipynb - Contains code to evaluate and visualize predictions from the model

The repository has the following structure:
```bash
├── data
│   ├── podcasts/
│   └── wiki-sample/
├── imgs
├── models
├── src
│   ├── SliceNet.py
│   ├── netUtils.py
│   ├── postprocess.py
│   ├── preprocess.py
│   ├── runPreprocess.py
│   └── spacyOps.py
├── EDA.ipynb
├── train.ipynb
├── inference.ipynb
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```
# Data
Our data is separated into 2 main datasets: a large scale training set of Wikipedia articles, Wiki-727k, and a novel dataset of hand-labeled podcast transcriptions.

## Wiki-727k
The Wiki-727k dataset is the primary source of training data. It was introduced by Koshorek et al in the paper "Text Segmentation as a Supervised Learning Task". This dataset consists of text from roughly 727,000 Wikipedia articles. Due to rigid structure of Wikipedia articles, each article can be easily segmented using the table of contents. This dataset was chosen due to the consistent labeling and volume of data.

Data Structure:

* train set: 582160 text files
* test set: 73232 text files
* dev set: 72354 text files

## Small-scale podcast dataset
This dataset consists of podcast transcripts from three different shows, obtained either directly from the authors or using Google Speech to Text. For segmentation labeling, we use the data provided by the authors or by listeners who have provided their own segmentations.

# Reproducing Results
Following these instructions will get you a copy of the project up and running on a Google Cloud Compute instance to train and test the models provided in this repository.

Running the code in this repository requires elementary knowledge of both Jupyter and Pip. Note that python version 3.6.7 was used for this project. 
It is recommended that this code be run on a GPU enabled cloud instance for quick inference times. Additionally, training is only possible with a GPU enabled instance. The simplest method is to deploy Google's preconfigured [Deep Learning VM](https://cloud.google.com/deep-learning-vm/) which comes preinstalled with important software including CUDA 10.
Once the instance has been created, clone this repository to your machine using the command
```
git clone https://github.com/bmmidei/SliceCast.git
```
Within the top level directory, you will find a 'requirements.txt' file, which includes a comprehensive list of dependencies necessary to execute the functionality of this repository. With your new environment active, use the following command to install these dependencies:
```
pip3 install -r requirements.txt
pip3 install git+https://github.com/boudinfl/pke.git
python3 -m spacy download en_core_web_sm
```
Next, navigate to data directory and extract the tar file. This will provide you with a small sample of the Wiki dataset for exploratory data analysis and inference:
```
tar -xvf wiki-sample.tar.gz
```
If you would like access to the full Wiki-727k training set, this must be downloaded from the source found [here](https://github.com/koomri/text-segmentation).

After following these steps, you should be able to run each of the three jupyter notebooks. If you have any trouble, feel free to submit an issue and we'll be happy to help. 

# Built With
* [TensorFlow](https://www.tensorflow.org) - The Neural Network backend used
* [Keras](https://keras.io) - The Neural Network high-level API
* [NumPy](http://www.numpy.org/) - Matrix operations and linear algebra
* [SpaCy](https://spacy.io) - Natural language process for sentence tokenization

# Authors

* **Brian Midei** - [bmmidei](https://github.com/bmmidei)
* **Marko Mandic** - [markomandic](https://github.com/markomandic)

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

# Acknowledgments

* Omri Koshorek, Adir Cohen, Noam Mor, Michael Rotman, Jonathan Berant for proposing text segmentation as a supervised
learning task and for providing the Wiki-727k dataset
    * Link to original paper - https://arxiv.org/abs/1803.09337
