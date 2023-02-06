# Programming Metodologies for Data Analysis - Project

## Installation
First, we create an environment called *pmda* using *Anaconda Prompt (Anaconda3)*:
```
conda create -n pmda python=3.6.7
```
We need to activate the environment in order to install the required libraries:
```
conda activate pmda
conda install -c anaconda git
pip install ipykernel
ipython kernel install --user --name=pmda
```
**remark:** the last two commands allows us to create notebooks which use *pmda* environment instead of the base one.

Then, we install the libraries specified in `requirements.txt`:
```
pip install -r requirements.txt
```
Then, we install `pytorch` and `torchvision`:
```
conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch
```
Finally, we need to go back to the base environment in order to launch *jupyter notebook*:
```
conda deactivate
jupyter notebook
```
**remark:** When we activate the environment, we change from `<base>` to `<pmda>` in the *Anaconda Prompt (Anaconda3)*; when we deactivate it, we return to `<base>`.

## Contents
The core of the project is `main.ipynb`, which allows to pre-process and analyze a collection of documents, from now on called *corpus*, using topic modeling techniques. The notebook is divided into 8 sections as follows:

1. Introduction
2. Guardian dataset
	1. Download using API
	2. Selection of the variable of interest
	3. Pre-processing of the texts
3. Exploratory data analysis
4. Word embeddings
	1. Random embeddings
	2. Word2vec on train corpus
	3. FastText on train corpus
	4. Google’s Word2vec
	5. Stanford’s GloVe
5. Embedded Topic Model
6. Model comparison
	1. Quantitative analysis
	2. Qualitative analysis
7. Dynamic Embedded Topic Model
8. Conclusions
	1. Future work

The repository also includes the following folders:
* *[data](https://github.com/giovannitoto/pmda/tree/master/data)* \
  It contains files related to the corpus and its pre-processing.
* *[embeddings](https://github.com/giovannitoto/pmda/tree/master/embeddings)* \
  It contains files related to word embeddings.
* *[images](https://github.com/giovannitoto/pmda/tree/master/images)* \
  It contains images used in `main.ipynb`.
* *[results](https://github.com/giovannitoto/pmda/tree/master/results)* \
  It contains the estimated topic models.
* *[src](https://github.com/giovannitoto/pmda/tree/master/src)* \
  It contains the modules imported by `main.ipynb`; this is partially an adaptation of the code that accompanies the papers titled "Topic Modeling in Embedding Spaces" and "The Dynamic Embedded Topic Model" by Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei. \
  First paper original GitHub repository: https://github.com/adjidieng/ETM. \
	Second paper original GitHub repository: https://github.com/adjidieng/DETM.

## Usage
Use `main.ipynb`!

## Authors

* Lorenzo Dell'Oro - [lorenzodelloro](https://github.com/lorenzodelloro)
* Giovanni Toto - [giovannitoto](https://github.com/giovannitoto)
* Gian Luca Vriz - [GianVriz](https://github.com/GianVriz)

## References
* Airoldi, E. M., Blei, D., Erosheva, E. A., & Fienberg, S. E. (2014). Handbook of Mixed Membership Models and Their Applications. [ACM Digital Library](https://dl.acm.org/doi/book/10.5555/2765552)
* Blei, D. M., & Lafferty, J. D. (2006, June). Dynamic topic models. In Proceedings of the 23rd international conference on Machine learning (pp. 113-120). [ACM Digital Library](https://dl.acm.org/doi/10.1145/1143844.1143859)
* Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993-1022. [ACM Digital Library](https://dl.acm.org/doi/10.5555/944919.944937)
* Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with subword information. Transactions of the association for computational linguistics, 5, 135-146. [MIT Press Direct](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00051/43387/Enriching-Word-Vectors-with-Subword-Information)
* Dieng, A. B., Ruiz, F. J., & Blei, D. M. (2019). The dynamic embedded topic model. arXiv preprint arXiv:[1907.05545](https://arxiv.org/abs/1907.05545).
* Dieng, A. B., Ruiz, F. J., & Blei, D. M. (2020). Topic modeling in embedding spaces. Transactions of the Association for Computational Linguistics, 8, 439-453. [ACM Anthology](https://aclanthology.org/2020.tacl-1.29/),  [Arxiv link](https://arxiv.org/abs/1907.04907)
* Gupta, P., & Jaggi, M. (2021). Obtaining better static word embeddings using contextual embedding models. arXiv preprint arXiv:[2106.04302](https://arxiv.org/abs/2106.04302).
* Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013a). Efficient estimation of word representations in vector space. arXiv preprint arXiv:[1301.3781](https://arxiv.org/abs/1301.3781).
* Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013b). Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems, 26. [ACM Digital Library](https://dl.acm.org/doi/abs/10.5555/2999792.2999959)
* Pennington, J., Socher, R., & Manning, C. D. (2014, October). Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) (pp. 1532-1543). [ACL Anthology](https://aclanthology.org/D14-1162/)
