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
The core of the project is `main.ipynb`, which allows to pre-process and analyse a collection of documents, from now on called *corpus*, using topic modeling techniques. The notebook is divided into 6 sections as follows:

1. Introduction
2. Pre-processing of the corpus
3. Exploratory analysis of the processed corpus
4. Estimation of the topic models:
	  1. Latent Dirichlet Allocation (LDA)
	  2. Dynamic Topic Model (DTM)
	  3. Embedded Topic Model (ETM)
	  4. Dynamic Embedded Topic Model (DETM)
5. Model comparison:
	  1. Quantitative analysis
	  2. Qualitative analysis
6. Conclusions

The repository also includes the following folders:
- *[docs](https://github.com/giovannitoto/pmda/tree/master/docs)* \
  It contains pdf files of the four articles in which the topic models are introduced.

- *[src](https://github.com/giovannitoto/pmda/tree/master/src)* \
  It contains an adaptation of the code that accompanies the papers titled "The Dynamic Embedded Topic Model" and "Topic Modeling in Embedding Spaces" by Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei. \
  First paper original GitHub repository: https://github.com/adjidieng/DETM. \
	Second paper original GitHub repository: https://github.com/adjidieng/ETM.

## Usage
Use `main.ipynb`!


## References

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993-1022. [ACM Digital Library](https://dl.acm.org/doi/10.5555/944919.944937)

- Blei, D. M., & Lafferty, J. D. (2006, June). Dynamic topic models. In Proceedings of the 23rd international conference on Machine learning (pp. 113-120). [ACM Digital Library](https://doi.org/10.1145/1143844.1143859)

- Dieng, A. B., Ruiz, F. J., & Blei, D. M. (2019). The dynamic embedded topic model. arXiv preprint arXiv:1907.05545. [Arxiv link](https://arxiv.org/abs/1907.05545)

- Dieng, A. B., Ruiz, F. J., & Blei, D. M. (2020). Topic modeling in embedding spaces. Transactions of the Association for Computational Linguistics, 8, 439-453. [ACM Anthology](https://aclanthology.org/2020.tacl-1.29/),  [Arxiv link](https://arxiv.org/abs/1907.04907)
