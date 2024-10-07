# HMGCN
This repository provides a reference implementation of HMGCN as described in the paper:
> HMGCN: Heterogeneous Multilayer Graph Convolutional Network 
> 
> Hamed Kabiri Kenari
> 

Available at 

## Dependencies
Recent versions of the following packages for Python 3 are required:
* numpy==1.21.2
* torch==1.9.1
* scipy==1.7.1
* scikit-learn==0.24.2
* pandas==0.25.0

## Datasets
### Link
The used datasets are available at:
* Alibaba https://tianchi.aliyun.com/competition/entrance/231719/information/
* Amazon http://jmcauley.ucsd.edu/data/amazon/
* Aminer https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding/tree/master/Aminer
* IMDB https://github.com/seongjunyun/Graph_Transformer_Networks
* DBLP https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?dl=0

### Preprocess
We compress the data set into a mat format file, which includes the following contents.
* edges: array of subnetworks after coupling, each element in the array is a subnetwork.
* features: attributes of each node in the network.
* labels: label of labeled points.
* train: index of training set points for node classification. 
* valid: index of validation set points for node classification.
* test: index of test set points for node classification.

In addition, we also sample the positive and negative edges in the network, and divide them into three text files: train, valid and test for link prediction.


## Usage
This code in `Model.py` is for Alibaba dataset. For other dataset you should modify `Model.py`.

First, you need to determine the data set. If you want to do node classification tasks, you need to modify the data set path in `Node_classification.py`. If you want to do link prediction, you need to modify the dataset path in `Link_prediction.py`.

Second, you need to modify in `Model.py` according to each dataset. 


Execute the following command to run the node classification task:

* `python Node_Classification.py`

Execute the following command to run the link prediction task:

* `python Link_Prediction.py`

## Citing
If you find HMGCN useful in your research, please cite the following paper:

