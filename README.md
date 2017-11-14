# Graph-CNN-in-3D-Point-Cloud-Classification

One Paragraph of project description goes here

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```
### Installing instructions

1. Clone this repository.

```
git clone git@github.com:maggie0106/Graph-CNN-in-3D-Point-Cloud-Classification.git
```
2. Install the dependencies.

```
pip install -r requirements.txt
```
3. Download data       
We are using the data from 3D benchmark data set ModelNet http://modelnet.cs.princeton.edu/.     
The mesh polygon data format from ModelNet is preprocessed into Point Cloud format by  Charles R. Qi et al. (PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation, CVPR 2017 https://arxiv.org/abs/1612.00593)   
Download the data and put it in the data folder from the following link: 
```
https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip 
```
## Useful github repo for graph convolutional neural network and deep learning on point set related research
1. ChebyNet https://github.com/mdeff/cnn_graph 
Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst, Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering, Neural Information Processing Systems (NIPS 2016)
2. GCN https://github.com/tkipf/gcn    
Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
3. PointNet https://github.com/charlesq34/pointnet
Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas, PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation (CVPR 2017)


## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
