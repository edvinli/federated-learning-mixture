# Federated learning using a mixture of experts
This repo contains code for the paper [Specialized federated learning using a mixture of experts](https://arxiv.org/abs/2010.02056).

# Example
To run an experiment on the CIFAR-10 dataset, use the following line.

`python main_fed.py --model 'cnn' --dataset 'cifar10' --num_classes 10 --n_data 100 --n_data_val 200 --n_data_test 200 --frac 5 --num_clients 100 --epochs 1250 --local_ep 3 --opt 0 --p 1.0 --lr 1e-5 --alpha 0 --gpu 0 --runs 1 --overlap`

To run an experiment on the AG News dataset, use the following line

`python main_fed_ag.py --dataset 'agnews' --num_classes 4 --n_data 100 --n_data_val 100 --n_data_test 200 --frac 50 --num_clients 1000 --epochs 1250 --local_ep 3 --opt 0 --p 1.0 --lr 1e-6 --alpha 0 --gpu 0 --overlap --runs 1`

The results will be saved in 'save/results' for CIFAR-10 and Fashion-MNIST and in 'save/results_ag' for AG News.

# Results
Results for CIFAR-10 running above example, for p in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].  Average accuracy over three runs per value of majority class fraction.

![cifar-10](https://github.com/edvinli/federated-learning-mixture/blob/main/figures/cifar10_opt0.png)

# Cite
If you find this work useful, please cite us using the following bibtex:
```bibtex
@article{listozec2020federated,
  title={Specialized federated learning using a mixture of experts},
  author={Listo Zec, Edvin and Mogren, Olof and Martinsson, John and S{\"u}tfeld, Leon Ren{\'e} and Gillblad, Daniel},
  journal={arXiv preprint arXiv:2010.02056},
  year={2020}
}

```

# Acknowledgements
The code developed in this repo was was adapted from https://github.com/shaoxiongji/federated-learning.
