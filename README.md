# federated-learning-mixture
This repo contains code for the paper [Federated learning using a mixture of experts](https://arxiv.org/abs/2010.02056).

# Example
To run the code on cifar100 with 50 clients, run the following line. Results will be saved in /save/results.

`python main_fed.py --model 'cnn' --dataset 'cifar100' --n_data 500 --num_clients 50 --num_classes 100 --epochs 45 --train_frac 0.2 --local_ep 3 --opt 0 --p 1.0 --gpu 0 --runs 1`
# Results
Results for running above example, for p in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].  Average accuracy over three runs per value of majority class fraction.

![cifar-100](https://github.com/edvinli/federated-learning-mixture/blob/main/figures/c_100(1).png)

# Cite
If you find this work useful, please cite us using the following bibtex:
```bibtex
@article{listozec2020federated,
  title={Federated learning using a mixture of experts},
  author={Listo Zec, Edvin and Mogren, Olof and Martinsson, John and S{\"u}tfeld, Leon Ren{\'e} and Gillblad, Daniel},
  journal={arXiv preprint arXiv:2010.02056},
  year={2020}
}

```

# Acknowledgements
The code developed in this repo was was adapted from https://github.com/shaoxiongji/federated-learning.
