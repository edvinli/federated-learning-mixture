# federated-learning-mixture
This repo contains code for the paper [Federated learning using a mixture of experts](https://arxiv.org/abs/2010.02056).

# Example
To run the code on cifar10 with 5 clients, run the following line. Results will be saved in /save/results.

`python main_fed.py --model 'cnn' --dataset 'cifar10' --n_data 500 --num_clients 5 --epochs 45 --train_frac 0.2 --local_ep 3 --opt 0 --p 1.0 --gpu 0 --runs 1`
# Results
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
