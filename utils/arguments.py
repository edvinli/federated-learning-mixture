import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="global epochs")
    parser.add_argument('--num_clients', type=int, default=100, help="number of clients")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--n_data', type=float, default=500, help="datasize on each client")
    parser.add_argument('--train_frac', type=float, default=0.1, help="fraction of training data size")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='which model to use')
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
                        
    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--runs', type=int, default=1, help='number of runs to do experiment')
    parser.add_argument('--opt', type=float, default=0.5, help='fraction of clients that opt-in (default: 0.5)')
    parser.add_argument('--p', type=float, default = 0.3, help='majority class percentage (default: 0.3)')
    parser.add_argument('--train_gate_only', action='store_true', help='whether to train gate only or not')
    args = parser.parse_args()
    return args
