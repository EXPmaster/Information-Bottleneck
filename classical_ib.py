import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def is_positive(a):
    return (a >= 0).all()


def relative_entropy(p_Y_X, q_Y_T):
    assert is_positive(q_Y_T), 'q is not positive'
    # p_Y_X [dim_Y, dim_X]
    # q_Y_T [dim_Y, dim_T]
    p = p_Y_X[:, None, :]
    q = q_Y_T[:, :, None]
    return (p * (torch.log2(p) - torch.log2(q))).sum(0)


def Z_x_beta(pX, beta, lambdaX, p_Y_X, pY):
    """ Deprecated. """
    return torch.exp(-beta * (p_Y_X * (torch.log2(p_Y_X) - torch.log2(pY)[:, None])).sum(0))


def initialize_q_T_X(args):
    # return torch.softmax(torch.randn(args.dim_T, args.dim_X), 0).to(device)
    mat1 = torch.eye(args.dim_T) * 0.75
    mat2 = torch.rand(args.dim_T, args.dim_X).fill_diagonal_(0.)
    mat2 = mat2 / mat2.sum(0) * 0.25
    return (mat1 + mat2).to(device)


def classical_original_ib(args, pX, p_Y_X):
    """
    Classical original Information Bottleneck algorithm

    Args:
        args: argument in parser
        pX: probability distribution of input X
        p_Y_X: p(Y|X) of dims: [dim_Y, dim_X]
    
    Returns:
        mutural information I(T;X), I(Y;T)
    """
    # row: Y, col: X
    pXY = p_Y_X * pX
    pY = pXY.sum(1)
    q_T_X = initialize_q_T_X(args)

    q_T = q_T_X @ pX
    q_Y_T = pXY @ q_T_X.T / q_T
    z_x_b = Z_x_beta(pX, args.beta, 1., p_Y_X, pY)

    previous_loss = float('inf')
    converged = False
    converge_count = 0
    n = 0
    while not converged and n < 3000:
        previous_q_T_X = q_T_X.clone()
        q_T_X = q_T[:, None] * torch.exp(-args.beta * relative_entropy(p_Y_X, q_Y_T))
        q_T_X /= q_T_X.sum(0)
        q_T = q_T_X @ pX
        q_Y_T = pXY @ q_T_X.T / q_T
        loss = torch.dist(previous_q_T_X, q_T_X)
        # print('loss:', loss.item())
        args.writer.add_scalar("obj/J_fixed_gamma", loss.real, n)
        if abs(loss - previous_loss) < args.cvg_thres:
            converge_count += 1
        else:
            converge_count = 0
        if converge_count == 10:
            converged = True
        previous_loss = loss
        n += 1
    # print(q_T_X[:, 0])
        q_TX = q_T_X * pX[None, :]
        I_TX = (q_TX * (torch.log2(q_TX + 1e-12) - torch.log2(pX[None, :] * q_T[:, None]))).sum()
        q_YT = q_Y_T * q_T[None, :]
        I_YT = (q_YT * (torch.log2(q_YT + 1e-12) - torch.log2(q_T[None, :] * pY[:, None]))).sum()
        f_alpha = I_TX - args.beta * I_YT
        args.writer.add_scalar("obj/f_alpha", f_alpha, n)
    # print(f'Algorithm converged in {n} iterations.')
    return I_TX, I_YT, n


def gen_p_Y_X(args):
    """ Generate p_{Y|X} """
    alphas = torch.logspace(-1.3, 1.3, args.dim_X)
    p_Y_X = []
    for alpha in alphas:
        distr_y = torch.distributions.dirichlet.Dirichlet(torch.ones(args.dim_Y, dtype=torch.float) * alpha)
        p_Y_X.append(distr_y.sample())
    p_Y_X = torch.stack(p_Y_X).T
    return p_Y_X.to(device)


@torch.no_grad()
def main(args):
    # pX = torch.ones(args.dim_X) / args.dim_X
    distr_x = torch.distributions.dirichlet.Dirichlet(torch.ones(args.dim_X, dtype=torch.float) * 1000)
    pX = distr_x.sample().to(device)
    p_Y_X = gen_p_Y_X(args).to(device)
    x_coords = []
    y_coords = []
    betas = []
    iterations = []
    # for beta in tqdm(torch.arange(0.001, 50, 0.1)):
    #     args.beta = beta
    #     betas.append(beta)
    #     I_TX, I_YT, n = classical_original_ib(args, pX, p_Y_X)
    #     iterations.append(n)
    #     x_coords.append(I_TX.cpu())
    #     y_coords.append(I_YT.cpu())

    args.beta = 6.
    I_TX, I_YT, n = classical_original_ib(args, pX, p_Y_X)

    assert False
    fig = plt.figure()
    plt.scatter(x_coords, y_coords)
    plt.xlabel('I(X;T)')
    plt.ylabel('I(T;Y)')
    plt.savefig('ib_plane.png')

    fig = plt.figure()
    plt.plot(betas, iterations)
    plt.xlabel('beta')
    plt.ylabel('Iters')
    plt.savefig('converge_itrs.png')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', default=0.2, type=float)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=20., type=float)
    parser.add_argument('--dim_X', default=2**8, type=int, help='dimension of X')
    parser.add_argument('--dim_Y', default=2**6, type=int, help='dimension of Y')
    parser.add_argument('--dim_T', default=2**8, type=int, help='dimension of T')
    parser.add_argument('--cvg_thres', default=1e-3, type=int, help='threshold for convergence of algorithm')
    args = parser.parse_args()

    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    print(f'==== Running classical algorithm on {device}....')

    # set_seed(0)
    args.writer = SummaryWriter(log_dir='./logs/classical-gamma-0.2-8')
    main(args)
    args.writer.flush()
    args.writer.close()
    