import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pdb
import argparse
import sys
import os
torch.manual_seed(0)
np.random.seed(0)


parser = argparse.ArgumentParser(description='Hypter Parameters')
parser.add_argument('-D', required=True, type=int, metavar='d1', help='input1 dimension')
parser.add_argument('-N', required=True, type=int, metavar='d2', help='input2 dimension')
parser.add_argument('-H', required=True, type=int, metavar='h', help='hidden 1 dimension')
parser.add_argument('-ds', required=True, type=int, metavar='DS', help='dataset size')
parser.add_argument('-bs', required=True, type=int, metavar='BS', help='mini-batch size')
parser.add_argument('-e', required=True, type=int, metavar='epoch', help='number of epochs')
parser.add_argument('-mv', default='100', type=int, metavar='w', help='moving average window size')
parser.add_argument('-lr', required=True, type=float, metavar='LR', help='learning rate')
parser.add_argument('-net', required=True, type=str, help='Network Type')
parser.add_argument('-n', default='Extensive_Experiment', type=str, help='experiment name')

class NetA(nn.Module):
    def __init__(self, input_size1=32, input_size2=32, hidden_size1=100):
        super(NetA,self).__init__()
        self.ma_et = 1
        self.fc1 = nn.Linear(input_size1 + input_size2, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, 1)
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x, y):
        input = torch.cat((x,y), 1)
        output = F.relu(self.fc1(input))
        output = self.fc2(output)
        return output  


class NetB(nn.Module):
    def __init__(self, input_size1=32, input_size2=32, hidden_size1=100, hidden_size2=100):
        super(NetB,self).__init__()
        self.ma_et = 1
        self.fc1 = nn.Linear(input_size1 + input_size2, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant(self.fc3.bias, 0)

    def forward(self, x, y):
        input = torch.cat((x,y), 1)
        output = F.relu(self.fc1(input))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output  


class NetC(nn.Module):
    def __init__(self, input_size1=32, input_size2=32, hidden_size1=100, hidden_size2=100, hidden_size3=100):
        super(NetC,self).__init__()
        self.ma_et = 1
        self.fc1 = nn.Linear(input_size1 + input_size2, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, 1)
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant(self.fc3.bias, 0)
        nn.init.normal_(self.fc4.weight,std=0.02)
        nn.init.constant(self.fc4.bias, 0)

    def forward(self, x, y):
        input = torch.cat((x,y), 1)
        output = F.relu(self.fc1(input))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        output = self.fc4(output)
        return output  


class NetD(nn.Module):
    def __init__(self, input_size1=32, input_size2=32, hidden_size1=100, hidden_size2=100):
        super(NetD,self).__init__()
        self.ma_et = 1
        self.fc1 = nn.Linear(input_size1,hidden_size1)
        self.fc2 = nn.Linear(input_size2,hidden_size1)
        self.fc3 = nn.Linear(2 * hidden_size1, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, 1)
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant(self.fc3.bias, 0)
        nn.init.normal_(self.fc4.weight,std=0.02)
        nn.init.constant(self.fc4.bias, 0)

    def forward(self, x, y):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(y))
        output = torch.cat((out1,out2), 1)
        output = F.relu(self.fc3(output))
        output = self.fc4(output)
        return output 

class NetE(nn.Module):
    def __init__(self, input_size1=32, input_size2=32, hidden_size1=100, hidden_size2=100, hidden_size3=100):
        super(NetE,self).__init__()
        self.ma_et = 1
        self.fc1 = nn.Linear(input_size1,hidden_size1)
        self.fc2 = nn.Linear(input_size2,hidden_size1)
        self.fc3 = nn.Linear(2 * hidden_size1, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, hidden_size3)
        self.fc5 = nn.Linear(hidden_size3, 1)
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant(self.fc3.bias, 0)
        nn.init.normal_(self.fc4.weight,std=0.02)
        nn.init.constant(self.fc4.bias, 0)
        nn.init.normal_(self.fc5.weight,std=0.02)
        nn.init.constant(self.fc5.bias, 0)

    def forward(self, x, y):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(y))
        output = torch.cat((out1,out2), 1)
        output = F.relu(self.fc3(output))
        output = F.relu(self.fc4(output))
        output = self.fc5(output)
        return output 

class NetF(nn.Module):
    def __init__(self, input_size1=32, input_size2=32, hidden_size1=100, hidden_size2=100):
        super(NetF,self).__init__()
        self.ma_et = 1
        self.fc1 = nn.Linear(input_size1,hidden_size1)
        self.fc2 = nn.Linear(input_size2,hidden_size1)
        self.fc3 = nn.Linear(hidden_size1, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, 1)
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant(self.fc3.bias, 0)
        nn.init.normal_(self.fc4.weight,std=0.02)
        nn.init.constant(self.fc4.bias, 0)

    def forward(self, x, y):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(y))
        output = out1 + out2
        output = F.relu(self.fc3(output))
        output = self.fc4(output)
        return output 

class NetG(nn.Module):
    def __init__(self, input_size1=32, input_size2=32, hidden_size1=100, hidden_size2=100, hidden_size3=100):
        super(NetG,self).__init__()
        self.ma_et = 1
        self.fc1 = nn.Linear(input_size1,hidden_size1)
        self.fc2 = nn.Linear(input_size2,hidden_size1)
        self.fc3 = nn.Linear(hidden_size1, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, hidden_size3)
        self.fc5 = nn.Linear(hidden_size3, 1)
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant(self.fc3.bias, 0)
        nn.init.normal_(self.fc4.weight,std=0.02)
        nn.init.constant(self.fc4.bias, 0)
        nn.init.normal_(self.fc5.weight,std=0.02)
        nn.init.constant(self.fc5.bias, 0)

    def forward(self, x, y):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(y))
        output = out1 + out2
        output = F.relu(self.fc3(output))
        output = F.relu(self.fc4(output))
        output = self.fc5(output)
        return output


def get_cov(D, N):
    dummy_data = np.random.rand(D + N, 2 * (D + N))
    return np.cov(dummy_data, bias=True)

def get_Net(type, D, N, H):
    if type == 'A':
        return NetA(input_size1=D, input_size2=N, hidden_size1=H)
    elif type == 'B':
        return NetB(input_size1=D, input_size2=N, hidden_size1=H, hidden_size2=H)
    elif type == 'C':
        return NetC(input_size1=D, input_size2=N, hidden_size1=H, hidden_size2=H, hidden_size3=H)
    elif type == 'D':
        return NetD(input_size1=D, input_size2=N, hidden_size1=H, hidden_size2=H)
    elif type == 'E':
        return NetE(input_size1=D, input_size2=N, hidden_size1=H, hidden_size2=H, hidden_size3=H)
    elif type == 'F':
        return NetD(input_size1=D, input_size2=N, hidden_size1=H, hidden_size2=H)
    elif type == 'G':
        return NetE(input_size1=D, input_size2=N, hidden_size1=H, hidden_size2=H, hidden_size3=H)
    else:
        raise Exception('The network type is not valid! Please enter a valid type!')

def ma(a, window_size=100):
    return np.array([np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)])


def gen_x_y(joint, bs, D, N):
    index = np.random.choice(range(joint.shape[0]), size=bs, replace=False)
    # joint = np.random.multivariate_normal(mean=np.zeros(D + N), cov=C, size = bs
    x = joint[index][:,0:D].reshape(-1, D)
    y = joint[index][:,D:].reshape(-1, N)
    return x, y


def pass_input(net, train_set, bs, D, N):
    x_sample, y_sample = gen_x_y(train_set, bs, D, N)
    y_shuffle=np.random.permutation(y_sample)
    x_sample = torch.from_numpy(x_sample).type(torch.cuda.FloatTensor)
    y_sample = torch.from_numpy(y_sample).type(torch.cuda.FloatTensor)
    y_shuffle = torch.from_numpy(y_shuffle).type(torch.cuda.FloatTensor)
    pred_xy = net(x_sample, y_sample)
    pred_x_y = net(x_sample, y_shuffle)
    return pred_xy, pred_x_y

def train(n_epoch, model, train_set, bs, optimizer, D, N):
    MI_MINE = []
    with tqdm(total=n_epoch, ncols=0, file=sys.stdout, desc='Training ...') as pbar:
        for epoch in range(n_epoch):
            pred_xy, pred_x_y = pass_input(model, train_set, bs, D, N)
            B = pred_x_y.shape[0]
            et = torch.exp(pred_x_y)
            mi_lb = torch.mean(pred_xy) - torch.log(torch.mean(et))
            model.ma_et = (1 - 0.01) * model.ma_et + 0.01 * torch.mean(et)
            loss = -(torch.mean(pred_xy) - (1/model.ma_et.mean()).detach()*torch.mean(et))
            MI_MINE.append(mi_lb.data.cpu().numpy())
            loss.backward()

            # TODO: test
            if (epoch + 1) % 10:
                optimizer.step()
                optimizer.zero_grad()
            
            pbar.set_postfix(MI=mi_lb.item())
            pbar.update()
    return MI_MINE



def save_figure(exp_name, MI_MINE, mi_trad):
    plt.figure(figsize=[8, 5])
    MI_MINE = np.asarray(MI_MINE)
    MI_MINE[MI_MINE < 0] = 0
    MI_MINE_ma = ma(MI_MINE)
    estimated = MI_MINE_ma[-1].round(4)
    print('estimated mi = ' + str(estimated))
    print("true mi = " + str(mi_trad))
    plot_x = np.arange(len(MI_MINE_ma))
    plt.plot(plot_x, MI_MINE_ma, label='MINE')
    plt.plot(plot_x, [mi_trad] * plot_x.size, label='True MI', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Mutual Information')
    plt.legend()
    plt.savefig('./' + exp_name + '/' + exp_name + '.png', dpi=500)





def main():

    # Parse Arguments
    args = parser.parse_args()
    print(args)
    N = args.N
    D = args.D
    H = args.H
    bs = args.bs
    n_epoch = args.e
    mv = args.mv
    lr = args.lr
    name = args.n
    ds = args.ds
    batch_size = bs
    net = args.net

    try:
        # make the directory and check if the experiment has been done previously
        exp_name = 'NetworkType_{}_InputSize1_{}_InputSize2_{}_BatchSize_{}__HiddenSize_{}_LR_{}_Epochs_{}_TrainSetSize_{}'.format(net, D, N, bs, H, lr, n_epoch, ds)
        os.mkdir('./' + exp_name)

        
        # Define a new covariance matrix from a random data
        C = get_cov(D, N)
        train_set = data = np.random.multivariate_normal(mean=np.zeros(D + N), cov=C, size = ds)
        
        # Computing the traditional mutual information
        mi_trad = 0.5 *( np.log(np.linalg.det(C[0:D, 0:D])) + np.log(np.linalg.det(C[D:, D:])) - np.log(np.linalg.det(C)))

        # Define the Network and Optimizer
        model = get_Net(net, D, N, H)
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.zero_grad()
        eps = 10e-20 # is it necessary?

        # Start Training
        MI_MINE = train(n_epoch, model, train_set, bs, optimizer, D, N)
        
        # save true MI and MINE MI. Save figure 

        np.save('./' + exp_name + '/' + exp_name + 'MI_MINE.npy', np.concatenate(([mi_trad], MI_MINE)))
        np.save('./' + exp_name + '/' + exp_name + 'MI_MINE_ma.npy', np.concatenate(([mi_trad], ma(MI_MINE))))
        save_figure(exp_name, MI_MINE, mi_trad)
    
    except:
        print('This experiment has been already done!')
    





if __name__ == '__main__':
    main()
