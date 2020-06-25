# matplotlib inline
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

parser = argparse.ArgumentParser(description='Hypter Parameters')

parser.add_argument('-N', default=32, type=int, metavar='N', help='input1 dimension')
parser.add_argument('-D', default=32, type=int, metavar='N', help='input2 dimension')
parser.add_argument('-H1', default=256, type=int, metavar='N', help='hidden 1 dimension')
parser.add_argument('-H2', default=256, type=int, metavar='N', help='hidden 2 dimension')
parser.add_argument('-ds', default=50000, type=int, metavar='N', help='dataset size')
parser.add_argument('-bs', default=4000, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-e', default=5000, type=int, metavar='N', help='number of epochs')
parser.add_argument('-lr', default=0.01, type=float, metavar='N', help='learning rate')
parser.add_argument('-c', default=0.9, type=float, metavar='N', help='correlation coefficient')
parser.add_argument('-n', default='multi', type=str, help='experiment name')

args = parser.parse_args()
print(args)
# To DO : fusion ideas concat-adding-....
# torch.manual_seed(0)
# np.random.seed(0)



N = args.N
D = args.D
H1 = args.H1
H2 = args.H2
bs = args.bs
n_epoch = args.e
lr = args.lr
name = args.n
rho = args.c
ds = args.ds

batch_size = bs


# W = np.random.randn(N,D)
W = np.identity(N)
# W = 2 * np.identity(N)


mu = np.zeros(N)
cov = np.identity(N)
mv_gaussian = multivariate_normal(mu, cov)



data = np.random.rand(D + N, 2 * (D + N))
C = np.cov(data,bias=True)


data = np.random.multivariate_normal(mean=np.zeros(D + N), cov=C, size = ds)

mi_trad = 0.5 *( np.log(np.linalg.det(C[0:D, 0:D])) + np.log(np.linalg.det(C[D:, D:])) - np.log(np.linalg.det(C)))
print("true mi = " + str(mi_trad))


# data = np.random.multivariate_normal(mean=[0,0], cov=[[1,rho],[rho, 1]], size = 300)

def func(x):
    return x

def gen_x():
    # return np.random.normal(0.,1.,[batch_size,N])
    return mv_gaussian.rvs(batch_size)

def gen_y(x):
    if D == 1:
        return W[0] * x
    return np.matmul(x, W)

def gen_x_y(joint):
    index = np.random.choice(range(joint.shape[0]), size=batch_size, replace=False)
    x = joint[index][:,0:D]
    y = joint[index][:,D:]

    # joint = np.random.multivariate_normal(mean=[0,0], cov=[[1,rho],[rho, 1]], size = batch_size)
    # return joint[:,0], joint[:,1]

    return x, y
    




import sys
import numpy as np
import pdb
# np.random.seed(0)
import torch
# torch.manual_seed(0)


class Net(nn.Module):
    def __init__(self, input_size=D + N, hidden_size1=H1, hidden_size2=H2):
        super(Net,self).__init__()
        self.ma_et = 1
        self.fc1 = nn.Linear(input_size,hidden_size1 )
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant(self.fc3.bias, 0)

    def forward(self, x, y):
        # TO DO : Mormalization

        if D==1:
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
        input = torch.cat((x,y), 1)
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output   

def pass_input(net):
    x_sample, y_sample = gen_x_y(data)
    # x_sample1=gen_x()
    # y_sample1=gen_y(x_sample)
    # pdb.set_trace()
    y_shuffle=np.random.permutation(y_sample)
    x_sample = torch.from_numpy(x_sample).type(torch.cuda.FloatTensor)
    y_sample = torch.from_numpy(y_sample).type(torch.cuda.FloatTensor)
    y_shuffle = torch.from_numpy(y_shuffle).type(torch.cuda.FloatTensor)
    pred_xy = net(x_sample, y_sample)
    pred_x_y = net(x_sample, y_shuffle)
    return pred_xy, pred_x_y

model = Net()
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer.zero_grad()
eps = 10e-20
MI_MINE = []
with tqdm(total=n_epoch, ncols=0, file=sys.stdout, desc='Training ...') as pbar:
    for epoch in range(n_epoch):
        
        pred_xy, pred_x_y = pass_input(model)
        B = pred_x_y.shape[0]


        # -------------------------- Added By Hossein -----------------------------
        et = torch.exp(pred_x_y)
        mi_lb = torch.mean(pred_xy) - torch.log(torch.mean(et))
        model.ma_et = (1 - 0.01) * model.ma_et + 0.01 * torch.mean(et)
        loss = -(torch.mean(pred_xy) - (1/model.ma_et.mean()).detach()*torch.mean(et))
        MI_MINE.append(mi_lb.data.cpu().numpy())
        # ---------------------------------------------------------------------
        

    
        # -------------------------- Added By Pirazh -----------------------------
        # mi_lb = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        # et1 = torch.mean(torch.exp(pred_x_y[:int(B / 2)]))
        # if model.ma_et is None:
        #     model.ma_et = et1.detach().item()
        # model.ma_et += 0.01 * (et1.detach().item() - model.ma_et)
        # ret1 = torch.mean(pred_xy[:int(B / 2)]) - torch.log(et1+eps) * et1.detach() / model.ma_et
        # et2 = torch.mean(torch.exp(pred_x_y[int(B / 2):]))
        # if model.ma_et is None:
        #     model.ma_et = et2.detach().item()
        # model.ma_et += 0.01 * (et2.detach().item() - model.ma_et)
        # ret2 = torch.mean(pred_xy[int(B / 2):]) - torch.log(et2+eps) * et2.detach() / model.ma_et
        # # ---------------------------------------------------------------------
        # smoothness_loss = 10 * (ret1 - ret2).abs()
        # # ret1 = torch.mean(pred_xy[:int(B / 2)]) - torch.log(torch.mean(torch.exp(pred_x_y[:int(B / 2)])))
        # # ret2 = torch.mean(pred_xy[int(B / 2):]) - torch.log(torch.mean(torch.exp(pred_x_y[int(B / 2):])))        
        # MI = (ret1 + ret2) / 2
        # loss = - MI 
        # # loss = -(torch.mean(pred_xy[int(B / 2):])- et1.detach() / model.ma_et + torch.mean(pred_xy[int(B / 2):])- et2.detach() / model.ma_et)/2
        # MI_MINE.append(mi_lb.data.cpu().numpy())
        # ---------------------------------------------------------------------



        
        loss.backward()

        # TO DO: test
        if (epoch + 1) % 10:
            optimizer.step()
            optimizer.zero_grad()
        
        pbar.set_postfix(MI=mi_lb.item())
        pbar.update()




# mi_trad= D * 0.5 * np.log(2 * np.pi * np.exp(1)) + 0.5 * np.log(np.linalg.det(np.matmul(W.T, W)))

# mi_trad= D * 0.5 * np.log(2 * np.pi * np.exp(1)) + 0.5 * np.log(np.linalg.det(np.matmul(W.T, W)))

# mi_trad = - 0.5 * np.log(1 - rho ** 2)

# mi_trad = 0.5 *( np.log(np.linalg.det(C[0:2, 0:2])) + np.log(np.linalg.det(C[2:, 2:])) - np.log(np.linalg.det(C)))



def ma(a, window_size=100):
    return np.array([np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)])




plt.figure(figsize=[8, 5])
MI_MINE = np.asarray(MI_MINE)
MI_MINE[MI_MINE < 0] = 0
MI_MINE_ma = ma(MI_MINE)
estimated = MI_MINE_ma[-1].round(4)
print('estimated mi = ' + str(estimated))
plot_x = np.arange(len(MI_MINE_ma))
plt.plot(plot_x, MI_MINE_ma, label='MINE')
plt.plot(plot_x, [mi_trad] * plot_x.size, label='True MI', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Mutual Information')
plt.legend()
# pdb.set_trace()
plt.savefig('./fig4/{}_datasetSize_{}_Iterations_{}_batchSize_{}_N_{}_D_{}_H1_{}_H2_{}_lr_{}_trueMI_{}.png'.format(name, ds, 
n_epoch, bs, N, D, H1, H2, lr, mi_trad.round(4)), dpi=500)