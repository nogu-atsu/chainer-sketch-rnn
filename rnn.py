import argparse
import logging
#cuDNNは無効にしないと動かない
import os
os.system('export CHAINER_CUDNN=0')

import pickle
import random

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, link, optimizers, reporter, serializers, Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cupy as xp

parser = argparse.ArgumentParser(description='sketch-rnn')
parser.add_argument('--out_dir', '-o', default='', help='Output directory path')

args = parser.parse_args()
if args.out_dir != None:
    if not os.path.exists(args.out_dir):
        try:
            os.mkdir(args.out_dir)
        except:
            print 'cannot make directory {}'.format(args.out_dir)
            exit()
    elif not os.path.isdir(args.out_dir):
        print 'file path {} exists but is not directory'.format(args.out_dir)
        exit()
            
            
            
with open("../sketch-rnn/data/kanji.cpkl","rb") as f:
    data = pickle.load(f,encoding="bytes")

data = [d for d in data if len(d)<40 ]
print(len(data))

l_max = np.max([len(d) for d in data])

one_hot = xp.eye(3)


data = [xp.concatenate([xp.array(d[:,:2]),one_hot[d[:,2].astype(xp.int32)]],axis=1).astype(xp.float32) for d in data]
data2 = [Variable(xp.concatenate([d, xp.array([[0, 0, 0, 0, 1]],dtype=xp.float32)[np.zeros(l_max-d.shape[0]+1,dtype=np.int32)]])) for d in data]
data = [Variable(xp.concatenate([d, xp.array([[0, 0, 0, 0, 1]],dtype=xp.float32)[np.zeros(1,dtype=np.int32)]])) for d in data]

"""
l1=L.NStepLSTM(1, 5, 50, 0.5)
l1.to_gpu()
h1,c1,y1=l1(xp.zeros((1,len(data),50),xp.float32),xp.zeros((1,len(data),50),xp.float32),data,False)
print(h1.shape)#最終隠れ層の状態 (1, 11167, 50)
print(c1.shape)#最終セル状態 (1, 11167, 50)
print(len(y1),y1[0].shape)#出力xバッチ数 11167  (12, 50)
"""
def show_data(data, path):
    for i in range(len(data)):
        plt.subplot(8,4,i+1)
        lines_x = [[0.0]]
        lines_y = [[0.0]]
    
        for d in data[i]:
            if d[4]==1:
                break
            if lines_x[-1]!=[]:
                lines_x[-1].append(d[0]+lines_x[-1][-1])
                lines_y[-1].append(-d[1]+lines_y[-1][-1])
            else:
                lines_x[-1].append(d[0]+lines_x[-2][-1])
                lines_y[-1].append(-d[1]+lines_y[-2][-1])
            if d[3] == 1:
                lines_x.append([])
                lines_y.append([])
        for lx, ly in zip(lines_x, lines_y):
            plt.plot(lx,ly)
            
            
            plt.tick_params(labelbottom='off')
            plt.tick_params(labelleft='off')
            plt.tick_params(labeltop='off')
            plt.tick_params(labelright='off')
            
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            
            plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(path)
    print("finished saving!!")
    plt.close()
    
    
    
class BLSTM(chainer.Chain):

    def __init__(self, layer_num, n_in, n_out, dropout=0.2):
        super(BLSTM, self).__init__(
            f_lstm=L.NStepLSTM(layer_num, n_in, n_out, dropout),
            b_lstm=L.NStepLSTM(layer_num, n_in, n_out, dropout),
        )
        self.n_out = n_out

    def __call__(self, xs, train=True):
        xs_b = [x[::-1] for x in xs]
        leng = len(xs)
        hs_f, _, _ = self.f_lstm(xp.zeros((1, leng, self.n_out), xp.float32), xp.zeros((1, leng, self.n_out), xp.float32), xs, train)
        hs_b, _, _ = self.b_lstm(xp.zeros((1, leng, self.n_out), xp.float32), xp.zeros((1, leng, self.n_out), xp.float32), xs_b, train)
        return F.concat([hs_f[-1], hs_b[-1]], axis=1)

class sketch_rnn(chainer.Chain):
    def __init__(self, h_dim, M):
        super(sketch_rnn, self).__init__(
            bl = BLSTM(1, 5, h_dim//2),
            l1 = L.Linear(h_dim, h_dim),
            l2 = L.Linear(h_dim, h_dim),
            l3 = L.Linear(h_dim, h_dim),
            l4 = L.Linear(h_dim, h_dim),
            l = L.NStepLSTM(1, h_dim + 5, h_dim, 0.2),
            #l = L.NStepGRU(1, h_dim + 5, h_dim, 0.2),
            l5 = L.Linear(h_dim, 6 * M + 3),
        )
        self.gmm_m = M
        
    def chainer_2d_normal(self, x1, x2, mu1, mu2, s1, s2, rho):
        norm1 = x1 - mu1
        norm2 = x2 - mu2
        s1s2 = s1 * s2
        z = F.square(norm1 / s1) + F.square(norm2 / s2) - 2 * (rho * norm1 * norm2) / s1s2
        neg_rho = 1 - F.square(rho)
        result = F.exp(- z / (2 * neg_rho))
        denom = (2 * np.pi) * s1s2 * F.sqrt(neg_rho)
        result /= denom
        return result
    
    def get_lossfunc(self, z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen_logits, x1_data, x2_data, pen_data):
        
        result0 = self.chainer_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
        result1 = result0 * z_pi
        result1 = F.sum(result1,axis=1)
        result1 = - F.log(result1 +1e-10)
        
        fs = 1.0 - pen_data[:, 2]
        result1 = F.sum(result1 * fs)/result1.shape[0]
        result2 = F.softmax_cross_entropy(z_pen_logits,xp.argmax(pen_data,axis=1).astype(xp.int32))
        return result1 + result2
        
    def __call__(self, x, x2, train=True):
        h = self.bl(x, train)
        mean = self.l1(h)
        sigma = F.exp(self.l2(h) / 2.0)
        rand = xp.random.normal(0, 1, mean.data.shape).astype(xp.float32)
        z  = mean + sigma * rand
        h0 = F.tanh(self.l3(z))
        c0 = F.tanh(self.l4(z))
        x_ = [xp.concatenate([xp.array([[0, 0, 1, 0, 0]],dtype=xp.float32), xx.data[:-1]], axis=0) for xx in x2]
        x_ = [F.concat([xx, F.concat([F.expand_dims(zz,0) for _ in range(xx.shape[0])],axis=0)], axis=1) for xx, zz in zip(x_, z)]
        
        _, _, y = self.l(h0.reshape(1, len(x), -1), c0.reshape(1, len(x), -1), x_, train)
        #_, y = self.l(h0.reshape(1, len(x), -1), x_, train)
        y = [self.l5(y_) for y_ in y]
        loss = [self.get_lossfunc(F.softmax(y_[:,:self.gmm_m]), y_[:,self.gmm_m:self.gmm_m*2], y_[:,self.gmm_m*2:self.gmm_m*3], F.exp(y_[:,self.gmm_m*3:self.gmm_m*4]), F.exp(y_[:,self.gmm_m*4:self.gmm_m*5]), F.tanh(y_[:,self.gmm_m*5:self.gmm_m*6]), y_[:,-3:], t_.data[:,:1], t_.data[:,1:2], t_.data[:,-3:]) for y_,t_ in zip(y, x2)]
        return sum(loss)
    
    def sample(self, x, tmp=0.01):
        h = self.bl(x, False)
        mean = self.l1(h)
        sigma = F.exp(self.l2(h) / 2.0)
        rand = xp.random.normal(0, 1, mean.data.shape).astype(xp.float32)
        z  = mean + sigma * rand
        h0 = F.tanh(self.l3(z)).reshape(1, len(x), -1)
        c0 = F.tanh(self.l4(z)).reshape(1, len(x), -1)
        sampled = []
        prev_x = [xp.array([[0, 0, 1, 0, 0]], dtype=xp.float32) for i in range(len(x))]
        for i in range(l_max+1):
            x_ = [F.concat([xx, F.concat([F.expand_dims(zz,0) for _ in range(xx.shape[0])],axis=0)], axis=1) for xx, zz in zip(prev_x, z)]
            h0, c0, y = self.l(h0, c0, x_, False)
            #h0,  y = self.l(h0.reshape(1, len(x), -1), x_, False)
            y = [self.l5(y_) for y_ in y]
            pi = [F.softmax(y_[:,:self.gmm_m] / tmp).data[0] for y_ in y]
            idx = [int(xp.random.choice(self.gmm_m, 1, p=pi_)[0]) for pi_ in pi]
            mu1 = [float(y_.data[0, self.gmm_m+id_]) for y_,id_ in zip(y,idx)]
            mu2 = [float(y_.data[0, self.gmm_m * 2+id_]) for y_,id_ in zip(y,idx)]
            s1 = [float(F.exp(y_[0,self.gmm_m*3+id_]).data) * tmp**0.5 for y_,id_ in zip(y,idx)]
            s2 = [float(F.exp(y_[0,self.gmm_m*4+id_]).data) * tmp**0.5 for y_,id_ in zip(y,idx)]
            rho = [float(F.tanh(y_[0,self.gmm_m*5+id_]).data) for y_,id_ in zip(y,idx)]
            q = [F.softmax(y_[:,-3:] / tmp)[0].data for y_ in y]
            mean = [[mu1_, mu2_] for mu1_, mu2_ in zip(mu1, mu2)]
            cov = [[[s1_*s1_, rho_*s1_*s2_], [rho_*s1_*s2_, s2_*s2_]] for s1_, s2_, rho_ in zip(s1, s2, rho)]
            next_pos = [np.random.multivariate_normal(mean_, cov_, 1).astype(np.float32)[0] for mean_, cov_ in zip(mean, cov)]
            next_state = [np.eye(3).astype(np.float32)[np.random.choice(3, 1, p=cuda.to_cpu(q_))[0]] for q_ in q]
            prev_x = [np.r_[n_p,n_s][np.newaxis,:] for n_p,n_s in zip(next_pos, next_state)]
            if sampled == []:
                sampled = prev_x
            else:
                sampled = [np.r_[sam,pre] for sam,pre in zip(sampled,prev_x)]
            #global data2
            #prev_x = [d.data[i][np.newaxis,:] for d in data2]
            prev_x = [xp.array(pre) for pre in prev_x]
        return sampled
    
batchsize = 16

sr = sketch_rnn(128, 20)
serializers.load_hdf5(args.out_dir + 'sketch-rnn23500.model',sr)
sr.to_gpu()

optimizer = optimizers.MomentumSGD(0.05,0.9)
#optimizer = optimizers.Adam(alpha=0.01)
optimizer.setup(sr)
#optimizer.add_hook(chainer.optimizer.WeightDecay(0.001))
optimizer.add_hook(chainer.optimizer.GradientClipping(5.))


for epoch in range(23501, 100000):
    optimizer.lr *= 0.99996
    sr.zerograds()
    inds = np.random.randint(0, len(data), batchsize)
    #xs = random.sample(data, batchsize)
    xs = [data[ind] for ind in inds]
    xs2 = [data2[ind] for ind in inds]
    loss = sr(xs, xs2)
    if epoch%10==0:
        print(epoch, loss.data/batchsize, "lr", optimizer.lr)
        
    loss.backward()
    loss.unchain_backward()
    optimizer.update()
    if epoch%500==0:
        serializers.save_hdf5(args.out_dir + 'sketch-rnn{}.model'.format(str(epoch)), sr)
        
        sampled = sr.sample(data[100:116])+[cuda.to_cpu(d.data) for d in data[100:116]]
        show_data(sampled, "output{}.jpg".format(str(epoch)))