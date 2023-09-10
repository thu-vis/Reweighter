import torch
import numpy as np
import torch.nn.functional as F

class LinearProgramming:
    def __init__(self, matrix, gt, lam, col_label, pos_idx, neg_idx, loss_coef=(1,1,1), temp=100):
        self.all_matrix = torch.from_numpy(matrix).float()*temp
        self.matrix = self.all_matrix.clone()
        self.lam = torch.from_numpy(lam).float()
        self.loss_coef = loss_coef
        self.col_label = torch.from_numpy(col_label)
        self.n_class = len(self.col_label.unique())
        self.forward()
        self.get_weights()
        
        self.pos_idx = torch.tensor(pos_idx).long()
        self.neg_idx = torch.tensor(neg_idx).long()
        self.loss1_idx = torch.cat((self.pos_idx, self.neg_idx))
        self.loss1_target = torch.cat([torch.ones(len(pos_idx)), torch.zeros(len(neg_idx))])
        self.loss1_weight = torch.cat([torch.ones(len(pos_idx)) * (len(pos_idx) + len(neg_idx)) / len(pos_idx) / 2,
                                torch.ones(len(neg_idx))* (len(pos_idx) + len(neg_idx)) / len(neg_idx) / 2])
        self.look_idx = torch.tensor(pos_idx).long()

        self.gt = torch.from_numpy(gt)
        self.weights_goal = (None, None)

    def __str__(self): return "V5"
    def __repr__(self): return "V5"
        
    def forward(self):
        self.logits = torch.matmul(self.lam, self.matrix)

    def loss1(self):
        return F.binary_cross_entropy_with_logits(self.logits[self.loss1_idx], self.loss1_target, weight=self.loss1_weight)

    def set_lam_range(self, lower, upper):
        self.lower, self.upper = torch.from_numpy(lower).float(), torch.from_numpy(upper).float()

    def set_active_goal(self, updown_idx, updown_target):
        self.weights_goal = (torch.from_numpy(updown_idx).long(), torch.from_numpy(updown_target).float())
        self.look_idx = self.look_idx.numpy()
        self.look_idx = np.union1d(self.look_idx, updown_idx[updown_target==1])
        self.look_idx = np.setdiff1d(self.look_idx, updown_idx[updown_target!=1])
        self.look_idx = torch.tensor(self.look_idx.astype(np.int)).long()

    def loss2(self):
        if self.weights_goal[0] is None or len(self.weights_goal[0]) == 0: return torch.tensor(0)
        updown_idx, updown_target = self.weights_goal
        return F.binary_cross_entropy_with_logits(self.logits[updown_idx], updown_target)

    def loss3(self):
        loss = torch.tensor(0)
        labels = self.col_label[self.look_idx]
        for l in range(self.n_class):
            idx = torch.where(labels==l)[0]
            val = self.weights[self.look_idx[idx]].sum()
            if val > 0: loss = loss + val*torch.log(val)
        return loss + np.log(self.n_class)

    def loss(self):
        l1, l2, l3 = self.loss1(), self.loss2(), self.loss3()
        #print((l1+l2)/self.sigma1+l3/self.sigma2+torch.log(self.sigma1)+torch.log(self.sigma2), l1, l2, l3)
        #return (l1+l2)/self.sigma1+l3/self.sigma2+torch.log(self.sigma1)+torch.log(self.sigma2), l1, l2, l3
        return (l1+l2)/2/self.sigma1/self.sigma1+l3/2/self.sigma2/self.sigma2+torch.log(self.sigma1)+torch.log(self.sigma2), l1, l2, l3
        # return (l1+l2)/2/torch.exp(self.sigma1)+l3/2/torch.exp(self.sigma2)+(self.sigma1+self.sigma2)/2, l1, l2, l3

    def get_weights(self):
        self.weights = F.relu(self.logits)
        self.weights = self.weights / (self.weights.sum()+1e-5)
        return self.weights.clone().detach()
    
    def optimize(self, lr=.1, decay=(0.5,[-1]), max_iter=10000, freq=500, tot=1e-5):
        self.old_weights = self.get_weights()
        self.eval(True)
        self.lam = torch.max(torch.min(self.lam.clone().detach(), self.upper), self.lower).requires_grad_(True)
        self.sigma1, self.sigma2 = torch.tensor(1.0).requires_grad_(True), torch.tensor(1.0).requires_grad_(True)
        # self.sigma1, self.sigma2 = torch.tensor(0.0).requires_grad_(True), torch.tensor(0.0).requires_grad_(True)
        datas = []
        i, diff = 0, 1
        while i < max_iter and diff > tot:
            i += 1
            self.forward()
            self.get_weights()
            loss,l1,l2,l3 = self.loss()
            if i % freq == 0:
                print(i, loss.item(), l1.item(), l2.item(), l3.item())
            czp,cnp,zp,sp,lp,dp,acc1, acc2 = self.eval(i%freq==0)
            datas.append([czp,cnp,zp,sp,lp,dp,acc1,acc2,loss.item(), l1.item(), l2.item(), l3.item()])
            loss.backward()
            with torch.no_grad():
                x, grad = self.lam.data, -self.lam.grad.data
                grad = grad - (grad.dot(x)) * x
                grad[(self.lam.data<self.lower) & (grad < 0)] = 0
                grad[(self.lam.data>self.upper) & (grad > 0)] = 0
                if torch.norm(grad, 2)>0:
                    grad = grad / torch.norm(grad, 2)
                phi = 0.5
                while phi > tot:
                    self.lam.data = np.sqrt(1-phi*phi) * x + phi * grad
                    if self.lam.data.min() < 0:
                        phi *= decay[0]
                        continue
                    self.sigma1.data = self.sigma1.data - self.sigma1.grad.data * phi
                    self.sigma2.data = self.sigma2.data - self.sigma2.grad.data * phi
                    self.forward()
                    self.get_weights()
                    new_loss,_,_,_ = self.loss()
                    if new_loss < loss:
                        break
                    else:
                        phi *= decay[0]
                        
                diff = loss - new_loss
                self.lam.grad.data.zero_()
                self.sigma1.grad.data.zero_()
                self.sigma2.grad.data.zero_()
                # print(self.sigma1, self.sigma2)
        self.new_weights = self.get_weights()
        self.eval(True)
        self.compare()
        return datas

    def eval(self, verbose=False):
        w = self.get_weights()
        thres = 1 / len(w)
        w_zero = (w==0)
        w_small = ((w > 0) & (w <= thres))
        w_large = (w > thres)
        czp = w_zero[self.neg_idx].sum().item() / len(self.neg_idx)
        cnp = w_large[self.pos_idx].sum().item() / len(self.pos_idx)
        zp = w_zero.sum().item(), self.gt[w_zero].sum().item(), (w_zero.sum()-self.gt[w_zero].sum()).item()
        sp = w_small.sum().item(), self.gt[w_small].sum().item(), (w_small.sum()-self.gt[w_small].sum()).item()
        lp = w_large.sum().item(), self.gt[w_large].sum().item(), (w_large.sum()-self.gt[w_large].sum()).item()
        dp = (self.gt*w).sum().item()
        acc1 = (zp[2]+sp[1]+lp[1]) / (zp[0]+sp[0]+lp[0])
        acc2 = (zp[2]+lp[1]) / (zp[0]+lp[0])
        if verbose:
            print("constraint purity: ", czp, cnp)
            print("zero: ", zp, " small: ", sp, " large: ", lp)
            print("dot: ", dp, " acc1: ", acc1, " acc2: ", acc2)
        return czp,cnp,zp[2]/(zp[0]+1e-5),sp[1]/(sp[0]+1e-5),lp[1]/(lp[0]+1e-5),dp,acc1, acc2

    def compare(self):
        thres = 0
        up_weights = torch.where((self.new_weights > thres) & (self.old_weights <= 0))[0]
        down_weights = torch.where((self.old_weights > thres) & (self.new_weights <= 0))[0]
        print(f'up weights: {self.gt[up_weights].sum().item()}/{len(up_weights)}')
        print(f'down weights: {len(down_weights)-self.gt[down_weights].sum().item()}/{len(down_weights)}')
        thres = 1 / len(self.weights)
        up_weights = torch.where((self.new_weights > thres) & (self.old_weights <= 0))[0]
        down_weights = torch.where((self.old_weights > thres) & (self.new_weights <= 0))[0]
        print(f'large up weights: {self.gt[up_weights].sum().item()}/{len(up_weights)}')
        print(f'large down weights: {len(down_weights)-self.gt[down_weights].sum().item()}/{len(down_weights)}')