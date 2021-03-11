import torch
import numpy as np
import sys
import math

MSELoss = torch.nn.MSELoss(reduction='mean')
MAELoss = torch.nn.L1Loss(reduction='mean')

def custom_loss2(output, target, device, batch_names=None):
    target = target.reshape(output.shape).to(device) #view(-1, output.size(1)).to(device)
    A, B = get_system_delay_samples(output, target, device) 
   
    loss1 = MAELoss(A,B)
    loss2 = MSELoss(output, target) 

    if batch_names != None:
        print(loss1, loss2)
        l1 = torch.mean(abs(A-B), dim=1)
        l2 = torch.mean((output-target)**2, dim=1)
        for i, filename in enumerate(batch_names):
            correl = np.corrcoef(A[i].cpu().detach().numpy(), B[i].cpu().detach().numpy())[1,0]
           
            if np.isnan(correl):
                print(A[i])
                print(B[i])

            if l1[i] + l2[i] > 1e+3:
                print(" %s -> loss1 %.3f loss2 %.3f (corr %.2f) [LARGE]" % (filename, l1[i], l2[i], correl))
            else:
                print(" %s -> loss1 %.3f loss2 %.3f (corr %.2f) " % (filename, l1[i], l2[i], correl))

    return loss1 + loss2
  


def get_system_delay_samples(output, target, device=None):

    if device ==None:
        device = torch.device('cpu')

    data_num = 1000
    param_raw = torch.normal(0, 1, size=(data_num, 10))
    param = torch.zeros(data_num, 10+31)
    param[:,0:10] = param_raw[:,0:10]
    cnt=10
    for k in range(0,6):
        for m in range(k,6):
            param[:,cnt]=param_raw[:,k]*param_raw[:,m];
            cnt = cnt+1

    for k in range(6,10):
        for m in range(k,10):
            param[:,cnt]=param_raw[:,k]*param_raw[:,m];
            cnt = cnt+1
    A= torch.matmul(param.to(device),output.t().to(device)).t()
    B= torch.matmul(param.to(device),target.t().to(device)).t()
    
    return A, B



