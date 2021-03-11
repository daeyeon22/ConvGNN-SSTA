import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import numpy as np
import torch
import torch_geometric
from statsmodels.graphics.gofplots import qqplot_2samples
from loss import MSELoss, MAELoss, get_system_delay_samples



def get_plot(output, target):
    target = target.reshape(output.shape).cpu()
    output = output.cpu()

    A, B = get_system_delay_samples(output, target)
    loss1 = MAELoss(A,B)
    loss2 = MSELoss(output, target) 
    tot_loss = loss1 + loss2
    A = A.cpu().detach().numpy().flatten()
    B = B.cpu().detach().numpy().flatten()

    corr = np.corrcoef(A, B)[1,0]
    
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot()
    ax.scatter(A,B)
    fig.suptitle("loss1 %.2f loss2 %.2f (corr %.2f tot %.2f)" % (loss1, loss2, corr, tot_loss))
    #fig.savefig('%s/%s_scatter.png' % (fig_home, fig_name), dpi=150)
    #plt.close(fig) 
    return fig




def draw_plot(model, args, test_dataset, dirname=None):
    print("=> draw plot")
    batch_size = 1 #args['batch']
    verbose = args['verbose']
    device = args['device']
    save_home = args['save_home']
    model_name = args['model_name']
    criterion = args['criterion']

    if dirname == None:
        fig_home = "%s/figs" % save_home
    else:
        fig_home = "%s/figs/%s" % (save_home, dirname)

    if not os.path.isdir(fig_home):
        os.makedirs(fig_home)

    test_loader = torch_geometric.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    #validation

    xs = np.array([])
    ys = np.array([])

    model.eval()
    for i, (data, file_names) in enumerate(test_loader):
        output = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)

        target = data.y
        target = target.reshape(output.shape).to(device)
        A, B = get_system_delay_samples(output, target, device)

        loss1 = MAELoss(A,B)
        loss2 = MSELoss(output, target) 

        tot_loss = loss1 + loss2
        A = A.cpu().detach().numpy().flatten()
        B = B.cpu().detach().numpy().flatten()
        # save figure
        bench_name = file_names[0]
        fig_name = file_names[0].split('.')[0]

        corr = np.corrcoef(A, B)[1,0]

        pp_A = sm.ProbPlot(A)
        pp_B = sm.ProbPlot(B)

        fig = qqplot_2samples(pp_A, pp_B, line="45")
        fig.suptitle("[%s] loss1 %.2f loss2 %.2f (corr %.2f tot %.2f)" % (bench_name, loss1, loss2, corr, tot_loss))
        fig.savefig('%s/%s_qqplot.png' % (fig_home, fig_name), dpi=150)
        plt.close(fig)


        fig = plt.figure(dpi=150)
        ax = fig.add_subplot()
        ax.scatter(A,B)
        fig.suptitle("[%s] loss1 %.2f loss2 %.2f (corr %.2f tot %.2f)" % (bench_name, loss1, loss2, corr, tot_loss))
        fig.savefig('%s/%s_scatter.png' % (fig_home, fig_name), dpi=150)
        plt.close(fig) 
