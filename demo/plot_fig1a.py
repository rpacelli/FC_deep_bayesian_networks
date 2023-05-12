import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
import matplotlib.lines as mlines

N0 = 784 

lambda0 = "1.0"
lambda1 = "1.0"
parentdir = "."
datasets= ["mnist","cifar10"]
markers = ["o","v","d","o","v","d"]

cm = 1/2.54 
matplotlib.rcParams.update({'font.size': 11})

fig, ax  = plt.subplots(figsize=(8*cm,15*cm))
fig.text(-0.1, 1., "(a)", va='top')

mnist_infwidths =[0.06967739373994239 ,0.03788431974904857, 0.017212101617447764 ]
cifar10_infwidths =[0.2582984345677857,0.22344282674723207,0.18900999481657496 ]
greens = ["#78AE99","#1DA891","#008080"]
blues = ["#2D82BD","#0787BE","#140CAB"]
colors = [*greens,*blues]
j = 0
lines = []
for el in datasets:
    tmp = 0
    exp_dir = f"{os.path.split(os.getcwd())[0]}/runs/teacher_{el}_net_1hl_actf_erf"
    exp_file = f"{exp_dir}/experiment_N_{N0}_lambda0_{lambda0}_lambda1_{lambda1}.txt"
    theory_file = f"{exp_dir}/theory_N_{N0}_lambda0_{lambda0}_lambda1_{lambda1}.txt"

    exp = np.loadtxt(exp_file)
    theory = np.loadtxt(theory_file)
    P_theo = theory[:,0]
    
    infwidths =eval(f"{el}_infwidths")
    i = 0
    for P in (100,300, 1000):
        P_exp = exp[:,0]
        to_take_theo = P_theo == np.full( len(P_theo),P)
        N_theory = theory[:,1][to_take_theo]
        test_theory = theory[:,3][to_take_theo]
        theo_order = np.argsort(N_theory)
        N_theory = N_theory[theo_order]
        test_theory = test_theory[theo_order]
        to_take_exp = P_exp == np.full( len(P_exp),P)
        N_toplot = exp[:,1][to_take_exp]
        test_toplot = exp[:,4][to_take_exp]
        order = np.argsort(N_toplot)
        N_toplot = N_toplot[order]
        test_toplot = test_toplot[order]
        bars = exp[:,5][to_take_exp]
        bars = bars[order]
        print(N_toplot,test_toplot)
        ax.scatter(N_toplot,test_toplot, 15,color = colors[j],marker = markers[i], label = f"P = {int(P)} {el}")
        ax.errorbar(N_toplot,test_toplot,bars,color = colors[j],alpha = 0.5,marker = "",linestyle="")
        if i==2:
                line = mlines.Line2D([0], [0], linestyle= "dotted", label = f"infinite width", color =colors[j])
                lines.append(line)
                line = mlines.Line2D([0], [0], linestyle= "-", label = f"theory", color =colors[j])
                lines.append(line)
        ax.plot(N_theory,test_theory,color = colors[j],marker = "",linestyle = "-")
        ax.axhline(y = infwidths[i], color = colors[j], linestyle = 'dotted')
        i += 1
        j +=1
        



ax.set_xlabel(r"$N_{1}$")

ax.set_ylabel("test loss")
legend2 = ax.legend(handles=[*lines],labelspacing = 0.2, loc = 'upper left', fontsize = 7)
ax.add_artist(legend2)
ax.legend(labelspacing = 0.2,fontsize = 7)
ax.set_yticks([i/10 for i in range(0,5,1)])
ax.set_ylim((-0.01,0.41))
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.xscale("log")
plt.savefig(f"./experiment_N_{N0}_lambda0_{lambda0}_lambda1_{lambda1}.png")