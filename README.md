# deepL_beyond_infwidth

This code trains deep fully connected networks on regression tasks with quadratic loss function in a teacher-student bayesian setting.

The script to be run is ```main.py```.  

### Arguments

The **mandatory** argument that must passed to ```deep_regression.py```is the **teacher type**. This is the function that will generate the input-output distribution. It can be chosen among: 
  - ```random``` (random data and labels)
  - ```mnist``` (0/1 classes of MNIST)
  - ```cifar10``` ("cars" and "planes" classes of CIFAR10) 
  
The optional arguments that can be passed are: 
  Network specifications: 
  - ```-N``` (size of input data)
  - ```-L``` (depth of the network)
  - ```-N1``` (size of the hidden layer(s)) 
  - ```-act``` (activation function. choose among relu, erf)   

  Training
  - ```-P``` (size of the training set)
  - ```-Ptest``` (size of the test set) 
  - ```-lr``` (learning rate)
  - ```-T``` (temperature)
  - ```-epochs``` (number training epochs)
  - ```-checkpoint``` (number of epochs between two checkpoints. Train and test loss are saved at each checkpoint)
  - ```-device``` (chose the device among cuda and cpu. Default is cpu)
  - ```-R``` (a run index. the default value is 0, you can change it if you want more than one run with the same parameters)
  - ```-lambda0``` (gaussian prior of weights of the first and intermediate layers)
  - ```-lambda1``` (gaussian prior of the last layer weights)
  - ```-compute_theory``` (set true to compute theory prediction with the specified trainset)
  - ```-infwidth``` (set true to compute infinite width theory, the default value is False that will trigger the computation of finite-width theory)


### Code output

Ouput files are stored in the ```./runs/``` repository in your working directory. 
The script will output a file ```run_P_(value_of_P)_(run_attributes)``` with the specifications of the single run (epoch, train error, test error), and if prompted will also produce a file with the theoretical expected quantities (P, N1, Qbar, expected error).
 
### Example command prompt line
```
python deep_regression.py cifar10 -L 3 -N 784 -N1 300 -act relu -compute_theory True -infwidth True
```
This will train a 3hl architecture with relu activation functions and hidden layers of size N1 = 300 on a regression task on the cifar10 dataset. The data will be coarse grained to a size of 28x28 and the infinite width predicted theory will be computed.


### Conda environment

To avoid compatibility issues, a conda environment is provided for linux machines. You can create the environment from the file ```env.yml``` running the following line: 

```conda env create -n deep_bayesian --file env.yml```

then activate it with:
``` conda activate deep_bayesian ```

To install conda visit https://conda.io/projects/conda/en/stable/user-guide/install/index.html


### Data analysis




