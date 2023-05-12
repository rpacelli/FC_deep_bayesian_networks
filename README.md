# deepL_beyond_infwidth

This code trains deep fully connected networks on regression tasks with quadratic loss function in a teacher-student setting.

The script to be run is ```deep_regression.py```.  

### Arguments

The **mandatory** argument that must passed to ```deep_regression.py```is the **teacher type**. This is the function that will generate the input-output distribution. It can be chosen among: 
  - ```linear``` (linear function) 
  - ```random``` (random data and labels)
  - ```mnist``` (binarised MNIST dataset of handwritten digits)
  

The optional arguments that can be passed are: 
  Network specifications: 
  - ```-N``` (size of input data)
  - ```-L``` (depth of the network)
  - ```-N1``` (size of the hidden layer(s)) 
  - ```-act``` (activation function. choose among relu, erf)   

  Trainset and testset
  - ```-P``` (size of the training set)
  - ```-Ptest``` (size of the test set) 
  - ```-bs``` (batch size, set to 0 for full batch learning)
  - ```-save_data``` (saves the generate synthetic data distribution) 
  - ```-resume_data``` (set true if you wish to resume training data from previous experiments)
  
  Training dynamics and checkpoint:
  - ```-opt``` (choose among adam, sgd)
  - ```-lr``` (learning rate)
  - ```-wd``` (weight decay)
  - ```-epochs``` (number training epochs)
  - ```-checkpoint``` (number of epochs between two checkpoints. at every checkpoint the errors appear at screen and are saved)
  - ```-device``` (chose the device among cuda and cpu. Default is cpu)
  - ```-R``` (a run index. the default value is 0, you can change it if you want more than one run with the same parameters)
  
  Theory computation:
  - ```-compute_theory``` (set true to compute theory with the specified trainset (resumed, synthetic or mnist))
  - ```-infwidth``` (set true to compute infinite width theory, the default value is False that will trigger the computation of finite-width theory)
  - ```-lambda0``` (inverse of squared variance of 1st layer weight at initialisation)
  - ```-lambda1``` (inverse of squared variance of intermediate layer(s) weight at initialisation)


### Code output

```deep_regression.py``` will create folders and output files in your home folder. 
Change ```mother_dir``` at line ```14``` if you want to modifiy the output folder. The default is ```/your_home_folder/deepL_beyond_infwidth_runs/``` The script will create folders and subfolders named after the arguments used.
The script will output a file ```run_P_(value_of_P)_(run_attributes)``` (in the aforementioned folders) with the specifications of the single run (epoch, train error, test error), and if prompted will also produce a file with the theoretical expected quantities (P, N1, Qbar, expected error).
 
### Example command prompt line
```
python deep_regression.py mnist -L 3 -N 400 -N1 300 -act relu -compute_theory True -infwidth True
```
This will train a 3hl architecture with relu activation functions and hidden layers of size N1 = 300 on a regression task on the mnist dataset. The data will be coarse grained to a size of 20x20 and the infinite width predicted theory will be computed.
