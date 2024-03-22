import utils, teachers 
from theory import compute_theory

start_time = utils.time.time()
args = utils.parseArguments()
if args.only_theo: 
	args.compute_theory = True 
print(f"\nThe input dimension is: {args.N} \nNumber of examples in the train set: {args.P} \nNumber of examples in the test set: {args.Ptest}" )
if args.L>1 and args.act == "erf":
	print("\nI cannot compute predicted gen error of an erf multilayer network")
	args.compute_theory = False
device = utils.find_device(args.device)

#CREATION OF FOLDERS AND TEACHERS
root_dir = f'./runs/'
utils.make_directory(root_dir)
first_subdir, run_folder = utils.make_folders(root_dir, args)
if args.teacher_type == 'mnist': 
	teacher_class = teachers.mnist_dataset(args.N)
elif args.teacher_type == 'cifar10': 
	teacher_class = teachers.cifar_dataset(args.N)
elif args.teacher_type == 'random': 
	teacher_class = teachers.random_dataset(args.N)

#CRREATION OF DADTASET AND THEORY CALCULATION
inputs, targets, test_inputs, test_targets = teacher_class.make_data(args.P, args.Ptest, device)
theoryFilename = f"{first_subdir}theory_N_{args.N}_lambda0_{args.lambda0}_lambda1_{args.lambda1}.txt"
if args.compute_theory:
	if not utils.os.path.isfile(theoryFilename):
		with open(theoryFilename,"a") as f:
			print('#1 P', '2 N1', '3 Qbar', '4 pred error', '5 s/P', file = f)
	start_time = utils.time.time()
	gen_error_pred, Qbar,yky = compute_theory(inputs, targets, test_inputs, test_targets, args)
	with open(theoryFilename, "a") as f:
		print(args.P, args.N1, Qbar, gen_error_pred, yky,file = f)
	print(f"\nPredicted error is: {gen_error_pred} \n theory computation took - {utils.time.time() - start_time} seconds -")
	start_time = utils.time.time()
inputs, targets, test_inputs, test_targets = inputs.to(device), targets.to(device), test_inputs.to(device), test_targets.to(device)
if args.only_theo:
	exit()

#NET INITIALISATION
net_class = utils.FCNet(args.N, args.N1, args.L)
bias = False
net = net_class.Sequential(bias, args.act)
utils.cuda_init(net, device)	

#TRAINING DYNAMICS SPECIFICATION
criterion = utils.reg_loss
test_criterion = utils.nn.MSELoss(reduction='mean')
optimizer = utils.LangevinOpt(net, lr = args.lr, temperature = args.T)               
train_args = [net,inputs,targets,criterion,optimizer,args.T,args.lambda0,args.lambda1]	
test_args = [net,test_inputs,test_targets,test_criterion]	

#RUN DYNAMICS
runFilename = f'{run_folder}run_P_{args.P}_replica_{args.R}.txt'
if not utils.os.path.exists(runFilename):
	with open(runFilename, 'a') as f:
		print('#1. epoch', '2. train error', '3. test error', file = f)
sourceFile = open(runFilename, 'a')
for epoch in range(args.epochs):
	train_loss = utils.train(*train_args)
	if epoch % args.checkpoint == 0 or epoch == args.epochs-1 :
		test_loss = utils.test(*test_args)
		print(epoch, train_loss, test_loss, file = sourceFile)
		sourceFile.close()
		sourceFile = open(runFilename, 'a')
		print(f'\nEpoch: {epoch} \nTrain error: {train_loss} \nTest error: {test_loss} \n training took --- {utils.time.time() - start_time} seconds ---')				
		start_time = utils.time.time()
sourceFile.close()
