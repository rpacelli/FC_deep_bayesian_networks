import statistics 
import math
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from operator import itemgetter 
from itertools import groupby
import argparse

def parseArguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("-k", help="how many lines until the run has termalised", type=int , default = 50)
	args = parser.parse_args()
	return args


def load_run(file, k):
	fstring = file.split("_")
	P = int(fstring[2]) #extract P from file name
	lines = np.loadtxt(file)
	n = len(lines)-k
	last_lines = lines[k:]
	trainerr = last_lines[:,1]
	testerr = last_lines[:,2]
	train, err_train = np.mean(trainerr),np.std(trainerr)
	test, err_test = np.mean(testerr), np.std(testerr)
	return P, train, err_train, test, err_test,n


def check_file(filename,n): #returns false if the file is unreadable
	try:
		file = np.loadtxt(filename)
	except:
		print(f"file unreadable: {os.getcwd()}/{filename}")
		return False		
	if (file == np.full_like(file, "inf")).any() or (file == np.full_like(file, "nan")).any():
		print(f"undetermined values: {os.getcwd()}/{filename}")
		return False      
	if len(file) < n:
		print(f"file too short: {os.getcwd()}/{filename}")
		return False
	return True


args = parseArguments()

pwd = os.getcwd().split("/")[-1]
parentdir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
pwd_specs = pwd.split("_")
N0 = pwd_specs[-3]
lambda0 = pwd_specs[5]
lambda1 = pwd_specs[-6]
N1 = pwd_specs[-1]
lr = pwd_specs[1]
T = pwd_specs[3]
output_file = f"{parentdir}/experiment_N_{N0}_lambda0_{lambda0}_lambda1_{lambda1}.txt"
if not isfile(output_file):
	sourceFile= open(output_file, 'a')
	print('#1. P', '2. N1', '3. train error', '4. train error stdev','5. test error', '6. test error stdev','7. n samples','8. lr', '9. temperature', '10. num samples ','11. size of tail', file = sourceFile)
	sourceFile.close()

k = args.k
onlyfiles = [f for f in listdir("./") if f[:3]=="run"]


to_keep = []
for f in onlyfiles:
	if check_file(f, k):
		to_keep.append(f)	

scal_list = [load_run(f,k) for f in to_keep]
scal_list = sorted(scal_list, key =itemgetter(0)) #sort by P 
scal_list = [list(g) for _,g in groupby(scal_list, key=itemgetter(0))] #group by P 

for item in scal_list:
	P = item[0][0]
	n = item[0][-1]
	mat = np.matrix(item)
	num_samples = len(mat[:,1])
	train = np.mean(mat[:,1])
	sq_train_std = np.square(mat[:,2])
	test = np.mean(mat[:,3])
	sq_test_std = np.square(mat[:,4])
	err_train = np.sqrt(np.sum(sq_train_std)/num_samples)
	err_test = np.sqrt(np.sum(sq_test_std)/num_samples)
	sourceFile= open(output_file, 'a')
	print(P, N1, train, err_train, test, err_test, lr, T, num_samples, n, file = sourceFile)
	sourceFile.close()