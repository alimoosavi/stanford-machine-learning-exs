import numpy as np
import pandas as pd

dataset = np.array(pd.read_csv("train.csv" , header =[1, 2])) 
X = dataset[:, 1:]
feature_number = 784
 
def sigmoid(x):
	
	sigm = 0.0
	
	try:
		sigm = 1.0 / (1 + np.exp(-1 * x))
	except:
		sigm = 1 if x > 0 else 0
		
	return sigm


		
def calc_loss_func_deriv(tetha , digit):
	
	y_label = lambda y: 1 if y == digit else 0
	tetha_minus_y = lambda example: sigmoid( tetha.T.dot(example[1:]) ) - y_label(example[0])
	
	h_minus_y_vector = np.array([tetha_minus_y(example) for example in dataset])
	
	return (1.0 / float(len(dataset)) ) * X.T.dot(h_minus_y_vector.T) 
	


def train_digit( digit):
	
	tetha = np.zeros(feature_number)
	
	rate = 0.01
		
	next_tetha = np.subtract( tetha , rate * calc_loss_func_deriv(tetha ,  digit) )
	
	iteration = 0
	max_iteration = 1000
	while  abs( np.linalg.norm ( calc_loss_func_deriv(tetha, digit) ) ) > abs( np.linalg.norm(calc_loss_func_deriv(next_tetha , digit))) and iteration < max_iteration :
		iteration+=1
		tetha = np.subtract(tetha , rate * calc_loss_func_deriv(tetha, digit) ) 
		next_tetha = np.subtract( next_tetha ,  rate * calc_loss_func_deriv(next_tetha , digit) )  	
		
		  		
	return tetha

digits_tetha = [ train_digit(digit) for digit in range(10) ]


