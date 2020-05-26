import numpy as np
import pandas as pd

dataset = np.array(pd.read_csv("train.csv" , header =[1, 2])) 

X = dataset[:, 1:]


input_layer_size = 784
hidden_layer_size = 784
output_layer_size = 10

def gen_y(digit):
	return np.array( [ (1 if i == digit else 0) for i in range(output_layer_size) ] )

Y = np.array( [gen_y( example[0] ) for example in dataset] )

tetha_l1 = np.random.rand( input_layer_size , hidden_layer_size )
tetha_l2 = np.random.rand( hidden_layer_size  , output_layer_size )

init_epsilon = 0.12
# shift two random tetha to be in range of (-init_epsilon , init_epsilon)
tetha_l1 = np.array( [ ( el - 0.5 ) * ( init_epsilon / 0.5 ) for el in tetha_l1 ] )
tetha_l2 = np.array( [ ( el - 0.5 ) * ( init_epsilon / 0.5 ) for el in tetha_l2 ] )

def sigmoid(x):
	return (1.0 / (1 + np.exp(-1 * x )))
	
def forward_propagation(input_X):
	hidden_layer = np.array( [ sigmoid(x) for x in tetha_l1.dot(input_X.T) ] )
	output_layer = np.array( [ sigmoid(x) for x in tetha_l2.T.dot(hidden_layer.T) ] )
	print(output_layer)	
	
forward_propagation(X[0])	

