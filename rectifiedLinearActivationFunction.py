'''
The rectified linear activation function (called ReLU) leads to very high-performance networks. This function takes a single number as an input, returning 0 if the input is negative, and the input if the input is positive.

Here are some examples:
relu(3) = 3
relu(-3) = 0 '''

import numpy as np
def relu(input):
    '''Definition relu activation function is here'''
    output = max(input,0)
    return(output)

input_data=np.array([2,1])

weights={'node_0':np.array([2,4]), 'node_1':np.array([4,-5]),'output':np.array([2,7])

node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)

node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)

hidden_layer_outputs = np.array([node_0_output, node_1_output])

# model output relu function is not applied here 
model_output = (hidden_layer_outputs * weights['output']).sum()
