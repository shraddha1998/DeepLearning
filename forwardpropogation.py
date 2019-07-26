import numpy as np
input_data=np.array([2,1])

weights={'node_0':np.array([2,4]), 'node_1':np.array([4,-5]),'output':np.array([2,7])
}
node_0_value = (input_data * weights['node_0']).sum()
node_1_value = (input_data * weights['node_1']).sum()
hidden_layer_outputs = np.array([node_0_value, node_1_value])
output =  (hidden_layer_outputs * weights['output']).sum()
print(output)
