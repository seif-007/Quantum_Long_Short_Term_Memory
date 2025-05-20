# 2024 11 24: Modern QLSTM version

# Datetime
from datetime import datetime
import time


import matplotlib.pyplot as plt
from pandas import DataFrame

import warnings

import pennylane as qml
import numpy as np

# Saving
import pickle
import os
import copy

# sklearn
from sklearn.preprocessing import StandardScaler

# Dataset

from data.damped_shm import get_damped_shm_data

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim


##
### Training routine

def train_epoch_full(opt, model, X, Y, batch_size):
	losses = []

	for beg_i in range(0, X.shape[0], batch_size):
		X_train_batch = X[beg_i:beg_i + batch_size]
		# print(x_batch.shape)
		Y_train_batch = Y[beg_i:beg_i + batch_size]

		# opt.step(closure)
		since_batch = time.time()
		opt.zero_grad()
		# print("CALCULATING LOSS...")
		model_res, _ = model(X_train_batch)
		loss = nn.MSELoss()
		loss_val = loss(model_res.transpose(0,1)[-1], Y_train_batch) # 2024 11 11: .transpose(0,1)
		# print("BACKWARD..")
		loss_val.backward()
		losses.append(loss_val.data.cpu().numpy())
		opt.step()
		# print("LOSS IN BATCH: ", loss_val)
		# print("FINISHED OPT.")
		# print("Batch time: ", time.time() - since_batch)
		# print("CALCULATING PREDICTION.")
	losses = np.array(losses)
	return losses.mean()

##############

### Plotting and Saving

def saving(exp_name, exp_index, train_len, iteration_list, train_loss_list, test_loss_list, model, simulation_result, ground_truth):
	# Generate file name
	file_name = exp_name + "_NO_" + str(exp_index) + "_Epoch_" + str(iteration_list[-1])
	saved_simulation_truth = {
	"simulation_result" : simulation_result,
	"ground_truth" : ground_truth
	}

	if not os.path.exists(exp_name):
		os.makedirs(exp_name)

	# Save the train loss list
	with open(exp_name + "/" + file_name + "_TRAINING_LOST" + ".txt", "wb") as fp:
		pickle.dump(train_loss_list, fp)

	# Save the test loss list
	with open(exp_name + "/" + file_name + "_TESTING_LOST" + ".txt", "wb") as fp:
		pickle.dump(test_loss_list, fp)

	# Save the simulation result
	with open(exp_name + "/" + file_name + "_SIMULATION_RESULT" + ".txt", "wb") as fp:
		pickle.dump(saved_simulation_truth, fp)

	# Save the model parameters
	torch.save(model.state_dict(), exp_name + "/" +  file_name + "_torch_model.pth")

	# Plot
	plotting_data(exp_name, exp_index, file_name, iteration_list, train_loss_list, test_loss_list)
	plotting_simulation(exp_name, exp_index, file_name, train_len, simulation_result, ground_truth)

	return


def plotting_data(exp_name, exp_index, file_name, iteration_list, train_loss_list, test_loss_list):
	# Plot train and test loss
	fig, ax = plt.subplots()
	# plt.yscale('log')
	ax.plot(iteration_list, train_loss_list, '-b', label='Training Loss')
	ax.plot(iteration_list, test_loss_list, '-r', label='Testing Loss')
	leg = ax.legend();

	ax.set(xlabel='Epoch', 
		   title=exp_name)
	fig.savefig(exp_name + "/" + file_name + "_" + "loss" + "_"+ datetime.now().strftime("NO%Y%m%d%H%M%S") + ".pdf", format='pdf')
	plt.clf()

	return

def plotting_simulation(exp_name, exp_index, file_name, train_len, simulation_result, ground_truth):
	# Plot the simulation
	plt.axvline(x=train_len, c='r', linestyle='--')
	plt.plot(simulation_result, '-')
	plt.plot(ground_truth.detach().numpy(), '--')
	plt.suptitle(exp_name)
	# savfig can only be placed BEFORE show()
	plt.savefig(exp_name + "/" + file_name + "_" + "simulation" + "_"+ datetime.now().strftime("NO%Y%m%d%H%M%S") + ".pdf", format='pdf')
	return


#################

## VQC components

##

def H_layer(nqubits):
		"""Layer of single-qubit Hadamard gates.
		"""
		for idx in range(nqubits):
			qml.Hadamard(wires=idx)

def RY_layer(w):
	"""Layer of parametrized qubit rotations around the y_tilde axis."""
	for idx, element in enumerate(w):
		qml.RY(element, wires=idx)

def entangling_layer(nqubits):
	""" Layer of CNOTs followed by another shifted layer of CNOT."""
	# In other words it should apply something like :
	# CNOT  CNOT  CNOT  CNOT...  CNOT
	#   CNOT  CNOT  CNOT...  CNOT
	for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
		qml.CNOT(wires=[i, i + 1])
	for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
		qml.CNOT(wires=[i, i + 1])


# Define actual circuit architecture
def q_function(x, q_weights, n_class):
	""" The variational quantum circuit. """

	# Reshape weights
	# θ = θ.reshape(vqc_depth, n_qubits)

	# Start from state |+> , unbiased w.r.t. |0> and |1>

	n_dep = q_weights.shape[0]
	n_qub = q_weights.shape[1]

	H_layer(n_qub)

	# Embed features in the quantum node
	RY_layer(x)

	# Sequence of trainable variational layers
	for k in range(n_dep):
		entangling_layer(n_qub)
		RY_layer(q_weights[k])

	# Expectation values in the Z basis
	exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_class)]  # only measure first "n_class" of qubits and discard the rest
	return exp_vals


# Wrapped previous model as a PyTorch Module
class VQC(nn.Module):
	def __init__(self, vqc_depth, n_qubits, n_class):
		super().__init__()
		self.weights = nn.Parameter(0.01 * torch.randn(vqc_depth, n_qubits))  # g rotation params
		self.dev = qml.device("default.qubit", wires=n_qubits)  # Can use different simulation backend or quantum computers.
		self.VQC = qml.QNode(q_function, self.dev, interface = "torch")

		self.n_class = n_class


	def forward(self, X):
		y_preds = torch.stack([torch.stack(self.VQC(x, self.weights, self.n_class)) for x in X]) # PennyLane 0.35.1
		return y_preds

##
##
##

class CustomLSTMCell(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(CustomLSTMCell, self).__init__()
		self.hidden_size = hidden_size

		# Linear layers for gates and cell update
		self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
		self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
		self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
		self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

		self.output_post_processing = nn.Linear(hidden_size, output_size)

	def forward(self, x, hidden):
		h_prev, c_prev = hidden

		# Concatenate input and hidden state
		combined = torch.cat((x, h_prev), dim=1)

		# Compute gates
		i_t = torch.sigmoid(self.input_gate(combined))  # Input gate
		f_t = torch.sigmoid(self.forget_gate(combined))  # Forget gate
		g_t = torch.tanh(self.cell_gate(combined))      # Cell gate
		o_t = torch.sigmoid(self.output_gate(combined)) # Output gate

		# Update cell state
		c_t = f_t * c_prev + i_t * g_t

		# Update hidden state
		h_t = o_t * torch.tanh(c_t)

		# Actual outputs
		out = self.output_post_processing(h_t)

		return out, h_t, c_t

##

class CustomQLSTMCell(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, vqc_depth):
		super(CustomQLSTMCell, self).__init__()
		self.hidden_size = hidden_size

		# Linear layers for gates and cell update
		# Change here to use PEennyLane Quantum VQCs.
		self.input_gate = VQC(vqc_depth = vqc_depth, n_qubits = input_size + hidden_size, n_class = hidden_size)
		self.forget_gate = VQC(vqc_depth = vqc_depth, n_qubits = input_size + hidden_size, n_class = hidden_size)
		self.cell_gate = VQC(vqc_depth = vqc_depth, n_qubits = input_size + hidden_size, n_class = hidden_size)
		self.output_gate = VQC(vqc_depth = vqc_depth, n_qubits = input_size + hidden_size, n_class = hidden_size)

		self.output_post_processing = nn.Linear(hidden_size, output_size)

	def forward(self, x, hidden):
		h_prev, c_prev = hidden

		# Concatenate input and hidden state
		combined = torch.cat((x, h_prev), dim=1)

		# Compute gates
		i_t = torch.sigmoid(self.input_gate(combined))  # Input gate
		f_t = torch.sigmoid(self.forget_gate(combined))  # Forget gate
		g_t = torch.tanh(self.cell_gate(combined))      # Cell gate
		o_t = torch.sigmoid(self.output_gate(combined)) # Output gate

		# Update cell state
		c_t = f_t * c_prev + i_t * g_t

		# Update hidden state
		h_t = o_t * torch.tanh(c_t)

		# Actual outputs
		out = self.output_post_processing(h_t)

		return out, h_t, c_t


##

class CustomLSTM(nn.Module):
	def __init__(self, input_size, hidden_size, lstm_cell_QT):
		super(CustomLSTM, self).__init__()
		self.hidden_size = hidden_size

		# Single LSTM cell
		self.cell = lstm_cell_QT

	def forward(self, x, hidden=None):
		batch_size, seq_len, _ = x.size()

		# Initialize hidden and cell states if not provided
		if hidden is None:
			h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
			c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
		else:
			h_t, c_t = hidden

		outputs = []

		# Process sequence one time step at a time
		for t in range(seq_len):
			x_t = x[:, t, :]  # Extract the t-th time step
			# print("x_t.shape: {}".format(x_t.shape))
			out, h_t, c_t = self.cell(x_t, (h_t, c_t))  # Update hidden and cell states
			# print("out: {}".format(out))
			outputs.append(out.unsqueeze(1))  # Collect output for this time step

		outputs = torch.cat(outputs, dim=1)  # Concatenate outputs across all time steps
		# print("outputs: {}".format(outputs))
		return outputs, (h_t, c_t)


def main():

	torch.manual_seed(0)

	#

	dtype = torch.DoubleTensor

	x, y = get_damped_shm_data()

	num_for_train_set = int(0.67 * len(x))

	x_train = x[:num_for_train_set].type(dtype)
	y_train = y[:num_for_train_set].type(dtype)

	x_test = x[num_for_train_set:].type(dtype)
	y_test = y[num_for_train_set:].type(dtype)

	print("x_train: ", x_train)
	print("x_test: ", x_test)
	print("x_train.shape: ", x_train.shape)
	print("x_test.shape: ", x_test.shape)

	x_train_transformed = x_train.unsqueeze(2)
	x_test_transformed = x_test.unsqueeze(2)

	print("x_train: ", x_train_transformed)
	print("x_test: ", x_test_transformed)
	print("x_train.shape: ", x_train_transformed.shape)
	print("x_test.shape: ", x_test_transformed.shape)

	print(x_train[0])
	print(x_train_transformed[0])

	print("y.shape: {}".format(y.shape))


	# Example usage
	input_size = 1
	hidden_size = 5
	seq_length = 4
	batch_size = 10

	output_size = 1

	qnn_depth = 5
	qlstm_cell = CustomQLSTMCell(input_size, hidden_size, output_size, qnn_depth).double()
	

	model = CustomLSTM(input_size, hidden_size, qlstm_cell).double()
	
	input_data = torch.randn(batch_size, seq_length, input_size).double()

	# Forward pass
	output, (h_n, c_n) = model(input_data)

	print("Output shape:", output.shape)  # [batch_size, seq_length, hidden_size]
	print("Hidden state shape:", h_n.shape)  # [batch_size, hidden_size]
	print("Cell state shape:", c_n.shape)  # [batch_size, hidden_size]

	print("Output BEFORE transpose: {}".format(output))

	output = output.transpose(0,1)
	print("Output shape:", output.shape)
	print("Output AFTER transpose: {}".format(output))

	print(output[-1])

	# Check the trainable parameters
	print("Show the parameters in QLSTM.")
	for name, param in model.named_parameters():
		if param.requires_grad:
			print(f"Parameter name: {name}")
			print(f"Parameter shape: {param.shape}")
			# print(f"Parameter grad: {param.grad}")
			# print(f"Parameter value: {param.data}\n")

	##

	exp_name = "QLSTM_TS_MODEL_DAMPED_SHM_1"
	exp_index = 1
	train_len = len(x_train_transformed)


	opt = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
	
	train_loss_for_all_epoch = []
	test_loss_for_all_epoch = []
	iteration_list = []

	for i in range(100):
		iteration_list.append(i + 1)
		train_loss_epoch = train_epoch_full(opt = opt, model = model, X = x_train_transformed, Y = y_train, batch_size = 10)


		# Calculate test loss
		test_loss = nn.MSELoss()
		model_res_test, _ = model(x_test_transformed)
		test_loss_val = test_loss(model_res_test.transpose(0,1)[-1], y_test).detach().numpy() # 2024 11 11: .transpose(0,1)
		print("TEST LOSS at {}-th epoch: {}".format(i, test_loss_val))

		train_loss_for_all_epoch.append(train_loss_epoch)
		test_loss_for_all_epoch.append(test_loss_val)

		# Run the test
		test_run_res, _ = model(x.type(dtype).unsqueeze(2))
		total_res = test_run_res.transpose(0,1)[-1].detach().cpu().numpy() # 2024 11 11: .transpose(0,1)
		ground_truth_y = y.clone().detach().cpu()

		saving(
				exp_name = exp_name, 
				exp_index = exp_index, 
				train_len = train_len, 
				iteration_list = iteration_list, 
				train_loss_list = train_loss_for_all_epoch, 
				test_loss_list = test_loss_for_all_epoch, 
				model = model, 
				simulation_result = total_res, 
				ground_truth = ground_truth_y)

	return


if __name__ == '__main__':
	main()


