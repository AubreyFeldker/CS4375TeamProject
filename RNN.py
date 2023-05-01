import numpy as np
import pandas as pd

class RNN:
  def __init__(self, input_size, hidden_size, output_size, update_weight=.05):
    # Set sizes
    self.input_size = input_size
    self.hidden_size = hidden_size
    # For simplicity, having the first x elements of the array be considered the outputs fed into the next input
    self.output_size = output_size
    self.update_weight = update_weight

    # Initialized random weights
    self.weight_matrix_input_multi_hidden = np.random.randn(input_size, hidden_size) * update_weight
    self.weight_matrix_hidden_multi_hidden = np.random.randn(hidden_size, hidden_size) * update_weight
    self.weight_matrix_output_multi_hidden = np.random.randn(output_size, hidden_size) * update_weight
    
    # Additional bias weight present for each layer transition
    self.bias_to_hidden = np.zeros((1, hidden_size))
    self.bias_to_output = np.zeros((1, output_size))

  # tanh activation function utilized
  def tanh(self, x):
      return np.tanh(x)
  def tanh_derivative(self, x):
      return 1 - self.tanh(x) ** 2

  def forward_propagation(self, input_sequence):
    hidden_states = {}
    outputs = {}

    # Each iteration is one pass through the network
    for i in range(len(input_sequence)):
      current_time_step_for_input_sequence = input_sequence.iloc[i].to_numpy().reshape(1, -1)

      if (i == 0):
        hidden = self.tanh(current_time_step_for_input_sequence.dot(self.weight_matrix_input_multi_hidden) + self.bias_to_hidden)
      else:
        hidden = self.tanh(current_time_step_for_input_sequence.dot(self.weight_matrix_input_multi_hidden) + hidden_states[i-1].dot(self.weight_matrix_hidden_multi_hidden) + self.bias_to_hidden)
      
      hidden_states[i] = hidden
      output = self.tanh(hidden.dot(self.weight_matrix_output_multi_hidden.T) + self.bias_to_output)
      outputs[i] = output
      
    return {"outputs": outputs, "hidden_states": hidden_states}


  def current_time_step_hidden_state_error_method(self, Weight_matrix_output_hidden, Weight_matrix_hidden_hidden, current_time_step_output_error, next_time_step_hidden_state_error, cache_hidden_states_i):
    error_from_output_layer = np.dot(Weight_matrix_output_hidden.T, current_time_step_output_error)
    
    if next_time_step_hidden_state_error.shape == (1, Weight_matrix_hidden_hidden.shape[0]):
        next_time_step_hidden_state_error = next_time_step_hidden_state_error.T

    error_from_next_time_step_hidden_state = np.dot(Weight_matrix_hidden_hidden.T, next_time_step_hidden_state_error.reshape(-1, 1))

    combined_error = error_from_output_layer.T + error_from_next_time_step_hidden_state.T
    hidden_state_derivative = self.tanh_derivative(cache_hidden_states_i)

    current_hidden_state_error = combined_error * hidden_state_derivative

    return current_hidden_state_error


  def backward_propagation(self, input_sequence, target_sequence, cache):
    gradient = [np.zeros_like(self.weight_matrix_input_multi_hidden),  
                np.zeros_like(self.weight_matrix_hidden_multi_hidden),  
                np.zeros_like(self.weight_matrix_output_multi_hidden),  
                np.zeros_like(self.bias_to_hidden),  
                np.zeros_like(self.bias_to_output)]   
    
    next_time_step_hidden_state_error = np.zeros_like(self.bias_to_hidden)

    for i in range(len(input_sequence)-1, -1, -1):
      current_time_step_output_error = (cache["outputs"][i][0] - target_sequence.iloc[i].to_numpy()) * self.tanh_derivative(cache["outputs"][i][0])
      current_time_step_gradient_output_to_hidden = current_time_step_output_error.dot(cache["hidden_states"][i][0])

      gradient[2] += current_time_step_gradient_output_to_hidden
      gradient[4] += np.sum(current_time_step_output_error).reshape(-1, 1)

      current_time_step_hidden_state_error = self.current_time_step_hidden_state_error_method(self.weight_matrix_output_multi_hidden, self.weight_matrix_hidden_multi_hidden, current_time_step_output_error, next_time_step_hidden_state_error, cache["hidden_states"][i] )
      
      if i == 0:
        previous_hidden_state = np.zeros_like(self.bias_to_hidden)
      else:
        previous_hidden_state = cache["hidden_states"][i-1][0]

      current_time_step_gradient_hidden_to_hidden = current_time_step_hidden_state_error.dot(previous_hidden_state.T)
      repeated_input_sequence = input_sequence.iloc[i].to_numpy()

      current_time_step_gradient_input_to_hidden = np.dot(current_time_step_hidden_state_error.T, (input_sequence.iloc[i].to_numpy().reshape(1, -1))).T

      gradient[1] += current_time_step_gradient_hidden_to_hidden
      gradient[0] += current_time_step_gradient_input_to_hidden
      print(gradient[0])
      gradient[3] += current_time_step_hidden_state_error

      next_time_step_hidden_state_error = current_time_step_hidden_state_error
    return gradient
  
  def update_weights_and_biases(self, gradients):
    self.weight_matrix_input_multi_hidden -= self.update_weight * gradients[0]
    self.weight_matrix_hidden_multi_hidden -= self.update_weight * gradients[1]
    self.weight_matrix_output_multi_hidden -= self.update_weight * gradients[2]
    self.bias_to_hidden -= self.update_weight * gradients[3]
    self.bias_to_output -= self.update_weight * gradients[4]

  # Train the RNN based off the input data given
  def train(self, training_data, epochs, sequence_length):
   
    for epoch in range(epochs):
      epoch_loss = 0

      for i in range(0, len(training_data) - sequence_length - 1):
        # The output sequence is a set of variables from the next expected value
        input_sequence = training_data[i : i + sequence_length]
        output_sequence = training_data[i+1 : i + sequence_length + 1].iloc[:, :self.output_size]
        cache = self.forward_propagation(input_sequence)
        sequence_loss = 0

        for j in range(sequence_length):
          sequence_loss += np.square(cache["outputs"][j][0] - output_sequence.to_numpy()[j])

        sequence_loss /= sequence_length
        epoch_loss += sequence_loss
        gradients = self.backward_propagation(input_sequence, output_sequence, cache)

        self.update_weights_and_biases(gradients)
          
      epoch_loss /= (len(training_data) - sequence_length + 1)
      print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")

  def predict(self, input_sequence):
    hidden_states = {}
    outputs = {}
    predictions = []

    for i in range(len(input_sequence)):
      current_time_step_for_input_sequence = input_sequence[i].reshape(-1, 1)
      if i == 0:
        hidden = self.tanh(current_time_step_for_input_sequence.dot(self.weight_matrix_input_multi_hidden) + self.bias_to_hidden)
      else:
        hidden = self.tanh(current_time_step_for_input_sequence.dot(self.weight_matrix_input_multi_hidden) + hidden_states[i-1].dot(self.weight_matrix_hidden_multi_hidden) + self.bias_to_hidden)
      
      hidden_states[i] = hidden
      output = self.tanh(hidden.dot(self.weight_matrix_hidden_multi_output) + self.bias_to_output)
      outputs[i] = output
      predictions.append(output)

    predictions = np.array(predictions).reshape(len(input_sequence), -1)
    return predictions
