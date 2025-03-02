from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, task_type):
        super(NeuralNetwork, self).__init__()
        
        layer_stack = []
        prev = input_size
        for hidden in hidden_layers:
            layer_stack.append(nn.Linear(prev, hidden))
            layer_stack.append(nn.ReLU())
            prev = hidden
        # Output
        layer_stack.append(nn.Linear(prev, output_size))
        if task_type == "titanic":  
            layer_stack.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layer_stack) # Unpack layer_stack

    def forward(self, x):
       return self.model(x)