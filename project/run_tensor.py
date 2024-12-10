"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch


# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


# TODO: Implement for Task 2.5.

##Network 1
# class Network(minitorch.Module):
#     def __init__(self, hidden_layers):
#         super().__init__()
#         # Input layer -> Hidden layer
#         self.layer1 = Linear(2, hidden_layers)
#         # Hidden layer -> Output layer
#         self.layer2 = Linear(hidden_layers, 1)

#     def forward(self, x):
#         # First layer with ReLU activation
#         h = self.layer1.forward(x).relu()
#         # Output layer with sigmoid activation
#         return self.layer2.forward(h).sigmoid()

##Network 2 3 layers
class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        h = self.layer1.forward(x).relu()
        h = self.layer2.forward(h).relu()
        return self.layer3.forward(h).sigmoid()


# #Linear 1
# class Linear(minitorch.Module):
#     def __init__(self, in_size, out_size):
#         super().__init__()
#         # Initialize weights and bias
#         self.weights = RParam(out_size, in_size)
#         self.bias = RParam(out_size)

#     def forward(self, x):
#         # Implement linear transformation: y = Wx + b
#         return x @ self.weights.value.transpose() + self.bias.value

# #linear 2
# class Linear(minitorch.Module):
#     def __init__(self, in_size, out_size):
#         super().__init__()
#         # Initialize weights and bias
#         self.weights = RParam(in_size, out_size)  # Transposed order
#         self.bias = RParam(out_size)

#     def forward(self, x):
#         # Matrix multiplication without transpose
#         # Assuming x is shape (batch, in_size)
#         # weights is shape (in_size, out_size)
#         # output should be (batch, out_size)
#         return (x @ self.weights.value) + self.bias.value

# ##Linear 3
# class Linear(minitorch.Module):
#     def __init__(self, in_size, out_size):
#         super().__init__()
#         self.weights = RParam(in_size, out_size)
#         self.bias = RParam(out_size)

#     def forward(self, x):
#         # Manual implementation of matrix multiplication using available operations
#         batch_size = x.shape[0]
#         out_size = self.weights.value.shape[1]

#         # Create output tensor
#         out = minitorch.zeros((batch_size, out_size))

#         # Manually compute matrix multiplication
#         for i in range(batch_size):
#             for j in range(out_size):
#                 sum_val = minitorch.zeros(())
#                 for k in range(self.weights.value.shape[0]):
#                     sum_val = sum_val + x[i, k] * self.weights.value[k, j]
#                 out[i, j] = sum_val + self.bias.value[j]

#         return out

##Linear 4
# class Linear(minitorch.Module):
#     def __init__(self, in_size, out_size):
#         super().__init__()
#         self.weights = RParam(in_size, out_size)
#         self.bias = RParam(out_size)

#     def forward(self, x):
#         # Initialize output
#         batch_size = x.shape[0]
#         out_size = self.weights.value.shape[1]
#         result = minitorch.zeros((batch_size, out_size))

#         # Manual element-wise multiplication and addition
#         for b in range(batch_size):
#             for o in range(out_size):
#                 val = 0.0
#                 for i in range(x.shape[1]):
#                     val += float(x[b, i]) * float(self.weights.value[i, o])
#                 result[b, o] = val + float(self.bias.value[o])

#         return result

##Linear 5
class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        # self.weights = RParam(in_size, 1, out_size)
        # self.bias = RParam(out_size)
        # self.in_size = in_size
        # self.out_size = out_size
        self.weights = RParam(in_size,out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, inputs):
        # Weights are (in, 1, out)
        # Inputs are (batch, in)

        # (in, batch, 1) for input shape
        #Original
        # b_size = inputs.shape[0]
        # inputs_T = inputs.view(1, b_size, self.in_size).permute(2, 1, 0)

        # # Use broadcasting to get:
        # # (in, 1, out) * (in, batch, 1) -> (in, batch, out)
        # broadcasted = self.weights.value * inputs_T
        # product = broadcasted.sum(0).view(b_size, self.out_size)  # Sum over in, then collapse
        # return product + self.bias.value

        batch, in_size = x.shape
        return (
            self.weights.value.view(1, in_size, self.out_size)
            * x.view(batch, in_size,1)
        ).sum(1).view(batch,self.out_size) + self.bias.value.view(self.out_size)


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    # PTS = 50
    # HIDDEN = 2
    # RATE = 0.5
    # data = minitorch.datasets["Simple"](PTS)
    # TensorTrain(HIDDEN).train(data, RATE)

    # #Simple
    # PTS = 50
    # HIDDEN = 8
    # RATE = 0.1
    # data = minitorch.datasets["Simple"](PTS)
    # TensorTrain(HIDDEN).train(data, RATE)

    # #Diag
    # PTS = 50
    # HIDDEN = 8
    # RATE = 0.1
    # data = minitorch.datasets["Diag"](PTS)
    # TensorTrain(HIDDEN).train(data, RATE)

    # #Split
    # PTS = 50
    # HIDDEN = 8
    # RATE = 0.1
    # data = minitorch.datasets["Split"](PTS)
    # TensorTrain(HIDDEN).train(data, RATE)

    # #Xor
    # PTS = 50
    # HIDDEN = 8
    # RATE = 0.1
    # data = minitorch.datasets["Xor"](PTS)
    # TensorTrain(HIDDEN).train(data, RATE)

    # #Circle
    # PTS = 50
    # HIDDEN = 8
    # RATE = 0.1
    # data = minitorch.datasets["Circle"](PTS)
    # TensorTrain(HIDDEN).train(data, RATE)

    #Spiral
    PTS = 50
    HIDDEN = 8
    RATE = 0.1
    data = minitorch.datasets["Spiral"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)


