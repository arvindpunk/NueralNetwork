import numpy as np

# variables
layerLengths = [2, 3, 1]

np.random.seed(0)

x = np.array([	[0, 0],
				[0, 1],
				[1, 0],
				[1, 1]	])

y = np.array([	[0],
				[1],
				[1],
				[0]		])

# weight matrix
w1 = np.random.rand(layerLengths[0], layerLengths[1])
w2 = np.random.rand(layerLengths[1], layerLengths[2])

# bias matrix
b1 = np.random.rand(layerLengths[1])
b2 = np.random.rand(layerLengths[2])

# activationFunction
def sigmoid(x, derivative=False):
	if derivative:
		return sigmoid(x) * (1 - sigmoid(x))
	else:
		return 1/(1+np.exp(-x))

def feedForward(input):
	hidden = sigmoid(np.add(np.matmul(input, w1), b1))
	# print("Hidden layer activations:")
	# print(hidden)
	output = sigmoid(np.add(np.matmul(hidden, w2), b2))
	# print("Output layer activations:")
	# print(output)
	return hidden, output

def backPropogation(inp, hidden, output, error):
	print(output)
	print(sigmoid(output))
	print(error * sigmoid(output, derivative=True))
	# delta1 = error * sigmoid(out, derivative=True)

def cost(out):
	return np.mean(np.absolute(y - out))

for i in range(1):
	inp = x
	hidden, output = feedForward(inp)
	error = cost(output)
	print(error)
	backPropogation(inp, hidden, output, error)
# def backpropogate(cost):	
	