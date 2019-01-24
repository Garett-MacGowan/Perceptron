import numpy as np

def main():
  data = np.genfromtxt('trainSeeds.csv', delimiter=',')
  data = np.insert(data, 0, 1, axis=1)
  # Setting random weights (eight because first one is theta threshold)
  weights = np.random.rand(data.shape[0], 8)
  weights = train(weights, data)

def activate(totalActivation, threshold):
  if (totalActivation >= threshold):
    return 1
  else:
    return 0

def predict(inputData, weight):
  threshold = weight[0]
  totalActivation = np.sum(np.dot(weight,inputData))
  return activate(totalActivation, threshold)

def weightTrainer(weight, inputData, classLabel):
  #print(weight[0])
  result = predict(inputData, classLabel)
  learningRate = 0.1
  #print('*')
  #print('result ', result)
  #print('classLabel ', classLabel)
  if (result != 1):
    difference = (classLabel - result) * learningRate
    weight = (difference * weight) + weight
  return weight

def train(weights, data):
  # Slice (take all of the first dimension (rows)
  # and all but the last item in the second dimension (columns))
  inputData = data[0:, 0:-1]
  classLabels = data[0:, -1:]
  counter = 0
  while(counter < 3000):
    weights = np.array(list(map(weightTrainer, weights, inputData, classLabels)))
    counter += 1
  return weights

main()
# Data format is
#   [area,
#   perimeter,
#   compactness,
#   length,
#   width,
#   asymmetry coefficient,
#   length of kernel,
#   class]