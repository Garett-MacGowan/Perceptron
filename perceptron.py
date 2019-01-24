import numpy as np

def main():
  data = np.genfromtxt('trainSeeds.csv', delimiter=',')
  data = np.insert(data, 0, 1, axis=1)
  # Setting random weights (eight because first one is theta threshold)
  # Three sets because we need three output neurons
  weights0 = np.random.rand(1, 8)
  weights1 = np.random.rand(1, 8)
  weights2 = np.random.rand(1, 8)
  weights = [weights0, weights1, weights2]
  weights = train(weights, data)

  # Verify accuracy on test set
  testData = np.genfromtxt('testSeeds.csv', delimiter=',')
  testData = np.insert(testData, 0, 1, axis=1)
  test(weights, testData)

def activate(totalActivation, threshold):
  if (totalActivation >= threshold):
    return 1
  else:
    return 0

def predict(weight, inputData):
  threshold = weight[0]
  totalActivation = np.sum(np.dot(weight,inputData))
  #print('threshold ', threshold)
 # print('totalAct ', totalActivation)
  return activate(totalActivation, threshold)

def weightTrainer(weight, inputData, classLabel, targetLabel):
  result = predict(weight, inputData)
  learningRate = 0.1
  # If result == 1 and classLabel == targetLabel -> good, don't update
  # If result != 1 and classLabel == targetLabel -> update weights positively
  # If result == 1 and classLabel != targetLabel -> update weights negatively
  if (result != 1 and classLabel == targetLabel):
    difference = (1) * learningRate
    weight = (difference * inputData) + weight
  elif (result == 1 and classLabel != targetLabel):
    difference = (-1) * learningRate
    weight = (difference * inputData) + weight
  return weight

def train(weights, data):
  # Slice (take all of the first dimension (rows)
  # and all but the last item in the second dimension (columns))
  inputData = data[0:, 0:-1]
  classLabels = data[0:, -1:]
  counter = 0
  while(counter < 100000):
    weights0 = np.mean(np.array(list(map(weightTrainer, np.full((data.shape[0], 8), weights[0]), inputData, classLabels, np.full((data.shape[0], 1), 1)))), 0)
    weights1 = np.mean(np.array(list(map(weightTrainer, np.full((data.shape[0], 8), weights[1]), inputData, classLabels, np.full((data.shape[0], 1), 2)))), 0)
    weights2 = np.mean(np.array(list(map(weightTrainer, np.full((data.shape[0], 8), weights[2]), inputData, classLabels, np.full((data.shape[0], 1), 3)))), 0)
    weights = [weights0, weights1, weights2]
    counter += 1
  return weights

def predictionDecider(class1, class2, class3):
  # According to data distribution shown in instruction
    # Kama is distributed between Rosa and Canadian where
    # Kama = class1, Rosa = class2, Canadian = class3
  # Since kama is between Rosa and Canadian, any time Kama
  # and one of the others is predicted, predict the other.
  
  if (class1 == 1):
    return 1
  if (class2 == 1):
    return 2
  if (class3 == 1):
    return 3
  return 1
  
  '''
  if (class1 == 1 and class2 == 1):
    return 2
  if (class1 == 1 and class3 == 1):
    return 3
  if (class2 == 1):
    return 2
  if (class3 == 1):
    return 3
  return 1
  '''

def checkAccuracy(prediction, classLabels):
  if (prediction == classLabels):
    return 1
  else:
    return 0

def test(weights, data):
  inputData = data[0:, 0:-1]
  classLabels = data[0:, -1:]
  class1 = np.array(list(map(predict, np.full((data.shape[0], 8), weights[0]), inputData)))
  class2 = np.array(list(map(predict, np.full((data.shape[0], 8), weights[1]), inputData)))
  class3 = np.array(list(map(predict, np.full((data.shape[0], 8), weights[2]), inputData)))

  truePositivesMatrix = checkAccuracy(predictionDecider(class1, class2, class3), classLabels.flatten())
  accuracy = np.sum(truePositivesMatrix) / truePositivesMatrix.shape[0]
  print(accuracy)


checkAccuracy = np.vectorize(checkAccuracy)
predictionDecider = np.vectorize(predictionDecider)
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