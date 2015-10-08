

# labelBias: [[float]], [int], int, obj -> float
# compute the signed bias of a set of labels on a given dataset
def labelBias(data, labels, protectedIndex, protectedValue):
   protectedClass = [(x,l) for (x,l) in zip(data, labels)
      if x[protectedIndex] == protectedValue]
   elseClass = [(x,l) for (x,l) in zip(data, labels)
      if x[protectedIndex] != protectedValue]

   if len(protectedClass) == 0:
      raise Exception("Nobody in the protected class")
   elif len(elseClass) == 0:
      raise Exception("Nobody in the unprotected class")
   else:
      protectedProb = sum(1 for (x,l) in protectedClass if l == 1) / len(protectedClass)
      elseProb = sum(1 for (x,l) in elseClass  if l == 1) / len(elseClass)

   return elseProb - protectedProb


# signedBias: [[float]], int, obj, h -> float
# compute the signed bias of a hypothesis on a given dataset
def signedBias(data, h, protectedIndex, protectedValue):
   return labelBias(data, [h(x) for x in data], protectedIndex, protectedValue)



if __name__ == "__main__":
   from data import adult
   train, test = adult.load(separatePointsAndLabels=True)

   tests = [('female', (1,0)),
            ('private employment', (2,1)),
            ('asian race', (33,1)),
            ('divorced', (12, 1))]

   for (name, (index, value)) in tests:
      print("anti-'%s' bias in training data: %.4f" %
         (name, labelBias(train[0], train[1], index, value)))


   indian = lambda x: x[47] == 1
   print(len([x for x in train[0] if indian(x)]) / len(train[0]))
   print(signedBias(train[0], indian, 1, 0))
