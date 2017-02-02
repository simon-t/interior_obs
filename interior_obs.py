# coding: utf-8
import numpy as np
import random
import time
from sklearn import svm

# valid weights are in the range [2.0, 20.0) kg
weightT = np.arange(2., 20., 0.1)
# range of invalid weights: [20.0, 120.0) kg
weightF = np.arange(20., 120., 0.1)

# valid temperature range: [35.5, 40.0) deg C
tempT = np.arange(35.5, 40., 0.1)
#invalid temperature range: [20.0, 35.5) and [40.0, 100.0) deg C
tempF = np.concatenate((np.arange(20., 35.5, 0.1), np.arange(40.,100.,0.1)), axis=0)

# create pairs of temperature, weight measurements for training
inpF1 = [(random.choice(tempF), random.choice(weightT)) for i in range(50)]
inpF2 = [(random.choice(tempT), random.choice(weightF)) for i in range(50)]
inpF3 = [(random.choice(tempF), random.choice(weightF)) for i in range(50)]
inpT = [(random.choice(tempT), random.choice(weightT)) for i in range(50)]
inp = inpF1 + inpF2 + inpF3 + inpT
# labels: 0: "no child", 1: "child" 
labels = [0 for i in range(150)] + [1 for i in range(50)]

# create a support vector classifier and fit the training data
classifier = svm.SVC()
classifier.fit(inp, labels)

while True:
    t, w = random.uniform(20., 100.), random.uniform(2., 120.)
    m = classifier.predict([(t, w)])[0]
    print('{} deg C, {} kg => {}child'.format(t, w, '' if m else 'not '))
    time.sleep(3)

