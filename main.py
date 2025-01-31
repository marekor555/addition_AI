import os

import sklearn as sk
import numpy as np
import random

n = 10000

X = np.array([[random.randint(0, n), random.randint(0, n)] for i in range(n)])
Y = np.array([X[i][0] + X[i][1] for i in range(n)])


print("Starting learning...")
model = sk.linear_model.LinearRegression()
model.fit(X, Y)

testedCnt = 0
successCnt = 0
for i in range(20):
    testVal = [random.randint(100, 200), random.randint(100, 200)]
    prediction = model.predict([testVal])
    if round(prediction[0]) == float(testVal[0] + testVal[1]):
        successCnt += 1
        print("Guess: ", testVal[0], "+", testVal[1], "=", prediction[0], ", should be: ", testVal[0] + testVal[1])
    else:
        print("NO! ", testVal[0], "+", testVal[1], "=", testVal[0] + testVal[1], " not ", prediction[0])
    testedCnt += 1

print("Successes:", successCnt)
input("Press Enter to continue...")
os.system("clear")

while True:
    num = int(input("Enter a number: "))
    num2 = int(input("Enter another number: "))
    prediction = model.predict([[num, num2]])
    print("Prediction: ", round(prediction[0]))
    input("Press Enter to continue...")
    os.system("clear")
