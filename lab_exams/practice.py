import numpy as np
arr1 = np.arange(10)
print(arr1)
arr2 = np.zeros(10)
arr2 = np.array([int(x) for x in arr2])
print(arr2)


arr2 = np.random.randint(0,100,size=12)
print(arr2)

arr2 = np.random.rand(3,100)
print(arr2)

arr2 = np.array([1,2,3,4,5,6])
print(arr2)


reshaped1 = arr2.reshape(3, 2)  # Convert 1D to 3x4
print(reshaped1)
