import os
import time

start = time.time()

a = 0
for i in range(1000):
    a = a + i

end = time.time()
print(end - start)

# path_folder = 'Data/train'

# print(len(os.listdir(path=path_folder)))