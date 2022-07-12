import numpy as np
import matplotlib.pyplot as plt

f = open('json.rj', 'r')
data = f.readlines()

avgs = []
maxs = []
mins = []

xs = []

gen = 1

for d in data:
    
    d = d.replace('\n', '')
    d = d.split(',')
    
    avgs.append(float(d[0]))
    maxs.append(float(d[1]))
    mins.append(float(d[2]))
    
    xs.append(gen)
    gen += 1
    
plt.plot(xs, avgs, label='Avg')
plt.plot(xs, maxs, label='Max')
plt.plot(xs, mins, label='Min')
plt.legend()
plt.show()
