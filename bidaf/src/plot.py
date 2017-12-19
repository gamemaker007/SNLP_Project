import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# with open('snlp_bidaf_log1.out') as fin:
# 	with open('snlp_bidaf_log1_.out','w') as fout:
# 		for line in fin:
# 			if line.startswith('INFO'):
# 				fout.write(line)
# 			if "NEW RUN" in line:
# 				fout.write(line)


F1 = [37.429,57.923, 62.315, 63.421, 64.062, 64.176, 64.176, 64.149, 63.628, 63.450 ]

EM = [26.565, 42.677, 45.818, 47.000, 47.530, 47.994,47.9 , 48.079, 47.417, 47.294]

V_L = [4.850, 3.353, 3.012, 2.950, 2.919, 2.972,3.019 , 3.095, 3.233, 3.395]

T_L = [5.857, 3.930, 3.024, 2.621, 2.344, 2.108,1.916 , 1.751, 1.613, 1.488]

plt.figure(1)
plt.plot(F1)
plt.plot(EM)
plt.legend(['F1', 'EM'], loc='upper right')
plt.xlabel('Epochs')
plt.show()
plt.savefig('F1vsEM.png')

plt.figure(2)
plt.plot(T_L)
plt.plot(V_L)
plt.legend(['Training Loss', 'validation Loss'], loc='upper right')
plt.xlabel('Epochs')
plt.show()
plt.savefig('TlvsVl.png')

