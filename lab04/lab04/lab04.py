import numpy as np
import matplotlib.pyplot as plt


#DYSKRETYZACJA
def dyskretyzacja(f, fs):
    t = np.linspace(0,1,fs)
    s = np.sin(2*np.pi*f*t)
    return t,s

f = 10
fs = [20, 21, 30, 45, 50, 100, 150, 200, 250, 1000]

for i,s in enumerate(fs):
    t,s = dyskretyzacja(f, s)
    plt.subplot(5,2,i+1)
    #plt.plot(t,s)
    plt.stem(t,s, use_line_collection=True)
    
#4.Tak, nazywa się prawem Shannona-Nyquista
#5. aliasing

#               7
#A = plt.imread('city.png')
#methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
#           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
#           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

#fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9, 6),
#subplot_kw={'xticks': [], 'yticks': []})


#for ax, interp_method in zip(axs.flat, methods):
#    ax.imshow(A, interpolation=interp_method, cmap='viridis')
#    ax.set_title(str(interp_method))


#plt.tight_layout()
#plt.savefig('cityPlot.png')
#plt.show()

#KWANTYZACJA

A = plt.imread('city.png')
print(str(A.shape))

B = np.zeros(A.shape[0:2])
C = np.zeros(A.shape[0:2])
D = np.zeros(A.shape[0:2])

print(str(B.shape))

B = (np.amax(A, 2) + np.amin(A, 2))/2
C = A.mean(axis=2)
#C = (A[:,:, 0] + A[:,:, 1]+A[:,:, 2])/3
D = 0.21*A[:,:, 0] + 0.72*A[:,:, 1]+0.07*A[:,:, 2]
print(B)


plt.figure()
plt.subplot(1,3,1)
plt.imshow(B, cmap = 'gray')
plt.subplot(1,3,2)
plt.imshow(C, cmap = 'gray')
plt.subplot(1,3,3)
plt.imshow(D, cmap = 'gray')

plt.figure()
plt.subplot(1,3,1)
plt.hist(B.flatten(), 256)
plt.subplot(1,3,2)
plt.hist(C.flatten(), 256)
plt.subplot(1,3,3)
plt.hist(D.flatten(), 256)

plt.figure()
values, bins, x = plt.hist(B.flatten(), 16)
E = np.digitize(B, bins)
F = 0.5*(bins[E[1:] - 1] + bins[E[:-1] - 1])
plt.figure()
plt.imshow(F, cmap = 'gray')
print('a')





#BINARYZACJA

A = plt.imread('circle.png')


def binaryzacja(A):
    B = (np.amax(A, 2) + np.amin(A, 2))/2
    plt.figure()
    values, bins, x = plt.hist(B.copy().flatten(), 2)
    C = B>bins[1]
    return B, C

B,C = binaryzacja(A)
plt.figure()
plt.hist(B.copy().flatten(), 256)
plt.figure()
plt.imshow(C, cmap = 'gray')
plt.show()