import numpy as np

a = a=np.array([1,2,3,4,5,6,7])
b= np.array([[1,2,3,4,5], [6,7,8,9,10]])
np.transpose(b)
c = np.arange(0, 100, 1)
print(c)
d = np.linspace(0, 2, 10)
print(d)
e = np.arange(0, 101, 5)
f = np.random.randn(20).round(2)
g = np.random.randint(1, 1000, [20, 5])
h = np.zeros([3, 2])
g = np.ones([3,2])
i = np.random.randint(20, 300, [5, 5])

a = np.random.rand(10) * 10
b = a.astype(int)
print(b)
a = np.rint(a)
a = a.astype(int)
print(a)


b=np.array([[1,2,3,4,5], [6,7,8,9,10]],dtype=np.int32)print(np.ndim(b))
print(np.size(b))
print(b[0, 1], b[0, 3])
print(b[0, :])
print(b[:, 0])
c = np.random.randint(0, 100, [20, 7])
print(c[:, 0:4])

a = np.random.rand(3, 3) * 10
b = np.random.rand(3, 3) * 10
c = np.add(a, b)
d = np.matmul(a, b)
aa = np.matmul(a, a)
bb = np.matmul(b, b)

if np.linalg.det(a) >= 4:
    print('Wyznacznik jest większy lub równy 4')

np.trace(b)
np.sum(b)
np.amin(b)
np.amax(b)
np.std(b)
np.mean(b, 1)
np.mean(b, 0)


d = np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]])
np.reshape(d, [10, 5])
np.resize(d, [25, 2])
e = np.ravel(d)
print(e)
x = np.random.randint(1, 100, [5,])
y = np.random.randint(1, 100, [4,])
x_new = x[:, np.newaxis]
print(np.ndim(x_new))
z = x_new+y
print(z)


a=np.random.randn(5,5)
a_sorted = np.sort(a)
print(a_sorted)
a_sorted = np.sort(a, 0)[::-1]
print(a_sorted)


b=np.array([(1,'MZ','mazowieckie'),(2,'ZP','zachodniopomorskie'),(3,'ML','małopolskie')])sorted_array = b[np.argsort(b[:, 1])]
print(sorted_array[2][2])
#zad1
a = np.random.randint(1, 100, [10, 5])
print(np.trace(a))
print(np.diag(a))

#zad2
a = np.random.normal([4,4])
b = np.random.normal([4,4])
np.matmul(a,b)

#zad3
a = np.random.randint(1, 100, [5,1])
b = np.random.randint(1, 100, [10,1])
a = np.reshape(a, [1, 5])
b = np.reshape(b, [2, 5])
np.add(a,b)

#zad4
a = np.random.randint(1, 100, [4,5])
b = np.random.randint(1, 100, [5,4])
a = np.transpose(a)
np.add(a,b)

#zad5
a = np.random.randint(1, 10, [3, 6])
b = np.random.randint(1, 10, [3, 5])
print(np.matmul(a[:, 2], b[:, 3]))

#zad6
a = np.random.normal([4,4])
b = np.random.normal([4,4])
c = np.random.uniform([4,4])
d = np.random.uniform([4,4])
a.mean()
b.mean()
c.mean()
d.mean()
a.std()
b.std()
c.std()
d.std()
a.var()
b.var()
c.var()
d.var()

#zad7
a = np.random.randint(1, 10, [4,4])
b = np.random.randint(1, 10, [4,4])
print(a*b)
print(a.dot(b))
#funkcja dot to iloczyn skalarny, a a*b to mnożenie macierzy

#zad8
print(np.lib.stride_tricks.as_strided(a, [2, 3], (16, 4)))

#zad9
print(np.vstack((a,b)))
print(np.hstack((a,b)))
#vstack łączy pionowo, a hstack poziomo

#zad10
a = np.array([[0,1,2,3,4,5,],[6,7,8,9,10,11],[12,13,14,15,16,17], [18,19,20,21,22,23]])
print(np.max(np.lib.stride_tricks.as_strided(a,(2,3),strides=(24,4))))

