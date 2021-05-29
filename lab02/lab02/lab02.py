import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

dates = pd.date_range('2020-03-01', periods=5, freq='D')
df = pd.DataFrame({ 'Date': dates, 'A' : np.random.randn(len(dates)), 'B' : np.random.randn(len(dates)), 'C' : np.random.randn(len(dates))})

df=pd.DataFrame(np.random.randn(20, 3), index=np.arange(20), columns=['A','B','C'])
print(df.sample(5))
print(df.loc[:, ['A']])
print(df.loc[:, ['A', 'B']])

print(df.iloc[0:3,0:2])
print(df.mean(axis=1))

C = np.transpose(C)

'Apple', 'Banana', 'Banana', 'Apple'], 'Pound': [10, 15, 50, 40, 5], 'Profit':[20, 30, 25, 20, 10]}
df3 = pd.DataFrame(slownik)
print(df3)
print(df3.groupby('Day').sum())
print(df3.groupby(['Day','Fruit']).sum())
df.index.name='id'
print(df)
#print(df)
#df.fillna(0, inplace=True) - zamienia notanumber na 0
#print(df)
#df.iloc[[0, 3], 1] = np.nan
#df=df.replace(to_replace=np.nan,value=-9999) - zamienia notanumber na -9999
#print(df)
#df.iloc[[0, 3], 1] = np.nan
#print(pd.isnull(df)) - sprawdza czy wartosc jest nullem, w miejscu nan mamy true
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([autos['width'], autos['length']])
kernel = st.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)

fig = plt.figure()
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
cfset = ax.contourf(xx, yy, f, cmap='Blues')
cset = ax.contour(xx, yy, f, colors='k')
ax.clabel(cset, inline=1, fontsize=8)
ax.set_xlabel('Szerokość samochodu')
ax.set_ylabel('Długość samochodu')

plt.savefig('out.png')
plt.savefig('out.pdf')

plt.show()