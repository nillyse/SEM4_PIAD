import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

dates = pd.date_range('2020-03-01', periods=5, freq='D')
df = pd.DataFrame({ 'Date': dates, 'A' : np.random.randn(len(dates)), 'B' : np.random.randn(len(dates)), 'C' : np.random.randn(len(dates))})

df=pd.DataFrame(np.random.randn(20, 3), index=np.arange(20), columns=['A','B','C'])print(df.loc[0:2])print(df.iloc[-3:])print(df.columns)print(df.to_string(index=False, header=False))
print(df.sample(5))
print(df.loc[:, ['A']])
print(df.loc[:, ['A', 'B']])

print(df.iloc[0:3,0:2])print(df.iloc[4, :])print(df.iloc[[0,5,6,7], [1,2]])df.describe()print(df>0)print(df[df>0])print(df.loc[:, 'A'][df.loc[:, 'A'] > 0])print(df.mean(axis=0))
print(df.mean(axis=1))
A = {'D' : np.random.randn(len(dates))}C = pd.concat([df, pd.DataFrame(A)])
C = np.transpose(C)
df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': ['a', 'b', 'a', 'b', 'b']}, index=np.arange(5))df.index.name='id'print(pd.DataFrame.sort_index)print(df.sort_values(['y'], ascending=1))slownik = {'Day': ['Mon', 'Tue', 'Mon', 'Tue', 'Mon'], 'Fruit': ['Apple',
'Apple', 'Banana', 'Banana', 'Apple'], 'Pound': [10, 15, 50, 40, 5], 'Profit':[20, 30, 25, 20, 10]}
df3 = pd.DataFrame(slownik)
print(df3)
print(df3.groupby('Day').sum())
print(df3.groupby(['Day','Fruit']).sum())df=pd.DataFrame(np.random.randn(20, 3), index=np.arange(20), columns=['A','B','C'])
df.index.name='id'
print(df)#df['B']=1 - zamienia całą kolumne B na wartość 1#print(df) - wyświetla całą macierz z nagłówkiem i indeksami#df.iloc[1,2]=10 - zamienia wartosc 1 wiersza, 2 kolumny na 10#df[df¡0]=-df - nic nie robi, bo nie dziala#df.iloc[[0, 3], 1] = np.nan - zamienia na NotaNumber
#print(df)
#df.fillna(0, inplace=True) - zamienia notanumber na 0
#print(df)
#df.iloc[[0, 3], 1] = np.nan
#df=df.replace(to_replace=np.nan,value=-9999) - zamienia notanumber na -9999
#print(df)
#df.iloc[[0, 3], 1] = np.nan
#print(pd.isnull(df)) - sprawdza czy wartosc jest nullem, w miejscu nan mamy truedf = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': ['a', 'b', 'a', 'b', 'b']})#1print(df.groupby('y').mean())#2print(df['x'].value_counts())print(df['y'].value_counts())#3#np.loadtxt(a,'autos.csv')autos = pd.read_csv('autos.csv')print(autos)#4print(autos.groupby('make').mean())#5print(autos.groupby('make')['fuel-type'].value_counts())#6a = np.polyfit(autos['length'], autos['city-mpg'], 1)b = np.polyfit(autos['length'], autos['city-mpg'], 2)#7print(st.pearsonr(autos['length'], autos['city-mpg']))#8c = np.linspace(autos['length'].min(), autos['length'].max(), 100)plt.scatter(autos['length'], autos['city-mpg'], label = 'proopki')plt.xlabel('Dlugosc')plt.ylabel('Paliwo')plt.plot(c, np.polyval(a, c), label = 'x')plt.plot(c, np.polyval(b, c), label = 'x do 2')plt.legend()#9gauss = st.gaussian_kde(autos['length'])plt.figure()plt.plot(c, gauss(c), label = 'Estymator')plt.scatter(autos['length'], gauss(autos['length']), label = 'proopki')plt.legend()#10plt.figure()ax = plt.subplot(1, 2, 1)ax.plot(c, gauss(c), label = 'Estymator')ax.scatter(autos['length'], gauss(autos['length']), label = 'proopki')ax.set_title("Długość samochodu")plt.legend()szer_c = np.linspace(autos['width'].min(), autos['width'].max(), 500)szer_gauss = st.gaussian_kde(autos['width'])ax2 = plt.subplot(1, 2, 2)ax2.plot(szer_c, szer_gauss(szer_c), label = 'Estymator')ax2.scatter(autos['width'], szer_gauss(autos['width']), label = 'proopki')ax2.set_title("Szerokość samochodu")plt.legend()#11#https://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-pythonxmin, xmax, ymin, ymax = autos['width'].min(), autos['width'].max(), autos['length'].min(), autos['length'].max()xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
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