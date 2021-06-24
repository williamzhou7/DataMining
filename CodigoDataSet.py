import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

df = pd.read_csv(url, header = None)
df.head(5)
df.tail(5)

##Definindo os cabeçalhos
headers=["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels",
        "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type", "num-of-cylinders",
         "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg",
         "highway-mpg", "price"]

df.columns = headers

#print(df.dtypes)

##Exportando para csv

path = "/Users/ShreyamDuttaGupta/Desktop/automobile-data-set-analysis/cars.csv"
df.to_csv(path)


##Gerando status descritivos

print(df.describe(include="all"))
print(df.info)


##Data Formatting, replacing ? para NAN

df["price"].replace('?',np.nan, inplace = True)
path = "/Users/ShreyamDuttaGupta/Desktop/automobile-data-set-analysis/cars.csv"
df.to_csv(path)


##Data Formatting, convertendo prices de Object para Int, droppando valores NaN


df["price"].replace('?',np.nan, inplace = True)
df.dropna(subset=["price"], axis=0, inplace=True)
df["price"] = df["price"].astype("int")

##Data Formatting, convertendo peak-rpm de Object para Int, droppando valores NaN

df["peak-rpm"].replace('?',np.nan, inplace = True)
df.dropna(subset=["peak-rpm"], axis=0, inplace=True)
df["peak-rpm"] = df["peak-rpm"].astype("int")

print(df.info)


###Data Binning

binwidth = int((max(df["price"])-min(df["price"]))/3)
bins = range(min(df["price"]), max(df["price"]), binwidth)
group_names = ['low','medium','high']
df["price-binned"] = pd.cut(df["price"], bins, labels=group_names)
path = "/Users/ShreyamDuttaGupta/Desktop/automobile-data-set-analysis/cars.csv"
df.to_csv(path)
df.dropna(subset=["price-binned"], axis=0, inplace=True)


##Plottando Histogram do valor binned


plt.hist(df["price"],bins=3)
plt.title("Price Bins")
plt.xlabel("Count")
plt.ylabel("Price")
plt.show()



#TRANSFORMANDO VARIÁVEIS DE CATEGORIA EM VARIÁVEIS QUANTITATIVAS

df = (pd.get_dummies(df["fuel-type"]))


#ESTATÍSTICAS DESCRITIVAS- Value_counts

drive_wheels_counts = df["drive-wheels"].value_counts()
drive_wheels_counts.rename(columns={'drive-wheels':'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
print(drive_wheels_counts)


#Box Plots


sns.boxplot(x="drive-wheels", y="price", data=df)
plt.show()

#Scatterplot
y=df["engine-size"]
x=df["price"]
plt.scatter(x,y)
plt.title("Scatterplot of Engine Size vs Price")
plt.xlabel("Engine Size")
plt.ylabel("Price")
plt.show()
#Agrupe para visualizar o preço com base nas rodas motrizes e estilo da carroceria.
df_test = df[["drive-wheels", "body-style", "price"]]
df_group = df_test.groupby(['drive-wheels', 'body-style'], as_index = False).mean()
#Tabela dinâmica para visualizar o preço com base nas rodas motrizes e estilo da carroceria.
df_pivot = df_group.pivot(index = 'drive-wheels', columns= 'body-style')
print(df_pivot)
#Heat Maps
plt.pcolor(df_pivot, cmap='RdBu')
plt.colorbar()
plt.show()
#CORRELAÇÃO, relação linear positiva entre tamanho do motor e preço
sns.regplot(x='engine-size', y='price', data=df)
plt.title("Scatterplot of Engine Size vs Price")
plt.xlabel("Engine Size")
plt.ylabel("Price")
plt.ylim(0,)
plt.show()
#CORRELAÇÃO, Relação Linear Negetiva entre highway-mpg e preço
sns.regplot(x='highway-mpg', y='price', data=df)
plt.title("Scatterplot of highway-mpg vs price")
plt.xlabel("highway-mpg")
plt.ylabel("price")
plt.ylim(0,)
plt.show()
# CORRELAÇÃO FRACA entre pico-rpm e preço
sns.regplot(x='peak-rpm', y='price', data=df)
plt.title("Scatterplot of peak-rpm vs price")
plt.xlabel("peak-rpm")
plt.ylabel("price")
plt.ylim(0,)
plt.show()


#Estimador de modelo linear simples com gráfico de distribuição


lm = LinearRegression()
X=df[["highway-mpg"]]
Y=df["price"]
lm.fit(X,Y)
Yhat1 = lm.predict(X)
b0 = lm.intercept_
b1 = lm.coef_
estimated = b0 + b1*X

ax1 = sns.distplot(df["price"],hist = False, color="r", label="Actual Value")
sns.distplot(Yhat1, hist = False, color="b", label="Fitted Value", ax=ax1)
plt.ylim(0,)
plt.show()



#Regressão linear múltipla com gráfico de distribuição

lm = LinearRegression()
Z = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
Y=df["price"]
lm.fit(Z,Y)
Y=df["price"]
Yhat2 = lm.predict(Z)
ax1 = sns.distplot(df["price"],hist = False, color="r", label="Actual Value")
sns.distplot(Yhat2, hist = False, color="b", label="Fitted Value", ax=ax1)
plt.ylim(0,)
plt.show()


#Residual Plot

sns.residplot(df["highway-mpg"], df["price"])
plt.xlabel("highway-mpg")
plt.ylabel("price")
plt.show()
