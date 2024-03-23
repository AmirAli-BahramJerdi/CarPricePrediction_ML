import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

mu = 0
w=0
degree = 3
df = pd.read_excel('car_price.xlsx')


X = df['Age'].values
y = df['Price'].values

def plot_data(X=X , y=y):
    plt.scatter(X,y,s=20)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

plot_data()

def data_split(X, y, test_size=0.25 , random_state = 42):
    x1 = []
    y1 = []
    shuffle = np.random.permutation(len(X))

    for i in shuffle:
        x1.append(X[i])
        y1.append(y[i])

    x1 = np.array(x1)
    y1 = np.array(y1)

    sp = int(len(X)* (1-test_size))
    x_train = x1[:sp]

    x_test = x1[sp:]
    y_train = y1[:sp]
    y_test = y1[sp:]
    return x_train , x_test , y_train , y_test

def poly(x,degree):
    x = x.reshape(-1,1)
    for i in range(2,degree+1):
        x = np.c_[x, x[:,0]**i]

def normalize(x=X):
    global mu, s
    mu = x.mean(axis=0)
    s = np.std(x,axis=0)
    return (x-mu)/s
def fit(X, y): #, X1, y1):
    m = len(X)

    tolerance = 1e-15
    alpha = 0.2

    y = y.reshape(-1,1)
    X = poly(X, degree)
    X = normalize(X)
    X = np.c_[np.ones(len(X)),X]
    
    w = np.zeros((X.shape[1],1))

    for epoch in range(1000000):
        y_hat = X.dot(w)
        J = ((y_hat - y) ** 2).sum()/(2*m)
        g = X.T.dot(y_hat-y)/m

        w -= alpha*g

        if alpha*np.abs(g.mean())<=tolerance:
            print('epoch = ',epoch,'\tGradient = ',alpha*np.abs(g.mean()))
            break
    return w


def fit_neq(X, y):
    X = poly(X, degree)
    X = normalize(X)
    X = np.c_[np.ones(len(X)),X]
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return w       
def predict(x):
    global mu, s
    x = np.array(x)
    x = poly(x, degree)
    x = (x-mu)/s
    x = np.c_[np.ones(len(x)),x]
    return x.dot(w)

X_train,X_test,y_train,y_test=data_split(X, y, test_size=0.25, random_state=8)
#w = fit_neq(X_train, y_train)

def regression_khati(X, y):
    n = len(X)
    sum_x = np.sum(X)
    sum_y = np.sum(y)
    sum_xy = np.sum(X * y)
    sum_x_tavan2 = np.sum(X**2)

    shib = (n * sum_xy - sum_x * sum_y) / (n * sum_x_tavan2 - sum_x**2)
    taghato = (sum_y - shib * sum_x) / n
    return shib, taghato

shib , taghato  = regression_khati(X, y)

def regression_chandjomle(X, y, degree=2):
    n = len(X)
    X_yek_jomle = np.column_stack([X**i for i in range(1, degree+1)])
    zarayeb = np.linalg.inv(X_yek_jomle.T.dot(X_yek_jomle)).dot(X_yek_jomle.T).dot(y)
    return zarayeb

# zarayeb chand jomleii
zarib_2jomle = regression_chandjomle(X, y, degree=2)
zarib_3jomle = regression_chandjomle(X, y, degree=3)
zarib_4jomle = regression_chandjomle(X, y, degree=4)

# Regression khati
plt.plot(X, shib * X + taghato, label='Regrresion khati', color='red')

# Regression daragee 2
X_fit = np.linspace(np.min(X), np.max(X), 100)
X_darage2_fit = np.column_stack([X_fit**i for i in range(1, 3)])
y_darage2_fit = X_darage2_fit.dot(zarib_2jomle)
plt.plot(X_fit, y_darage2_fit, label='Regression chand jomleii (Degree 2)', color='green')

# Regression daragee 3
X_darage3_fit = np.column_stack([X_fit**i for i in range(1, 4)])
y_darage3_fit = X_darage3_fit.dot(zarib_3jomle)
plt.plot(X_fit, y_darage3_fit, label='Regression chand jomleii (Degree 3)', color='purple')

# Regression daragee 4
X_darage4_fit = np.column_stack([X_fit**i for i in range(1, 5)])
y_darage3_fit = X_darage4_fit.dot(zarib_4jomle)
plt.plot(X_fit, y_darage3_fit, label='Regression chand jomleii (Degree 4)', color='orange')

plt.legend()
plot_data()
plt.show()

#                               !!! TAVAJOOOH !!!!
#  !!! hatman be Adress file Exel tavajooh konid ta dataset ha vared shavand !!!