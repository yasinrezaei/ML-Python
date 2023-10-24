import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
import pandas as pd



df = pd.read_csv('dataset/cars.csv')[:200]
cdf = df[['km_driven','selling_price','year']]

regr = linear_model.LinearRegression()
train_x = np.asanyarray(cdf[['year','km_driven']][:150])
train_y = np.asanyarray(cdf[['selling_price']][:150])
regr.fit (train_x, train_y)

print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

fig = plt.figure(figsize=(10, 6))
ax2 = fig.add_subplot(111, projection='3d')
ax2.scatter(cdf['year'][:150], cdf['km_driven'][:150], cdf['selling_price'][:150], color='blue', marker='o', alpha=0.5)
ax2.set_xlabel('Year')
ax2.set_ylabel('KM Driven')
ax2.set_zlabel('Selling Price')

# Create a meshgrid for the plane
xx, yy = np.meshgrid(cdf['year'][:150], cdf['km_driven'][:150])
zz = regr.coef_[0][0] * xx + regr.coef_[0][1] * yy + regr.intercept_[0]


test_x = np.asanyarray(cdf[['year','km_driven']][150:])
test_y = np.asanyarray(cdf[['selling_price']][150:])
test_y_ = regr.predict(test_x)

print(f"Mean absolute error: {np.mean(np.absolute(test_y_ - test_y))}" )
print(f"Residual sum of squares (MSE): {np.mean((test_y_ - test_y) ** 2)}")
print(f"R2-score: {r2_score(test_y , test_y_)}")


ax2.plot_surface(xx, yy, zz, color='red', alpha=0.5)
plt.show()

