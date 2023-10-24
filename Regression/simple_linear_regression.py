import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
import pandas as pd


df = pd.read_csv('dataset/cars.csv')[:200]

cdf = df[['km_driven','selling_price','year']]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.scatter(cdf.km_driven, cdf.selling_price,  color='blue')
ax1.set_xlabel("km_driven")
ax1.set_ylabel("selling_price")

#regression
regr = linear_model.LinearRegression()
train_x = np.asanyarray(cdf[['km_driven']][:150])
train_y = np.asanyarray(cdf[['selling_price']][:150])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


ax2.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
ax2.set_xlabel("km_driven")
ax2.set_ylabel("selling_price")



ax3.scatter(cdf.km_driven, cdf.selling_price,  color='blue')
ax3.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
ax3.set_xlabel("km_driven")
ax3.set_ylabel("selling_price")

test_x = np.asanyarray(cdf[['km_driven']][150:])
test_y = np.asanyarray(cdf[['selling_price']][150:])
test_y_ = regr.predict(test_x)

print(f"Mean absolute error: {np.mean(np.absolute(test_y_ - test_y))}" )
print(f"Residual sum of squares (MSE): {np.mean((test_y_ - test_y) ** 2)}")
print(f"R2-score: {r2_score(test_y , test_y_)}")

plt.tight_layout()
plt.show()



