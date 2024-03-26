# %%
import numpy as np
import matplotlib.pyplot as plt
import csv

# %%
def get_data(path: str) -> tuple:
    x = []
    y = []
    with open(path) as fs:
        reader = csv.reader(fs, delimiter=',')
        for row in reader:
            try:
                x.append(float(row[1]))
                y.append(float(row[2]))
            except:
                ...
    return np.array(x), np.array(y)

# %%
def gradient_descent(x, y, b_0, b_1, l=0.01):
    y_pred = b_0 + b_1 * x
    n = len(x)
    D_b_0 = (-2 / n) * sum(y - y_pred)
    D_b_1 = (-2 / n) * sum(x * (y - y_pred))
    b_0 = b_0 - l * D_b_0
    b_1 = b_1 - l * D_b_1
    return b_0, b_1

# %%
train_x, train_y = get_data('lab_1_train.csv')
test_x, test_y = get_data('lab_1_test.csv')

plt.scatter(train_x, train_y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# %%
b_0, b_1 = 0, 0

n = len(train_x)
epochs = 2000
l = 0.1

print("{0:^10.5}{1:^10.5}{2:^10.5}{3:^10.5}".format('b_0', 'b_1', 'eps', 'epoch'))
for e in range(epochs):
    b_0, b_1 = gradient_descent(train_x, train_y, b_0, b_1, l)
    eps = sum((train_y - (b_0 + b_1 * train_x)) ** 2) / n
    if e % 100 == 0:
        print(f"{b_0:^10.5}{b_1:^10.5}{eps:^10.5}{e:^10}")
print(f"{b_0:^10.5}{b_1:^10.5}{eps:^10.5}{epochs:^10}")

### 1
plt.scatter(train_x, train_y, marker='x', label='Train data')
### 2
plt.scatter(test_x, test_y, marker='o', label='Test data')

### 3
X = np.concatenate((train_x, test_x))
Y = b_0 + b_1 * X
plt.plot(X, Y, ':', label='Predicted values')

### 4
pred_y = b_0 + b_1 * test_x
plt.plot(test_x, pred_y, 'g-', label='Predicted values')

plt.title('Result')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# %% [markdown]
# 


