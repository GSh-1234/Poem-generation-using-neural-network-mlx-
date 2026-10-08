import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Generate 100 records
np.random.seed(42)
gpa = np.random.uniform(4.0, 10.0, 100)
hours = np.random.uniform(0.0, 20.0, 100)
attendance = np.random.uniform(50.0, 100.0, 100)
X_np = np.column_stack((gpa, hours, attendance))

Y_np = (
    20
    + (gpa * 4.0)
    + (hours * 1.5)
    + (attendance * 0.1)
    + np.random.normal(0, 1.0, 100)
)
Y_np = np.clip(Y_np, 0, 100).reshape(-1, 1)

# 2. split the data
x_train, x_test, y_train, y_test = train_test_split(
    X_np, Y_np, test_size=0.2, random_state=42
)

# 3. normalize

scalar = StandardScaler()

x_trains = scalar.fit_transform(x_train)
x_tests = scalar.transform(x_test)

# (We keep Y simple by just dividing by 100)
y_train_scaled = y_train / 100.0
y_test_scaled = y_test / 100.0

# 4. conver to mlx array

x_train_mx = mx.array(x_trains, dtype=mx.float32)
y_train_mx = mx.array(y_train_scaled, dtype=mx.float32)

x_test_mx = mx.array(x_tests, dtype=mx.float32)
y_test_mx = mx.array(y_test, dtype=mx.float32)

# 5. model architecture


class markpredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 16)
        self.layer5 = nn.Linear(16, 1)

    def __call__(self, x):
        x = self.layer1(x)
        x = nn.relu(x)
        x = self.layer2(x)
        x = nn.relu(x)
        x = self.layer3(x)
        x = nn.relu(x)

        x = self.layer4(x)
        x = nn.relu(x)

        # Final layer remains raw for regression
        x = self.layer5(x)
        return x


model = markpredictor()
optimizer = optim.Adam(learning_rate=0.01)


def loss_fn(model, x, y):
    return mx.mean(mx.square(model(x) - y))


loss_and_grad_fn = nn.value_and_grad(model, loss_fn)


def step(model, optimizer, x, y):
    loss, grads = loss_and_grad_fn(model, x, y)
    optimizer.update(model, grads)
    return loss


mx.eval(model.parameters(), optimizer.state)

for epoch in range(1000):
    # ONLY train on the 80 records in the training set
    loss = step(model, optimizer, x_train_mx, y_train_mx)
    mx.eval(model.parameters(), optimizer.state)

# Calculate the final loss on data the model has never seen
test_loss = loss_fn(model, x_test_mx, y_test_mx)
print(f"\nFinal Test Loss: {test_loss.item():.4f}")

# Make a prediction for the first student in the test set
predicted_mark = model(x_test_mx[0:1]).item() * 100.0
actual_mark = y_test[0][0]

print(f"Student 1 Predicted Mark: {predicted_mark:.1f}")
print(f"Student 1 Actual Mark:    {actual_mark:.1f}")
