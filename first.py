import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

X = mx.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = mx.array([[0.0], [1.0], [1.0], [0.0]])


class XORnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)

    def __call__(self, x):
        x = self.layer1(x)
        x = nn.relu(x)
        x = self.layer2(x)
        x = nn.sigmoid(x)
        return x


model = XORnn()
optimizer = optim.Adam(learning_rate=0.01)


def lossfunction(model, x, y):
    predictions = model(x)
    return mx.mean(mx.square(predictions - y))


loss_and_grad_fn = nn.value_and_grad(model, lossfunction)


def step(model, optimizer, x, y):
    loss, grads = loss_and_grad_fn(model, x, y)
    optimizer.update(model, grads)
    return loss


# 6. Run the Training Loop
print("Starting training...")
mx.eval(model.parameters(), optimizer.state)

for epoch in range(500):
    loss = step(model, optimizer, X, Y)
    mx.eval(model.parameters(), optimizer.state)  # Force GPU to execute

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1} | Loss: {loss.item():.4f}")

# 7. See the final results
print("\nFinal Predictions (Should be close to 0, 1, 1, 0):")
print(model(X))
