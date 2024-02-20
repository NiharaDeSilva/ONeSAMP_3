import torch
import torch.nn as nn
from skorch import NeuralNetRegressor


class CustomFNN(nn.Module):
    def __init__(self, dropout_rate=0.5, weight_constraint=None):
        super(CustomFNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.weight_constraint = weight_constraint

        self.layer1 = nn.Linear(5, 20)
        self.layer2 = nn.Linear(20, 10)
        self.layer3 = nn.Linear(10, 5)
        self.output = nn.Linear(5, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X):
        X = torch.relu(self.layer1(X))
        X = self.dropout(X)
        X = torch.relu(self.layer2(X))
        X = self.dropout(X)
        X = torch.relu(self.layer3(X))
        X = self.dropout(X)
        X = self.output(X)
        return X

    def apply_weight_constraint(self):
        # Apply weight constraint to each layer if needed
        if self.weight_constraint is not None:
            for layer in [self.layer1, self.layer2, self.layer3, self.output]:
                nn.utils.clip_grad_norm_(layer.parameters(), self.weight_constraint)

# Usage with Skorch and GridSearchCV
model = NeuralNetRegressor(
    module=CustomFNN,
    criterion=nn.MSELoss,
    optimizer=torch.optim.Adam,
    lr=0.001,
    max_epochs=10,
    batch_size=64,
    verbose=0
)

param_grid = {
    'module__weight_constraint': [1.0, 2.0, 3.0],
    'module__dropout_rate': [0.0, 0.2, 0.5]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)  # Assuming X_train and y_train are defined
