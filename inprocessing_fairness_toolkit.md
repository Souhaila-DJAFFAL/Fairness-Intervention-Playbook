                        else:
                            continue  # Skip if no positive examples in either group
                    
                    constraints.append({
                        'constraint_value': constraint,
                        'group_a': group_a['description'],
                        'group_b': group_b['description'],
                        'type': self.fairness_metric
                    })
        
        return constraints
```

## Component 2: Constrained Optimization Methods

### Purpose
Integrate fairness constraints into standard ML optimization procedures.

### 2.1 Lagrangian Optimization

```python
class LagrangianFairOptimizer:
    """Optimize ML models with fairness constraints using Lagrangian methods"""
    
    def __init__(self, base_model, fairness_constraints, lambda_fairness=1.0):
        self.base_model = base_model
        self.fairness_constraints = fairness_constraints
        self.lambda_fairness = lambda_fairness
        self.lagrange_multipliers = None
    
    def lagrangian_objective(self, params, X, y, protected_attr):
        """Combined objective: prediction loss + fairness penalty"""
        
        # Set model parameters
        self.base_model.set_params(params)
        
        # Prediction loss
        predictions = self.base_model.predict_proba(X)[:, 1]
        prediction_loss = log_loss(y, predictions)
        
        # Fairness constraints
        constraint_violations = []
        
        if 'demographic_parity' in self.fairness_constraints:
            dp_violations = FairnessConstraints.demographic_parity_constraint(
                predictions, protected_attr
            )
            constraint_violations.extend(dp_violations)
        
        if 'equal_opportunity' in self.fairness_constraints:
            eo_violations = FairnessConstraints.equal_opportunity_constraint(
                predictions, y, protected_attr
            )
            constraint_violations.extend(eo_violations)
        
        # Penalty for constraint violations
        fairness_penalty = sum(max(0, violation) for violation in constraint_violations)
        
        # Combined objective
        total_loss = prediction_loss + self.lambda_fairness * fairness_penalty
        
        return total_loss
    
    def fit(self, X, y, protected_attr, max_iter=100):
        """Train model with fairness constraints"""
        
        # Initialize parameters
        initial_params = self.base_model.get_params()
        
        # Optimize using scipy
        from scipy.optimize import minimize
        
        result = minimize(
            fun=lambda params: self.lagrangian_objective(params, X, y, protected_attr),
            x0=self._flatten_params(initial_params),
            method='L-BFGS-B',
            options={'maxiter': max_iter}
        )
        
        # Set optimal parameters
        optimal_params = self._unflatten_params(result.x)
        self.base_model.set_params(optimal_params)
        
        return self
    
    def _flatten_params(self, params_dict):
        """Convert parameter dict to flat array"""
        # Implementation depends on model type
        if hasattr(self.base_model, 'coef_'):
            return self.base_model.coef_.flatten()
        else:
            # For more complex models, implement parameter extraction
            raise NotImplementedError("Parameter flattening not implemented for this model type")
    
    def _unflatten_params(self, params_array):
        """Convert flat array back to parameter dict"""
        # Implementation depends on model type
        if hasattr(self.base_model, 'coef_'):
            return {'coef_': params_array.reshape(self.base_model.coef_.shape)}
        else:
            raise NotImplementedError("Parameter unflattening not implemented for this model type")
```

### 2.2 Alternating Direction Method of Multipliers (ADMM)

```python
class ADMMFairOptimizer:
    """ADMM-based optimization for fairness constraints"""
    
    def __init__(self, base_model, fairness_metric='demographic_parity', rho=1.0):
        self.base_model = base_model
        self.fairness_metric = fairness_metric
        self.rho = rho  # Penalty parameter
        self.dual_variables = None
    
    def fit(self, X, y, protected_attr, max_iter=50, tolerance=1e-4):
        """Fit model using ADMM optimization"""
        
        n_samples, n_features = X.shape
        n_groups = len(protected_attr.unique())
        
        # Initialize variables
        theta = np.random.normal(0, 0.01, n_features)  # Model parameters
        z = np.zeros(n_groups)  # Auxiliary variables for fairness constraints
        u = np.zeros(n_groups)  # Dual variables
        
        for iteration in range(max_iter):
            theta_old = theta.copy()
            
            # Update theta (model parameters)
            theta = self._update_theta(X, y, protected_attr, z, u, theta)
            
            # Update z (auxiliary variables)
            z = self._update_z(X, protected_attr, theta, u)
            
            # Update u (dual variables)
            u = self._update_dual_variables(X, protected_attr, theta, z, u)
            
            # Check convergence
            if np.linalg.norm(theta - theta_old) < tolerance:
                break
        
        # Set final model parameters
        if hasattr(self.base_model, 'coef_'):
            self.base_model.coef_ = theta.reshape(self.base_model.coef_.shape)
        
        self.dual_variables = u
        
        return self
    
    def _update_theta(self, X, y, protected_attr, z, u, theta_current):
        """Update model parameters"""
        
        def objective(theta):
            # Prediction loss
            logits = X @ theta
            predictions = 1 / (1 + np.exp(-logits))
            loss = -np.mean(y * np.log(predictions + 1e-8) + 
                           (1 - y) * np.log(1 - predictions + 1e-8))
            
            # ADMM penalty term
            penalty = 0
            for i, group in enumerate(protected_attr.unique()):
                group_mask = protected_attr == group
                group_rate = predictions[group_mask].mean()
                penalty += self.rho / 2 * (group_rate - z[i] + u[i]) ** 2
            
            return loss + penalty
        
        # Optimize using gradient descent
        from scipy.optimize import minimize
        result = minimize(objective, theta_current, method='L-BFGS-B')
        
        return result.x
    
    def _update_z(self, X, protected_attr, theta, u):
        """Update auxiliary variables for fairness constraints"""
        
        logits = X @ theta
        predictions = 1 / (1 + np.exp(-logits))
        
        z_new = np.zeros(len(protected_attr.unique()))
        
        if self.fairness_metric == 'demographic_parity':
            # Project onto fairness constraint manifold
            group_rates = []
            for i, group in enumerate(protected_attr.unique()):
                group_mask = protected_attr == group
                group_rate = predictions[group_mask].mean()
                group_rates.append(group_rate + u[i])
            
            # Enforce equal rates (projection onto simplex)
            target_rate = np.mean(group_rates)
            z_new = np.full(len(group_rates), target_rate)
        
        return z_new
    
    def _update_dual_variables(self, X, protected_attr, theta, z, u_current):
        """Update dual variables"""
        
        logits = X @ theta
        predictions = 1 / (1 + np.exp(-logits))
        
        u_new = u_current.copy()
        
        for i, group in enumerate(protected_attr.unique()):
            group_mask = protected_attr == group
            group_rate = predictions[group_mask].mean()
            u_new[i] += self.rho * (group_rate - z[i])
        
        return u_new
```

## Component 3: Adversarial Debiasing

### Purpose
Use adversarial training to learn representations that are predictive but not discriminatory.

### 3.1 Basic Adversarial Architecture

```python
import torch
import torch.nn as nn
import torch.optim as optim

class AdversarialFairClassifier(nn.Module):
    """Adversarial neural network for fair classification"""
    
    def __init__(self, input_dim, hidden_dim=64, n_protected_attrs=2):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Task predictor
        self.task_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Adversarial discriminator (predicts protected attributes)
        self.adversary = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_protected_attrs),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        task_output = self.task_predictor(features)
        adversary_output = self.adversary(features)
        
        return task_output, adversary_output, features

class AdversarialTrainer:
    """Trainer for adversarial fair models"""
    
    def __init__(self, model, lambda_adv=1.0, learning_rate=0.001):
        self.model = model
        self.lambda_adv = lambda_adv
        
        # Optimizers
        self.optimizer_task = optim.Adam(
            list(model.feature_extractor.parameters()) + 
            list(model.task_predictor.parameters()),
            lr=learning_rate
        )
        self.optimizer_adv = optim.Adam(
            model.adversary.parameters(),
            lr=learning_rate
        )
        
        # Loss functions
        self.task_criterion = nn.BCELoss()
        self.adv_criterion = nn.CrossEntropyLoss()
    
    def train_step(self, X_batch, y_batch, protected_batch):
        """Single training step"""
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_batch)
        y_tensor = torch.FloatTensor(y_batch).unsqueeze(1)
        protected_tensor = torch.LongTensor(protected_batch)
        
        # Forward pass
        task_output, adv_output, features = self.model(X_tensor)
        
        # Task loss
        task_loss = self.task_criterion(task_output, y_tensor)
        
        # Adversarial loss
        adv_loss = self.adv_criterion(adv_output, protected_tensor)
        
        # Update task predictor (minimize task loss, maximize adversarial loss)
        self.optimizer_task.zero_grad()
        total_task_loss = task_loss - self.lambda_adv * adv_loss
        total_task_loss.backward(retain_graph=True)
        self.optimizer_task.step()
        
        # Update adversary (minimize adversarial loss)
        self.optimizer_adv.zero_grad()
        adv_loss.backward()
        self.optimizer_adv.step()
        
        return {
            'task_loss': task_loss.item(),
            'adv_loss': adv_loss.item(),
            'total_loss': total_task_loss.item()
        }
    
    def fit(self, X, y, protected_attr, epochs=100, batch_size=64):
        """Train the adversarial model"""
        
        # Encode protected attributes
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        protected_encoded = le.fit_transform(protected_attr)
        
        # Training loop
        history = {'task_loss': [], 'adv_loss': [], 'total_loss': []}
        
        for epoch in range(epochs):
            epoch_losses = {'task_loss': [], 'adv_loss': [], 'total_loss': []}
            
            # Mini-batch training
            n_batches = len(X) // batch_size
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]
                protected_batch = protected_encoded[start_idx:end_idx]
                
                losses = self.train_step(X_batch, y_batch, protected_batch)
                
                for key, value in losses.items():
                    epoch_losses[key].append(value)
            
            # Record epoch averages
            for key in history.keys():
                history[key].append(np.mean(epoch_losses[key]))
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Task Loss = {history['task_loss'][-1]:.4f}, "
                      f"Adv Loss = {history['adv_loss'][-1]:.4f}")
        
        return history
```

### 3.2 Advanced Adversarial Techniques

```python
class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for stable adversarial training"""
    
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None

class AdvancedAdversarialModel(nn.Module):
    """Advanced adversarial model with gradient reversal and multiple adversaries"""
    
    def __init__(self, input_dim, hidden_dim=64, protected_attributes=None):
        super().__init__()
        
        self.protected_attributes = protected_attributes or ['gender', 'race']
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Task predictor
        self.task_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Multiple adversaries (one per protected attribute)
        self.adversaries = nn.ModuleDict()
        for attr in self.protected_attributes:
            self.adversaries[attr] = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 2),  # Assuming binary attributes
                nn.Softmax(dim=1)
            )
    
    def forward(self, x, lambda_grl=1.0):
        # Encode features
        features = self.encoder(x)
        
        # Task prediction
        task_output = self.task_predictor(features)
        
        # Adversarial predictions with gradient reversal
        adv_outputs = {}
        for attr in self.protected_attributes:
            reversed_features = GradientReversalLayer.apply(features, lambda_grl)
            adv_outputs[attr] = self.adversaries[attr](reversed_features)
        
        return task_output, adv_outputs, features

class MultiAttributeAdversarialTrainer:
    """Trainer for multi-attribute adversarial fairness"""
    
    def __init__(self, model, lambda_adv=1.0, learning_rate=0.001):
        self.model = model
        self.lambda_adv = lambda_adv
        
        # Single optimizer for all parameters
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        self.task_criterion = nn.BCELoss()
        self.adv_criterion = nn.CrossEntropyLoss()
    
    def train_step(self, X_batch, y_batch, protected_dict, lambda_grl=1.0):
        """Training step with multiple protected attributes"""
        
        X_tensor = torch.FloatTensor(X_batch)
        y_tensor = torch.FloatTensor(y_batch).unsqueeze(1)
        
        # Convert protected attributes to tensors
        protected_tensors = {}
        for attr, values in protected_dict.items():
            protected_tensors[attr] = torch.LongTensor(values)
        
        # Forward pass
        task_output, adv_outputs, features = self.model(X_tensor, lambda_grl)
        
        # Task loss
        task_loss = self.task_criterion(task_output, y_tensor)
        
        # Combined adversarial loss
        total_adv_loss = 0
        for attr in self.model.protected_attributes:
            if attr in adv_outputs and attr in protected_tensors:
                adv_loss = self.adv_criterion(adv_outputs[attr], protected_tensors[attr])
                total_adv_loss += adv_loss
        
        # Combined loss (task loss + adversarial loss with gradient reversal)
        total_loss = task_loss + self.lambda_adv * total_adv_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'task_loss': task_loss.item(),
            'adv_loss': total_adv_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def fit(self, X, y, protected_dict, epochs=100, batch_size=64):
        """Train with multiple protected attributes"""
        
        # Encode all protected attributes
        encoded_protected = {}
        label_encoders = {}
        
        for attr, values in protected_dict.items():
            le = LabelEncoder()
            encoded_protected[attr] = le.fit_transform(values)
            label_encoders[attr] = le
        
        # Training loop with adaptive lambda
        history = {'task_loss': [], 'adv_loss': [], 'total_loss': []}
        
        for epoch in range(epochs):
            # Adaptive lambda schedule
            lambda_grl = 2.0 / (1.0 + np.exp(-10 * epoch / epochs)) - 1.0
            
            epoch_losses = {'task_loss': [], 'adv_loss': [], 'total_loss': []}
            
            # Mini-batch training
            n_batches = len(X) // batch_size
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]
                
                # Get protected attribute batch
                protected_batch = {}
                for attr in encoded_protected.keys():
                    protected_batch[attr] = encoded_protected[attr][start_idx:end_idx]
                
                losses = self.train_step(X_batch, y_batch, protected_batch, lambda_grl)
                
                for key, value in losses.items():
                    epoch_losses[key].append(value)
            
            # Record averages
            for key in history.keys():
                history[key].append(np.mean(epoch_losses[key]))
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Task Loss = {history['task_loss'][-1]:.4f}, "
                      f"Adv Loss = {history['adv_loss'][-1]:.4f}, Lambda = {lambda_grl:.3f}")
        
        return history, label_encoders
```

## Component 4: Fairness Regularization

### Purpose
Add soft fairness penalties to standard loss functions for more flexible fairness-accuracy trade-offs.

### 4.1 Regularization Methods

```python
class FairnessRegularizers:
    """Collection of fairness regularization methods"""
    
    @staticmethod
    def demographic_parity_regularizer(predictions, protected_attr, weight=1.0):
        """Regularizer for demographic parity"""
        
        penalty = 0
        groups = protected_attr.unique()
        
        for i, group_a in enumerate(groups):
            for j, group_b in enumerate(groups):
                if i < j:
                    mask_a = protected_attr == group_a
                    mask_b = protected_attr == group_b
                    
                    rate_a = predictions[mask_a].mean()
                    rate_b = predictions[mask_b].mean()
                    
                    penalty += (rate_a - rate_b) ** 2
        
        return weight * penalty
    
    @staticmethod
    def mutual_information_regularizer(features, protected_attr, weight=1.0):
        """Regularizer based on mutual information between features and protected attributes"""
        
        # Estimate mutual information using KDE
        from sklearn.feature_selection import mutual_info_regression
        
        # Convert protected attribute to numeric if needed
        if protected_attr.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            protected_numeric = le.fit_transform(protected_attr)
        else:
            protected_numeric = protected_attr
        
        # Calculate mutual information for each feature
        mi_scores = []
        for i in range(features.shape[1]):
            mi = mutual_info_regression(
                features[:, i].reshape(-1, 1), 
                protected_numeric
            )[0]
            mi_scores.append(mi)
        
        # Penalty is sum of mutual information scores
        penalty = np.sum(mi_scores)
        
        return weight * penalty
    
    @staticmethod
    def fairness_gap_regularizer(predictions, true_labels, protected_attr, 
                                metric='equal_opportunity', weight=1.0):
        """Regularizer for various fairness metrics"""
        
        if metric == 'equal_opportunity':
            # Penalty for TPR differences
            penalty = 0
            positive_mask = true_labels == 1
            
            groups = protected_attr.unique()
            for i, group_a in enumerate(groups):
                for j, group_b in enumerate(groups):
                    if i < j:
                        mask_a = (protected_attr == group_a) & positive_mask
                        mask_b = (protected_attr == group_b) & positive_mask
                        
                        if mask_a.sum() > 0 and mask_b.sum() > 0:
                            tpr_a = predictions[mask_a].mean()
                            tpr_b = predictions[mask_b].mean()
                            penalty += (tpr_a - tpr_b) ** 2
        
        elif metric == 'equalized_odds':
            # Penalty for both TPR and FPR differences
            penalty = FairnessRegularizers.fairness_gap_regularizer(
                predictions, true_labels, protected_attr, 'equal_opportunity', weight
            )
            
            # Add FPR penalty
            negative_mask = true_labels == 0
            groups = protected_attr.unique()
            
            for i, group_a in enumerate(groups):
                for j, group_b in enumerate(groups):
                    if i < j:
                        mask_a = (protected_attr == group_a) & negative_mask
                        mask_b = (protected_attr == group_b) & negative_mask
                        
                        if mask_a.sum() > 0 and mask_b.sum() > 0:
                            fpr_a = predictions[mask_a].mean()
                            fpr_b = predictions[mask_b].mean()
                            penalty += weight * (fpr_a - fpr_b) ** 2
        
        return penalty

class RegularizedFairModel:
    """Model with fairness regularization"""
    
    def __init__(self, base_model, regularizers=None, regularizer_weights=None):
        self.base_model = base_model
        self.regularizers = regularizers or ['demographic_parity']
        self.regularizer_weights = regularizer_weights or [1.0]
        
        if len(self.regularizers) != len(self.regularizer_weights):
            raise ValueError("Number of regularizers must match number of weights")
    
    def fit(self, X, y, protected_attr, epochs=100, learning_rate=0.01):
        """Fit model with fairness regularization"""
        
        # For neural networks, use custom training loop
        if hasattr(self.base_model, 'predict_proba'):
            # Sklearn-style model with regularized objective
            return self._fit_sklearn_style(X, y, protected_attr)
        else:
            # Neural network with custom training
            return self._fit_neural_network(X, y, protected_attr, epochs, learning_rate)
    
    def _fit_sklearn_style(self, X, y, protected_attr):
        """Fit sklearn-style model with regularization"""
        
        # Define regularized objective
        def regularized_objective(params):
            # Set model parameters
            self.base_model.set_params(**self._unflatten_params(params))
            
            # Prediction loss
            predictions = self.base_model.predict_proba(X)[:, 1]
            prediction_loss = log_loss(y, predictions)
            
            # Fairness penalties
            total_penalty = 0
            for regularizer, weight in zip(self.regularizers, self.regularizer_weights):
                if regularizer == 'demographic_parity':
                    penalty = FairnessRegularizers.demographic_parity_regularizer(
                        predictions, protected_attr, weight
                    )
                elif regularizer == 'equal_opportunity':
                    penalty = FairnessRegularizers.fairness_gap_regularizer(
                        predictions, y, protected_attr, 'equal_opportunity', weight
                    )
                # Add more regularizers as needed
                
                total_penalty += penalty
            
            return prediction_loss + total_penalty
        
        # Optimize
        from scipy.optimize import minimize
        initial_params = self._flatten_params(self.base_model.get_params())
        
        result = minimize(
            regularized_objective,
            initial_params,
            method='L-BFGS-B'
        )
        
        # Set optimal parameters
        optimal_params = self._unflatten_params(result.x)
        self.base_model.set_params(**optimal_params)
        
        return self
    
    def _fit_neural_network(self, X, y, protected_attr, epochs, learning_rate):
        """Custom training for neural networks with fairness regularization"""
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Optimizer
        optimizer = optim.Adam(self.base_model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.base_model(X_tensor)
            predictions = outputs.squeeze()
            
            # Prediction loss
            prediction_loss = criterion(predictions, y_tensor)
            
            # Fairness regularization
            total_penalty = 0
            predictions_np = predictions.detach().numpy()
            
            for regularizer, weight in zip(self.regularizers, self.regularizer_weights):
                if regularizer == 'demographic_parity':
                    penalty = FairnessRegularizers.demographic_parity_regularizer(
                        predictions_np, protected_attr, weight
                    )
                elif regularizer == 'equal_opportunity':
                    penalty = FairnessRegularizers.fairness_gap_regularizer(
                        predictions_np, y, protected_attr, 'equal_opportunity', weight
                    )
                
                total_penalty += penalty
            
            # Total loss
            total_loss = prediction_loss + torch.tensor(total_penalty, requires_grad=True)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")
        
        return self
```

## Component 5: Multi-Objective Optimization

### Purpose
Explicitly trade off between fairness and accuracy using Pareto optimization.

### 5.1 Pareto-Optimal Fairness

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

class FairnessMultiObjectiveProblem(Problem):
    """Multi-objective problem for fairness-accuracy trade-off"""
    
    def __init__(self, X, y, protected_attr, model_class, fairness_metrics):
        self.X = X
        self.y = y
        self.protected_attr = protected_attr
        self.model_class = model_class
        self.fairness_metrics = fairness_metrics
        
        # Problem dimensions depend on model parameters
        n_vars = self._get_n_variables()
        
        super().__init__(
            n_var=n_vars,
            n_obj=1 + len(fairness_metrics),  # Accuracy + fairness objectives
            xl=-5.0,  # Lower bounds
            xu=5.0    # Upper bounds
        )
    
    def _get_n_variables(self):
        """Get number of optimization variables"""
        # For logistic regression: n_features + 1 (intercept)
        if self.model_class.__name__ == 'LogisticRegression':
            return self.X.shape[1] + 1
        else:
            # For other models, define appropriate parameter count
            raise NotImplementedError(f"Parameter count not defined for {self.model_class}")
    
    def _evaluate(self, X_params, out, *args, **kwargs):
        """Evaluate objectives for given parameters"""
        
        n_solutions = X_params.shape[0]
        objectives = np.zeros((n_solutions, self.n_obj))
        
        for i, params in enumerate(X_params):
            # Create and configure model with these parameters
            model = self.model_class()
            
            # Set parameters (implementation depends on model type)
            if hasattr(model, 'coef_'):
                model.coef_ = params[:-1].reshape(1, -1)
                model.intercept_ = params[-1:]
            
            # Calculate predictions
            try:
                predictions = model.predict_proba(self.X)[:, 1]
            except:
                # If model can't predict, assign worst possible scores
                objectives[i, :] = [1.0] * self.n_obj  # High values = bad
                continue
            
            # Objective 1: Negative accuracy (to minimize)
            accuracy = accuracy_score(self.y, (predictions > 0.5).astype(int))
            objectives[i, 0] = 1.0 - accuracy  # Minimize negative accuracy
            
            # Fairness objectives
            for j, metric in enumerate(self.fairness_metrics):
                if metric == 'demographic_parity':
                    # Calculate demographic parity violation
                    groups = self.protected_attr.unique()
                    max_disparity = 0
                    
                    for group_a in groups:
                        for group_b in groups:
                            if group_a != group_b:
                                mask_a = self.protected_attr == group_a
                                mask_b = self.protected_attr == group_b
                                
                                rate_a = predictions[mask_a].mean()
                                rate_b = predictions[mask_b].mean()
                                
                                disparity = abs(rate_a - rate_b)
                                max_disparity = max(max_disparity, disparity)
                    
                    objectives[i, j + 1] = max_disparity
                
                elif metric == 'equal_opportunity':
                    # Calculate equal opportunity violation
                    positive_mask = self.y == 1
                    groups = self.protected_attr.unique()
                    max_tpr_disparity = 0
                    
                    for group_a in groups:
                        for group_b in groups:
                            if group_a != group_b:
                                mask_a = (self.protected_attr == group_a) & positive_mask
                                mask_b = (self.protected_attr == group_b) & positive_mask
                                
                                if mask_a.sum() > 0 and mask_b.sum() > 0:
                                    tpr_a = predictions[mask_a].mean()
                                    tpr_b = predictions[mask_b].mean()
                                    
                                    tpr_disparity = abs(tpr_a - tpr_b)
                                    max_tpr_disparity = max(max_tpr_disparity, tpr_disparity)
                    
                    objectives[i, j + 1] = max_tpr_disparity
        
        out["F"] = objectives

class ParetoFairOptimizer:
    """Multi-objective optimizer for fairness-accuracy trade-offs"""
    
    def __init__(self, model_class, fairness_metrics=['demographic_parity']):
        self.model_class = model_class
        self.fairness_metrics = fairness_metrics
        self.pareto_solutions = None
        self.pareto_models = None
    
    def fit(self, X, y, protected_attr, population_size=100, n_generations=50):
        """Find Pareto-optimal solutions"""
        
        # Define problem
        problem = FairnessMultiObjectiveProblem(
            X, y, protected_attr, self.model_class, self.fairness_metrics
        )
        
        # Configure algorithm
        algorithm = NSGA2(pop_size=population_size)
        
        # Optimize
        result = minimize(
            problem,
            algorithm,
            ('n_gen', n_generations),
            verbose=True
        )
        
        # Store Pareto-optimal solutions
        self.pareto_solutions = result.X
        self.pareto_objectives = result.F
        
        # Create models for each Pareto solution
        self.pareto_models = []
        for params in self.pareto_solutions:
            model = self.model_class()
            
            # Set parameters
            if hasattr(model, 'coef_'):
                model.coef_ = params[:-1].reshape(1, -1)
                model.intercept_ = params[-1:]
            
            self.pareto_models.append(model)
        
        return self
    
    def get_trade_off_analysis(self):
        """Analyze trade-offs between objectives"""
        
        if self.pareto_objectives is None:
            raise ValueError("Must call fit() first")
        
        # Create DataFrame for analysis
        columns = ['accuracy_loss'] + [f'{metric}_violation' for metric in self.fairness_metrics]
        trade_offs = pd.DataFrame(self.pareto_objectives, columns=columns)
        
        # Convert accuracy loss back to accuracy
        trade_offs['accuracy'] = 1.0 - trade_offs['accuracy_loss']
        trade_offs = trade_offs.drop('accuracy_loss', axis=1)
        
        return trade_offs
    
    def select_solution(self, criteria='balanced'):
        """Select a solution based on criteria"""
        
        if self.pareto_objectives is None:
            raise ValueError("Must call fit() first")
        
        if criteria == 'balanced':
            # Select solution with best balanced performance
            # Normalize all objectives to [0, 1] scale
            normalized_objectives = self.pareto_objectives.copy()
            for i in range(normalized_objectives.shape[1]):
                col_min = normalized_objectives[:, i].min()
                col_max = normalized_objectives[:, i].max()
                if col_max > col_min:
                    normalized_objectives[:, i] = (
                        (normalized_objectives[:, i] - col_min) / (col_max - col_min)
                    )
            
            # Find solution with minimum sum of normalized objectives
            balanced_scores = normalized_objectives.sum(axis=1)
            best_idx = np.argmin(balanced_scores)
            
        elif criteria == 'max_accuracy':
            # Select solution with highest accuracy (lowest accuracy loss)
            best_idx = np.argmin(self.pareto_objectives[:, 0])
            
        elif criteria == 'max_fairness':
            # Select solution with best overall fairness
            fairness_scores = self.pareto_objectives[:, 1:].sum(axis=1)
            best_idx = np.argmin(fairness_scores)
            
        else:
            raise ValueError(f"Unknown selection criteria: {criteria}")
        
        return self.pareto_models[best_idx], self.pareto_objectives[best_idx]
    
    def visualize_pareto_front(self):
        """Visualize Pareto front for 2-objective problems"""
        
        if self.pareto_objectives.shape[1] != 2:
            raise ValueError("Visualization only supported for 2-objective problems")
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.scatter(
            1.0 - self.pareto_objectives[:, 0],  # Convert back to accuracy
            self.pareto_objectives[:, 1],
            alpha=0.7,
            s=50
        )
        plt.xlabel('Accuracy')
        plt.ylabel(f'{self.fairness_metrics[0]} Violation')
        plt.title('Pareto Front: Accuracy vs Fairness')
        plt.grid(True, alpha=0.3)
        
        # Highlight different solution types
        balanced_model, balanced_obj = self.select_solution('balanced')
        max_acc_model, max_acc_obj = self.select_solution('max_accuracy')
        max_fair_model, max_fair_obj = self.select_solution('max_fairness')
        
        plt.scatter(1.0 - balanced_obj[0], balanced_obj[1], 
                   color='red', s=100, label='Balanced', marker='s')
        plt.scatter(1.0 - max_acc_obj[0], max_acc_obj[1], 
                   color='green', s=100, label='Max Accuracy', marker='^')
        plt.scatter(1.0 - max_fair_obj[0], max_fair_obj[1], 
                   color='blue', s=100, label='Max Fairness', marker='D')
        
        plt.legend()
        plt.show()
        
        return plt.gcf()
```

## Component 6: Integration and Validation Framework

### Purpose
Coordinate multiple in-processing techniques and validate their effectiveness.

### 6.1 Integrated In-Processing Pipeline

```python
class InProcessingPipeline:
    """Integrated pipeline for in-processing fairness interventions"""
    
    def __init__(self, base_model, intervention_strategy='hybrid'):
        self.base_model = base_model
        self.intervention_strategy = intervention_strategy
        self.fitted_components = {}
        self.final_model = None
    
    def fit(self, X, y, protected_attr, fairness_goals=None):
        """Fit model with integrated fairness interventions"""
        
        fairness_goals = fairness_goals or {
            'demographic_parity': 0.05,
            'equal_opportunity': 0.05
        }
        
        if self.intervention_strategy == 'constraints_only':
            # Use constrained optimization only
            model = LagrangianFairOptimizer(
                self.base_model,
                fairness_constraints=list(fairness_goals.keys()),
                lambda_fairness=1.0
            )
            model.fit(X, y, protected_attr)
            self.final_model = model
            
        elif self.intervention_strategy == 'adversarial_only':
            # Use adversarial debiasing only
            if isinstance(protected_attr, pd.Series):
                protected_dict = {protected_attr.name: protected_attr}
            else:
                protected_dict = {'protected': protected_attr}
            
            # Convert to neural network if needed
            input_dim = X.shape[1]
            model = AdvancedAdversarialModel(
                input_dim, 
                protected_attributes=list(protected_dict.keys())
            )
            
            trainer = MultiAttributeAdversarialTrainer(model)
            trainer.fit(X, y, protected_dict)
            
            self.final_model = model
            self.fitted_components['trainer'] = trainer
            
        elif self.intervention_strategy == 'regularization_only':
            # Use fairness regularization only
            regularizers = list(fairness_goals.keys())
            weights = [1.0] * len(regularizers)
            
            model = RegularizedFairModel(
                self.base_model, 
                regularizers=regularizers,
                regularizer_weights=weights
            )
            model.fit(X, y, protected_attr)
            self.final_model = model
            
        elif self.intervention_strategy == 'pareto_optimization':
            # Use multi-objective optimization
            optimizer = ParetoFairOptimizer(
                type(self.base_model),
                fairness_metrics=list(fairness_goals.keys())
            )
            optimizer.fit(X, y, protected_attr)
            
            # Select balanced solution
            selected_model, selected_objectives = optimizer.select_solution('balanced')
            
            self.final_model = selected_model
            self.fitted_components['pareto_optimizer'] = optimizer
            
        elif self.intervention_strategy == 'hybrid':
            # Combine multiple approaches
            self._fit_hybrid_approach(X, y, protected_attr, fairness_goals)
            
        else:
            raise ValueError(f"Unknown intervention strategy: {self.intervention_strategy}")
        
        return self
    
    def _fit_hybrid_approach(self, X, y, protected_attr, fairness_goals):
        """Fit using hybrid approach combining multiple techniques"""
        
        # Stage 1: Train with regularization for initial fairness
        regularized_model = RegularizedFairModel(
            self.base_model,
            regularizers=['demographic_parity'],
            regularizer_weights=[0.5]
        )
        regularized_model.fit(X, y, protected_attr)
        
        # Stage 2: Fine-tune with constraints
        constrained_model = LagrangianFairOptimizer(
            regularized_model.base_model,
            fairness_constraints=list(fairness_goals.keys()),
            lambda_fairness=0.3
        )
        constrained_model.fit(X, y, protected_attr)
        
        # Stage 3: Final adversarial refinement if needed
        predictions = constrained_model.base_model.predict_proba(X)[:, 1]
        current_fairness = self._evaluate_fairness(predictions, y, protected_attr)
        
        if any(current_fairness[metric] > threshold 
               for metric, threshold in fairness_goals.items()):
            
            # Apply adversarial debiasing as final step
            if isinstance(protected_attr, pd.Series):
                protected_dict = {protected_attr.name: protected_attr}
            else:
                protected_dict = {'protected': protected_attr}
            
            # Convert to adversarial model
            input_dim = X.shape[1]
            adversarial_model = AdvancedAdversarialModel(
                input_dim,
                protected_attributes=list(protected_dict.keys())
            )
            
            # Initialize with constrained model weights if possible
            if hasattr(constrained_model.base_model, 'coef_'):
                # Transfer weights to adversarial model
                with torch.no_grad():
                    adversarial_model.encoder[0].weight.data = torch.FloatTensor(
                        constrained_model.base_model.coef_
                    )
            
            trainer = MultiAttributeAdversarialTrainer(adversarial_model, lambda_adv=0.5)
            trainer.fit(X, y, protected_dict, epochs=50)  # Fewer epochs for fine-tuning
            
            self.final_model = adversarial_model
            self.fitted_components['adversarial_trainer'] = trainer
        else:
            self.final_model = constrained_model.base_model
        
        # Store intermediate models for analysis
        self.fitted_components['regularized'] = regularized_model
        self.fitted_components['constrained'] = constrained_model
    
    def _evaluate_fairness(self, predictions, y, protected_attr):
        """Evaluate current fairness metrics"""
        
        fairness_metrics = {}
        
        # Demographic parity
        groups = protected_attr.unique()
        max_dp_violation = 0
        
        for group_a in groups:
            for group_b in groups:
                if group_a != group_b:
                    mask_a = protected_attr == group_a
                    mask_b = protected_attr == group_b
                    
                    rate_a = predictions[mask_a].mean()
                    rate_b = predictions[mask_b].mean()
                    
                    dp_violation = abs(rate_a - rate_b)
                    max_dp_violation = max(max_dp_violation, dp_violation)
        
        fairness_metrics['demographic_parity'] = max_dp_violation
        
        # Equal opportunity
        positive_mask = y == 1
        max_eo_violation = 0
        
        for group_a in groups:
            for group_b in groups:
                if group_a != group_b:
                    mask_a = (protected_attr == group_a) & positive_mask
                    mask_b = (protected_attr == group_b) & positive_mask
                    
                    if mask_a.sum() > 0 and mask_b.sum() > 0:
                        tpr_a = predictions[mask_a].mean()
                        tpr_b = predictions[mask_b].mean()
                        
                        eo_violation = abs(tpr_a - tpr_b)
                        max_eo_violation = max(max_eo_violation, eo_violation)
        
        fairness_metrics['equal_opportunity'] = max_eo_violation
        
        return fairness_metrics
    
    def predict(self, X):
        """Make predictions using the fitted model"""
        
        if self.final_model is None:
            raise ValueError("Must call fit() first")
        
        if hasattr(self.final_model, 'predict_proba'):
            return self.final_model.predict_proba(X)[:, 1]
        elif isinstance(self.final_model, torch.nn.Module):
            self.final_model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions, _, _ = self.final_model(X_tensor)
                return predictions.numpy().squeeze()
        else:
            return self.final_model.predict(X)
    
    def get_intervention_summary(self):
        """Get summary of interventions applied"""
        
        summary = {
            'strategy': self.intervention_strategy,
            'components_used': list(self.fitted_components.keys()),
            'final_model_type': type(self.final_model).__name__
        }
        
        return summary
```

## Integration with Other Toolkits

### Inputs from Pre-Processing Toolkit
- **Cleaned training data**: Bias-reduced datasets
- **Residual bias patterns**: Specific issues requiring model-level intervention
- **Feature importance**: Critical features needing constraints

### Inputs from Causal Fairness Toolkit
- **Causal pathways**: Which relationships to preserve vs. constrain
- **Counterfactual requirements**: Individual fairness specifications
- **Bias mechanisms**: Direct, indirect, proxy discrimination patterns

### Outputs to Post-Processing Toolkit
- **Trained fair models**: Models with embedded fairness constraints
- **Residual fairness gaps**: Issues requiring threshold adjustment
- **Group-specific behaviors**: Patterns needing calibration

## Documentation Template

```markdown
# In-Processing Intervention Report

## Executive Summary
- **Intervention Strategy**: [constraints/adversarial/regularization/pareto/hybrid]
- **Fairness Goals**: [target metrics and thresholds]
- **Results**: [fairness improvements and accuracy trade-offs]

## Intervention Details

### Methods Applied
| Component | Configuration | Rationale |
|-----------|---------------|-----------|
| Constrained Optimization | λ=1.0, demographic parity | Address systematic bias |
| Adversarial Debiasing | λ_adv=0.5, gradient reversal | Remove proxy correlations |

### Model Architecture
- **Base Model**: [type and configuration]
- **Fairness Components**: [constraints, adversaries, regularizers]
- **Training Strategy**: [optimization approach and hyperparameters]

## Results Analysis

### Fairness Metrics
| Metric | Target | Before | After | Improvement |
|--------|---------|---------|--------|-------------|
| Demographic Parity | <0.05 | 0.18 | 0.04 | 78% reduction |
| Equal Opportunity | <0.05 | 0.13 | 0.03 | 77% reduction |

### Performance Impact
- **Accuracy**: [before] → [after] ([change])
- **AUC**: [before] → [after] ([change])
- **Calibration**: [assessment across groups]

### Trade-off Analysis
- **Pareto Frontier**: [if applicable]
- **Sensitivity Analysis**: [robustness to hyperparameters]
- **Alternative Solutions**: [other points on trade-off curve]

## Validation and Monitoring
- **Cross-validation Results**: [stability across folds]
- **Robustness Testing**: [performance under distribution shift]
- **Monitoring Plan**: [key metrics to track in production]
```

This In-Processing Fairness Toolkit provides comprehensive methods for embedding fairness directly into model training, ensuring that fairness becomes an integral part of your AI system's decision-making process.# In-Processing Fairness Toolkit
## Embedding Fairness Constraints During Model Training

### Overview

The In-Processing Fairness Toolkit embeds fairness directly into the model training process through constrained optimization, adversarial learning, and fairness regularization. This approach ensures fairness is built into the model's decision boundaries rather than applied as an afterthought.

### When to Use This Toolkit

- Pre-processing interventions alone are insufficient
- Need mathematical guarantees of fairness in model behavior
- Working with complex, non-linear relationships where data transformation is difficult
- Require fine-grained control over fairness-performance trade-offs
- Dealing with high-dimensional or unstructured data (text, images)

### Core Components

## Component 1: Fairness Constraint Formulation

### Purpose
Translate fairness definitions into mathematical constraints that can be incorporated into optimization objectives.

### 1.1 Constraint Templates

```python
class FairnessConstraints:
    """Library of fairness constraints for different metrics"""
    
    @staticmethod
    def demographic_parity_constraint(predictions, protected_attr, tolerance=0.05):
        """Demographic parity: P(Ŷ=1|A=0) ≈ P(Ŷ=1|A=1)"""
        
        constraints = []
        
        for group_a in protected_attr.unique():
            for group_b in protected_attr.unique():
                if group_a != group_b:
                    mask_a = protected_attr == group_a
                    mask_b = protected_attr == group_b
                    
                    rate_a = predictions[mask_a].mean()
                    rate_b = predictions[mask_b].mean()
                    
                    # Inequality constraint: |rate_a - rate_b| <= tolerance
                    constraint = abs(rate_a - rate_b) - tolerance
                    constraints.append(constraint)
        
        return constraints
    
    @staticmethod
    def equal_opportunity_constraint(predictions, true_labels, protected_attr, tolerance=0.05):
        """Equal opportunity: P(Ŷ=1|Y=1,A=0) ≈ P(Ŷ=1|Y=1,A=1)"""
        
        constraints = []
        positive_mask = true_labels == 1
        
        for group_a in protected_attr.unique():
            for group_b in protected_attr.unique():
                if group_a != group_b:
                    mask_a = (protected_attr == group_a) & positive_mask
                    mask_b = (protected_attr == group_b) & positive_mask
                    
                    if mask_a.sum() > 0 and mask_b.sum() > 0:
                        tpr_a = predictions[mask_a].mean()
                        tpr_b = predictions[mask_b].mean()
                        
                        constraint = abs(tpr_a - tpr_b) - tolerance
                        constraints.append(constraint)
        
        return constraints
    
    @staticmethod
    def equalized_odds_constraint(predictions, true_labels, protected_attr, tolerance=0.05):
        """Equalized odds: Equal TPR and FPR across groups"""
        
        constraints = []
        
        # True Positive Rate constraints
        tpr_constraints = FairnessConstraints.equal_opportunity_constraint(
            predictions, true_labels, protected_attr, tolerance
        )
        constraints.extend(tpr_constraints)
        
        # False Positive Rate constraints
        negative_mask = true_labels == 0
        
        for group_a in protected_attr.unique():
            for group_b in protected_attr.unique():
                if group_a != group_b:
                    mask_a = (protected_attr == group_a) & negative_mask
                    mask_b = (protected_attr == group_b) & negative_mask
                    
                    if mask_a.sum() > 0 and mask_b.sum() > 0:
                        fpr_a = predictions[mask_a].mean()
                        fpr_b = predictions[mask_b].mean()
                        
                        constraint = abs(fpr_a - fpr_b) - tolerance
                        constraints.append(constraint)
        
        return constraints
    
    @staticmethod
    def counterfactual_fairness_constraint(model, data, protected_attr, tolerance=0.05):
        """Counterfactual fairness: Outcome shouldn't change if protected attribute changes"""
        
        constraints = []
        
        for idx, row in data.iterrows():
            original_prediction = model.predict(row.values.reshape(1, -1))[0]
            
            # Create counterfactual by changing protected attribute
            for alt_value in protected_attr.unique():
                if alt_value != row[protected_attr]:
                    counterfactual_row = row.copy()
                    counterfactual_row[protected_attr] = alt_value
                    
                    counterfactual_prediction = model.predict(
                        counterfactual_row.values.reshape(1, -1)
                    )[0]
                    
                    # Constraint: predictions should be similar
                    constraint = abs(original_prediction - counterfactual_prediction) - tolerance
                    constraints.append(constraint)
        
        return constraints
```

### 1.2 Intersectional Constraint Handling

```python
class IntersectionalConstraints:
    """Handle fairness constraints for intersectional groups"""
    
    def __init__(self, protected_attributes, fairness_metric='demographic_parity'):
        self.protected_attributes = protected_attributes
        self.fairness_metric = fairness_metric
    
    def generate_intersectional_groups(self, data):
        """Generate all intersectional combinations"""
        
        intersectional_groups = []
        
        # Get unique values for each protected attribute
        attr_values = {}
        for attr in self.protected_attributes:
            attr_values[attr] = data[attr].unique()
        
        # Generate all combinations
        from itertools import product
        for combination in product(*attr_values.values()):
            group_mask = pd.Series([True] * len(data))
            group_description = {}
            
            for attr, value in zip(self.protected_attributes, combination):
                group_mask &= (data[attr] == value)
                group_description[attr] = value
            
            if group_mask.sum() > 0:  # Only include non-empty groups
                intersectional_groups.append({
                    'mask': group_mask,
                    'description': group_description,
                    'size': group_mask.sum()
                })
        
        return intersectional_groups
    
    def create_intersectional_constraints(self, predictions, true_labels, data, tolerance=0.05):
        """Create fairness constraints for intersectional groups"""
        
        groups = self.generate_intersectional_groups(data)
        constraints = []
        
        # Create pairwise constraints between all groups
        for i, group_a in enumerate(groups):
            for j, group_b in enumerate(groups):
                if i < j:  # Avoid duplicate constraints
                    if self.fairness_metric == 'demographic_parity':
                        rate_a = predictions[group_a['mask']].mean()
                        rate_b = predictions[group_b['mask']].mean()
                        constraint = abs(rate_a - rate_b) - tolerance
                        
                    elif self.fairness_metric == 'equal_opportunity':
                        positive_a = group_a['mask'] & (true_labels == 1)
                        positive_b = group_b['mask'] & (true_labels == 1)
                        
                        if positive_a.sum() > 0 and positive_b.sum() > 0:
                            tpr_a = predictions[positive_a].mean()
                            tpr_b = predictions[positive_b].mean()
                            constraint = abs(tpr_a - tpr_b) - tolerance
                        else:
                            continue  #