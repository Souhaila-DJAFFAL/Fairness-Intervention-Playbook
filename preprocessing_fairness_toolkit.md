# Pre-Processing Fairness Toolkit
## Addressing Bias at the Data Level

### Overview

The Pre-Processing Fairness Toolkit provides systematic methods to identify and remediate bias in training data before it reaches the model. This proactive approach addresses bias at its source, often providing the most effective and interpretable fairness interventions.

### When to Use This Toolkit

- Causal analysis reveals data-level bias mechanisms
- Significant representation disparities across protected groups
- Historical discrimination embedded in training labels
- Proxy variables leaking protected attribute information
- Need for interpretable fairness interventions

### Core Components

## Component 1: Comprehensive Data Auditing Framework

### Purpose
Systematically identify bias patterns in datasets to guide intervention selection.

### Auditing Checklist

#### 1.1 Representation Analysis
```python
def analyze_representation(data, protected_attrs, reference_population=None):
    """Comprehensive representation analysis across protected attributes"""
    
    results = {}
    
    for attr in protected_attrs:
        # Basic representation
        attr_counts = data[attr].value_counts(normalize=True)
        results[attr] = {'distribution': attr_counts}
        
        # Compare to reference if available
        if reference_population and attr in reference_population:
            reference_dist = reference_population[attr]
            bias_score = calculate_representation_bias(attr_counts, reference_dist)
            results[attr]['bias_score'] = bias_score
        
        # Intersectional analysis
        if len(protected_attrs) > 1:
            intersectional_dist = data.groupby(protected_attrs).size() / len(data)
            results['intersectional'] = intersectional_dist
    
    return results

def calculate_representation_bias(observed, expected):
    """Calculate bias score based on deviation from expected distribution"""
    return sum(abs(observed - expected)) / 2  # Total variation distance
```

#### 1.2 Correlation Pattern Detection
```python
def detect_proxy_variables(data, protected_attrs, threshold=0.3):
    """Identify features that may serve as proxies for protected attributes"""
    
    proxy_candidates = {}
    
    for attr in protected_attrs:
        correlations = {}
        
        # Continuous features
        numeric_features = data.select_dtypes(include=[np.number]).columns
        for feature in numeric_features:
            if feature != attr:
                corr = abs(data[attr].astype('category').cat.codes.corr(data[feature]))
                if corr > threshold:
                    correlations[feature] = corr
        
        # Categorical features
        categorical_features = data.select_dtypes(include=['object', 'category']).columns
        for feature in categorical_features:
            if feature != attr:
                # Use Cramér's V for categorical associations
                cramers_v = calculate_cramers_v(data[attr], data[feature])
                if cramers_v > threshold:
                    correlations[feature] = cramers_v
        
        proxy_candidates[attr] = correlations
    
    return proxy_candidates

def calculate_cramers_v(x, y):
    """Calculate Cramér's V statistic for categorical association"""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
```

#### 1.3 Label Quality Assessment
```python
def assess_label_bias(data, outcome_col, protected_attrs):
    """Assess potential bias in outcome labels"""
    
    bias_indicators = {}
    
    # Outcome rate disparities
    for attr in protected_attrs:
        group_rates = data.groupby(attr)[outcome_col].mean()
        rate_disparity = group_rates.max() - group_rates.min()
        bias_indicators[f'{attr}_rate_disparity'] = rate_disparity
    
    # Conditional outcome analysis
    # Control for observable characteristics to isolate bias
    feature_cols = [col for col in data.columns if col not in protected_attrs + [outcome_col]]
    
    for attr in protected_attrs:
        # Logistic regression controlling for other features
        X = pd.get_dummies(data[feature_cols + [attr]])
        y = data[outcome_col]
        
        model = LogisticRegression()
        model.fit(X, y)
        
        # Extract coefficient for protected attribute
        attr_coef = model.coef_[0][X.columns.str.contains(attr)]
        bias_indicators[f'{attr}_conditional_bias'] = attr_coef.mean()
    
    return bias_indicators
```

### Audit Report Template

```python
class DataAuditReport:
    def __init__(self, data, protected_attrs):
        self.data = data
        self.protected_attrs = protected_attrs
        self.findings = {}
    
    def generate_report(self):
        """Generate comprehensive audit report"""
        
        # Representation analysis
        self.findings['representation'] = analyze_representation(
            self.data, self.protected_attrs
        )
        
        # Proxy detection
        self.findings['proxy_variables'] = detect_proxy_variables(
            self.data, self.protected_attrs
        )
        
        # Label bias assessment
        if 'outcome' in self.data.columns:
            self.findings['label_bias'] = assess_label_bias(
                self.data, 'outcome', self.protected_attrs
            )
        
        return self.create_summary()
    
    def create_summary(self):
        """Create executive summary of findings"""
        summary = {
            'high_risk_findings': [],
            'medium_risk_findings': [],
            'recommendations': []
        }
        
        # Analyze findings and categorize risk
        for finding_type, results in self.findings.items():
            risk_level = self.assess_risk_level(finding_type, results)
            if risk_level == 'high':
                summary['high_risk_findings'].append((finding_type, results))
            elif risk_level == 'medium':
                summary['medium_risk_findings'].append((finding_type, results))
        
        return summary
```

## Component 2: Intervention Selection Framework

### Purpose
Match identified bias patterns to appropriate pre-processing interventions.

### Decision Tree

```python
def select_preprocessing_intervention(audit_findings, constraints=None):
    """Select appropriate pre-processing interventions based on audit findings"""
    
    interventions = []
    
    # Representation bias → Reweighting/Resampling
    if 'representation' in audit_findings:
        for attr, findings in audit_findings['representation'].items():
            if attr != 'intersectional' and findings.get('bias_score', 0) > 0.1:
                interventions.append({
                    'type': 'reweighting',
                    'target': attr,
                    'method': 'inverse_frequency',
                    'priority': 'high'
                })
    
    # Proxy variables → Feature transformation
    if 'proxy_variables' in audit_findings:
        for attr, proxies in audit_findings['proxy_variables'].items():
            for proxy_var, correlation in proxies.items():
                if correlation > 0.5:
                    interventions.append({
                        'type': 'feature_transformation',
                        'target': proxy_var,
                        'method': 'disparate_impact_removal',
                        'protected_attr': attr,
                        'priority': 'high'
                    })
                elif correlation > 0.3:
                    interventions.append({
                        'type': 'feature_transformation',
                        'target': proxy_var,
                        'method': 'fair_representation',
                        'protected_attr': attr,
                        'priority': 'medium'
                    })
    
    # Label bias → Prejudice removal
    if 'label_bias' in audit_findings:
        for bias_indicator, value in audit_findings['label_bias'].items():
            if abs(value) > 0.1:  # Significant bias threshold
                interventions.append({
                    'type': 'label_correction',
                    'method': 'prejudice_removal',
                    'target': bias_indicator.split('_')[0],
                    'priority': 'high'
                })
    
    # Sort by priority and feasibility
    return prioritize_interventions(interventions, constraints)

def prioritize_interventions(interventions, constraints=None):
    """Prioritize interventions based on impact and feasibility"""
    
    priority_order = {'high': 3, 'medium': 2, 'low': 1}
    
    # Sort by priority
    interventions.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
    
    # Apply constraints if provided
    if constraints:
        filtered_interventions = []
        for intervention in interventions:
            if meets_constraints(intervention, constraints):
                filtered_interventions.append(intervention)
        return filtered_interventions
    
    return interventions
```

## Component 3: Reweighting and Resampling Techniques

### Purpose
Address representation disparities and historical bias through instance-level adjustments.

### 3.1 Instance Weighting Methods

```python
class InstanceWeighting:
    def __init__(self, fairness_metric='demographic_parity'):
        self.fairness_metric = fairness_metric
        self.weights = None
    
    def fit(self, X, y, protected_attr):
        """Calculate fairness-aware instance weights"""
        
        if self.fairness_metric == 'demographic_parity':
            self.weights = self._demographic_parity_weights(X, y, protected_attr)
        elif self.fairness_metric == 'equal_opportunity':
            self.weights = self._equal_opportunity_weights(X, y, protected_attr)
        else:
            raise ValueError(f"Unsupported fairness metric: {self.fairness_metric}")
        
        return self
    
    def _demographic_parity_weights(self, X, y, protected_attr):
        """Calculate weights for demographic parity"""
        weights = np.ones(len(X))
        
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            group_size = group_mask.sum()
            
            # Target: equal representation across groups
            target_size = len(X) / len(protected_attr.unique())
            weight_multiplier = target_size / group_size
            
            weights[group_mask] *= weight_multiplier
        
        return weights
    
    def _equal_opportunity_weights(self, X, y, protected_attr):
        """Calculate weights for equal opportunity"""
        weights = np.ones(len(X))
        
        # Focus on positive class (y=1)
        positive_mask = y == 1
        
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            group_positive_mask = group_mask & positive_mask
            
            group_positive_rate = group_positive_mask.sum() / group_mask.sum()
            overall_positive_rate = positive_mask.sum() / len(y)
            
            # Adjust weights to equalize positive rates
            if group_positive_rate > 0:
                weight_multiplier = overall_positive_rate / group_positive_rate
                weights[group_positive_mask] *= weight_multiplier
        
        return weights
    
    def transform(self, X, y, protected_attr):
        """Apply calculated weights"""
        if self.weights is None:
            raise ValueError("Must call fit() before transform()")
        
        return X, y, self.weights
```

### 3.2 Advanced Resampling Strategies

```python
class FairnessSMOTE:
    """SMOTE variant that considers fairness constraints"""
    
    def __init__(self, k_neighbors=5, fairness_target='demographic_parity'):
        self.k_neighbors = k_neighbors
        self.fairness_target = fairness_target
        self.smote_models = {}
    
    def fit_resample(self, X, y, protected_attr):
        """Resample data to achieve fairness goals"""
        
        # Calculate target distribution
        target_dist = self._calculate_target_distribution(y, protected_attr)
        
        # Apply group-specific SMOTE
        resampled_data = []
        
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            group_X = X[group_mask]
            group_y = y[group_mask]
            
            # Calculate how many samples needed for this group
            current_count = group_mask.sum()
            target_count = int(target_dist[group] * len(X))
            
            if target_count > current_count:
                # Oversample using SMOTE
                smote = SMOTE(k_neighbors=self.k_neighbors, random_state=42)
                
                # Create temporary dataset for SMOTE
                temp_y = np.zeros(len(group_y))  # SMOTE needs minority class
                if (group_y == 1).sum() < (group_y == 0).sum():
                    temp_y[group_y == 1] = 1
                else:
                    temp_y[group_y == 0] = 1
                
                resampled_X, resampled_temp_y = smote.fit_resample(group_X, temp_y)
                
                # Restore original labels proportionally
                resampled_y = np.random.choice(
                    group_y, size=len(resampled_X), 
                    p=group_y.value_counts(normalize=True)
                )
                
                resampled_data.append((
                    resampled_X, resampled_y, 
                    np.full(len(resampled_X), group)
                ))
            else:
                # Subsample if needed
                if target_count < current_count:
                    indices = np.random.choice(
                        np.where(group_mask)[0], 
                        size=target_count, replace=False
                    )
                    resampled_data.append((
                        X[indices], y[indices], protected_attr[indices]
                    ))
                else:
                    resampled_data.append((group_X, group_y, protected_attr[group_mask]))
        
        # Combine resampled data
        final_X = np.vstack([data[0] for data in resampled_data])
        final_y = np.concatenate([data[1] for data in resampled_data])
        final_protected = np.concatenate([data[2] for data in resampled_data])
        
        return final_X, final_y, final_protected
    
    def _calculate_target_distribution(self, y, protected_attr):
        """Calculate target distribution for fairness"""
        
        if self.fairness_target == 'demographic_parity':
            # Equal representation across groups
            unique_groups = protected_attr.unique()
            return {group: 1.0 / len(unique_groups) for group in unique_groups}
        
        elif self.fairness_target == 'equal_opportunity':
            # Equal positive class representation
            positive_mask = y == 1
            group_positive_rates = {}
            
            for group in protected_attr.unique():
                group_mask = protected_attr == group
                group_positive_rate = (group_mask & positive_mask).sum() / group_mask.sum()
                group_positive_rates[group] = group_positive_rate
            
            # Target: average positive rate across groups
            target_rate = np.mean(list(group_positive_rates.values()))
            
            # Adjust group sizes to achieve target rate
            target_dist = {}
            for group in protected_attr.unique():
                current_rate = group_positive_rates[group]
                if current_rate > 0:
                    size_multiplier = target_rate / current_rate
                    target_dist[group] = size_multiplier / sum(group_positive_rates.values())
                else:
                    target_dist[group] = 1.0 / len(protected_attr.unique())
            
            return target_dist
        
        else:
            raise ValueError(f"Unsupported fairness target: {self.fairness_target}")
```

## Component 4: Feature Transformation Methods

### Purpose
Remove bias embedded in feature correlations while preserving predictive utility.

### 4.1 Disparate Impact Removal

```python
class DisparateImpactRemover:
    """Remove disparate impact from features while preserving rank order"""
    
    def __init__(self, repair_level=1.0):
        self.repair_level = repair_level  # 0 = no repair, 1 = full repair
        self.repair_functions = {}
    
    def fit(self, X, protected_attr):
        """Learn repair transformations for each feature"""
        
        for feature in X.columns:
            if feature == protected_attr:
                continue
                
            self.repair_functions[feature] = self._fit_repair_function(
                X[feature], X[protected_attr]
            )
        
        return self
    
    def _fit_repair_function(self, feature_values, protected_attr):
        """Fit repair function for a single feature"""
        
        repair_function = {}
        
        # Calculate group-specific distributions
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            group_values = feature_values[group_mask]
            
            # Store empirical CDF for this group
            sorted_values = np.sort(group_values)
            repair_function[group] = {
                'values': sorted_values,
                'percentiles': np.linspace(0, 1, len(sorted_values))
            }
        
        # Calculate overall distribution for repair target
        overall_values = np.sort(feature_values)
        overall_percentiles = np.linspace(0, 1, len(overall_values))
        repair_function['overall'] = {
            'values': overall_values,
            'percentiles': overall_percentiles
        }
        
        return repair_function
    
    def transform(self, X, protected_attr):
        """Apply disparate impact removal"""
        
        X_repaired = X.copy()
        
        for feature in X.columns:
            if feature == protected_attr or feature not in self.repair_functions:
                continue
            
            repair_func = self.repair_functions[feature]
            
            for group in protected_attr.unique():
                group_mask = protected_attr == group
                group_values = X[feature][group_mask]
                
                # Map group values to percentiles within group
                group_percentiles = np.interp(
                    group_values,
                    repair_func[group]['values'],
                    repair_func[group]['percentiles']
                )
                
                # Map percentiles to overall distribution
                repaired_values = np.interp(
                    group_percentiles,
                    repair_func['overall']['percentiles'],
                    repair_func['overall']['values']
                )
                
                # Apply repair level (interpolate between original and repaired)
                final_values = (
                    (1 - self.repair_level) * group_values + 
                    self.repair_level * repaired_values
                )
                
                X_repaired.loc[group_mask, feature] = final_values
        
        return X_repaired
```

### 4.2 Fair Representation Learning

```python
class FairAutoEncoder:
    """Learn fair representations using adversarial autoencoder"""
    
    def __init__(self, encoding_dim=50, fairness_weight=1.0):
        self.encoding_dim = encoding_dim
        self.fairness_weight = fairness_weight
        self.encoder = None
        self.decoder = None
        self.adversary = None
    
    def build_model(self, input_dim, protected_attr_dim):
        """Build adversarial autoencoder architecture"""
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.encoding_dim, activation='linear', name='encoding')
        ])
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.encoding_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='linear')
        ])
        
        # Adversary (tries to predict protected attribute from encoding)
        self.adversary = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(self.encoding_dim,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(protected_attr_dim, activation='softmax')
        ])
    
    def fit(self, X, protected_attr, epochs=100):
        """Train fair autoencoder"""
        
        # Prepare data
        X_norm = self.normalize_data(X)
        protected_encoded = self.encode_protected_attr(protected_attr)
        
        # Build models
        self.build_model(X.shape[1], protected_encoded.shape[1])
        
        # Training loop
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                # Forward pass
                encoded = self.encoder(X_norm)
                decoded = self.decoder(encoded)
                protected_pred = self.adversary(encoded)
                
                # Reconstruction loss
                reconstruction_loss = tf.keras.losses.mse(X_norm, decoded)
                
                # Adversarial loss (want to prevent protected attribute prediction)
                adversarial_loss = tf.keras.losses.categorical_crossentropy(
                    protected_encoded, protected_pred
                )
                
                # Combined loss
                total_loss = (
                    reconstruction_loss - 
                    self.fairness_weight * adversarial_loss
                )
            
            # Update encoder and decoder to minimize reconstruction, 
            # maximize adversarial loss
            encoder_vars = self.encoder.trainable_variables
            decoder_vars = self.decoder.trainable_variables
            adversary_vars = self.adversary.trainable_variables
            
            # Gradient updates
            grads = tape.gradient(total_loss, encoder_vars + decoder_vars)
            optimizer.apply_gradients(zip(grads, encoder_vars + decoder_vars))
            
            # Update adversary separately (minimize adversarial loss)
            with tf.GradientTape() as adv_tape:
                encoded = self.encoder(X_norm)
                protected_pred = self.adversary(encoded)
                adv_loss = tf.keras.losses.categorical_crossentropy(
                    protected_encoded, protected_pred
                )
            
            adv_grads = adv_tape.gradient(adv_loss, adversary_vars)
            optimizer.apply_gradients(zip(adv_grads, adversary_vars))
    
    def transform(self, X):
        """Transform data to fair representation"""
        X_norm = self.normalize_data(X)
        fair_representation = self.encoder(X_norm)
        return fair_representation.numpy()
    
    def normalize_data(self, X):
        """Normalize input data"""
        return (X - X.mean()) / X.std()
    
    def encode_protected_attr(self, protected_attr):
        """One-hot encode protected attributes"""
        return pd.get_dummies(protected_attr).values
```

## Component 5: Synthetic Data Generation

### Purpose
Generate fair synthetic data when traditional methods are insufficient.

### 5.1 Fairness-Aware GAN

```python
class FairGAN:
    """Generate fair synthetic data using conditional GAN"""
    
    def __init__(self, latent_dim=100, fairness_constraint='demographic_parity'):
        self.latent_dim = latent_dim
        self.fairness_constraint = fairness_constraint
        self.generator = None
        self.discriminator = None
        self.fairness_discriminator = None
    
    def build_generator(self, data_dim, protected_attr_dim):
        """Build generator network"""
        
        # Input: noise + protected attribute
        noise_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        protected_input = tf.keras.layers.Input(shape=(protected_attr_dim,))
        
        # Combine inputs
        combined_input = tf.keras.layers.Concatenate()([noise_input, protected_input])
        
        # Generator layers
        x = tf.keras.layers.Dense(128, activation='relu')(combined_input)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        output = tf.keras.layers.Dense(data_dim, activation='tanh')(x)
        
        self.generator = tf.keras.Model([noise_input, protected_input], output)
        
        return self.generator
    
    def build_discriminator(self, data_dim):
        """Build discriminator network"""
        
        data_input = tf.keras.layers.Input(shape=(data_dim,))
        
        x = tf.keras.layers.Dense(512, activation='relu')(data_input)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        
        # Real/fake classification
        validity = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        self.discriminator = tf.keras.Model(data_input, validity)
        
        return self.discriminator
    
    def build_fairness_discriminator(self, data_dim, protected_attr_dim):
        """Build fairness discriminator to enforce fairness constraints"""
        
        data_input = tf.keras.layers.Input(shape=(data_dim,))
        
        x = tf.keras.layers.Dense(256, activation='relu')(data_input)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        
        # Protected attribute prediction
        protected_pred = tf.keras.layers.Dense(
            protected_attr_dim, activation='softmax'
        )(x)
        
        self.fairness_discriminator = tf.keras.Model(data_input, protected_pred)
        
        return self.fairness_discriminator
    
    def fit(self, X, protected_attr, epochs=1000, batch_size=64):
        """Train Fair GAN"""
        
        # Prepare data
        X_norm = self.normalize_data(X)
        protected_encoded = pd.get_dummies(protected_attr).values
        
        # Build models
        self.build_generator(X.shape[1], protected_encoded.shape[1])
        self.build_discriminator(X.shape[1])
        self.build_fairness_discriminator(X.shape[1], protected_encoded.shape[1])
        
        # Optimizers
        g_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
        d_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
        f_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
        
        # Training loop
        for epoch in range(epochs):
            # Sample batch
            idx = np.random.randint(0, X_norm.shape[0], batch_size)
            real_data = X_norm[idx]
            real_protected = protected_encoded[idx]
            
            # Generate fake data
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_protected = real_protected  # Use same protected attributes
            fake_data = self.generator.predict([noise, fake_protected])
            
            # Train discriminator
            with tf.GradientTape() as d_tape:
                real_validity = self.discriminator(real_data)
                fake_validity = self.discriminator(fake_data)
                
                d_loss_real = tf.keras.losses.binary_crossentropy(
                    tf.ones_like(real_validity), real_validity
                )
                d_loss_fake = tf.keras.losses.binary_crossentropy(
                    tf.zeros_like(fake_validity), fake_validity
                )
                d_loss = d_loss_real + d_loss_fake
            
            d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
            d_optimizer.apply_gradients(
                zip(d_grads, self.discriminator.trainable_variables)
            )
            
            # Train fairness discriminator
            with tf.GradientTape() as f_tape:
                fake_protected_pred = self.fairness_discriminator(fake_data)
                f_loss = tf.keras.losses.categorical_crossentropy(
                    fake_protected, fake_protected_pred
                )
            
            f_grads = f_tape.gradient(f_loss, self.fairness_discriminator.trainable_variables)
            f_optimizer.apply_gradients(
                zip(f_grads, self.fairness_discriminator.trainable_variables)
            )
            
            # Train generator
            with tf.GradientTape() as g_tape:
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                fake_data = self.generator([noise, fake_protected])
                
                # Adversarial loss
                fake_validity = self.discriminator(fake_data)
                g_loss_adv = tf.keras.losses.binary_crossentropy(
                    tf.ones_like(fake_validity), fake_validity
                )
                
                # Fairness loss (want to fool fairness discriminator)
                fake_protected_pred = self.fairness_discriminator(fake_data)
                g_loss_fair = -tf.keras.losses.categorical_crossentropy(
                    fake_protected, fake_protected_pred
                )
                
                g_loss = g_loss_adv + 0.5 * g_loss_fair
            
            g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
            g_optimizer.apply_gradients(
                zip(g_grads, self.generator.trainable_variables)
            )
    
    def generate_fair_data(self, n_samples, protected_attr_distribution):
        """Generate fair synthetic data"""
        
        # Sample noise
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        
        # Sample protected attributes according to fair distribution
        protected_samples = []
        for attr_value, proportion in protected_attr_distribution.items():
            n_attr_samples = int(n_samples * proportion)
            attr_encoded = np.zeros(len(protected_attr_distribution))
            attr_encoded[list(protected_attr_distribution.keys()).index(attr_value)] = 1
            
            attr_batch = np.tile(attr_encoded, (n_attr_samples, 1))
            protected_samples.append(attr_batch)
        
        protected_encoded = np.vstack(protected_samples)
        
        # Generate synthetic data
        synthetic_data = self.generator.predict([noise[:len(protected_encoded)], protected_encoded])
        
        return self.denormalize_data(synthetic_data), protected_encoded
```

## Component 6: Integration and Validation

### Purpose
Orchestrate multiple pre-processing interventions and validate their effectiveness.

### 6.1 Multi-Intervention Pipeline

```python
class PreprocessingPipeline:
    """Orchestrate multiple pre-processing interventions"""
    
    def __init__(self):
        self.interventions = []
        self.fitted_interventions = []
        self.audit_report = None
    
    def add_intervention(self, intervention_type, **kwargs):
        """Add intervention to pipeline"""
        
        intervention_map = {
            'reweighting': InstanceWeighting,
            'resampling': FairnessSMOTE,
            'disparate_impact': DisparateImpactRemover,
            'fair_representation': FairAutoEncoder,
            'synthetic_generation': FairGAN
        }
        
        if intervention_type not in intervention_map:
            raise ValueError(f"Unknown intervention type: {intervention_type}")
        
        intervention_class = intervention_map[intervention_type]
        intervention = intervention_class(**kwargs)
        
        self.interventions.append({
            'type': intervention_type,
            'instance': intervention,
            'params': kwargs
        })
    
    def fit_transform(self, X, y, protected_attr):
        """Apply all interventions in sequence"""
        
        current_X, current_y, current_protected = X.copy(), y.copy(), protected_attr.copy()
        weights = None
        
        for intervention in self.interventions:
            intervention_instance = intervention['instance']
            intervention_type = intervention['type']
            
            if intervention_type == 'reweighting':
                intervention_instance.fit(current_X, current_y, current_protected)
                current_X, current_y, weights = intervention_instance.transform(
                    current_X, current_y, current_protected
                )
                
            elif intervention_type == 'resampling':
                current_X, current_y, current_protected = intervention_instance.fit_resample(
                    current_X, current_y, current_protected
                )
                
            elif intervention_type == 'disparate_impact':
                intervention_instance.fit(current_X, current_protected)
                current_X = intervention_instance.transform(current_X, current_protected)
                
            elif intervention_type == 'fair_representation':
                intervention_instance.fit(current_X, current_protected)
                current_X = intervention_instance.transform(current_X)
                
            elif intervention_type == 'synthetic_generation':
                intervention_instance.fit(current_X, current_protected)
                # Generate additional fair samples
                n_synthetic = len(current_X) // 2  # Add 50% synthetic data
                synthetic_X, synthetic_protected = intervention_instance.generate_fair_data(
                    n_synthetic, current_protected.value_counts(normalize=True).to_dict()
                )
                
                # Combine real and synthetic data
                current_X = np.vstack([current_X, synthetic_X])
                synthetic_y = np.random.choice(current_y, size=len(synthetic_X))
                current_y = np.concatenate([current_y, synthetic_y])
                current_protected = np.concatenate([current_protected, synthetic_protected.argmax(axis=1)])
            
            self.fitted_interventions.append(intervention_instance)
        
        return current_X, current_y, current_protected, weights
    
    def validate_interventions(self, X_original, y_original, protected_original,
                             X_processed, y_processed, protected_processed):
        """Validate effectiveness of interventions"""
        
        validation_results = {}
        
        # Fairness metrics comparison
        fairness_before = self.calculate_fairness_metrics(
            X_original, y_original, protected_original
        )
        fairness_after = self.calculate_fairness_metrics(
            X_processed, y_processed, protected_processed
        )
        
        validation_results['fairness_improvement'] = {
            metric: fairness_before[metric] - fairness_after[metric]
            for metric in fairness_before.keys()
        }
        
        # Data quality assessment
        validation_results['data_quality'] = {
            'distribution_shift': self.measure_distribution_shift(X_original, X_processed),
            'correlation_preservation': self.measure_correlation_preservation(X_original, X_processed),
            'sample_size_change': len(X_processed) - len(X_original)
        }
        
        return validation_results
    
    def calculate_fairness_metrics(self, X, y, protected_attr):
        """Calculate key fairness metrics"""
        
        metrics = {}
        
        # Demographic parity
        group_rates = {}
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            group_rates[group] = y[group_mask].mean()
        
        metrics['demographic_parity'] = max(group_rates.values()) - min(group_rates.values())
        
        # Representation fairness
        group_representation = protected_attr.value_counts(normalize=True)
        ideal_representation = 1.0 / len(protected_attr.unique())
        metrics['representation_fairness'] = max(
            abs(group_representation - ideal_representation)
        )
        
        return metrics
```

## Integration with Other Toolkits

### Inputs from Causal Fairness Toolkit
- **Bias mechanisms identified**: Direct, indirect, proxy discrimination
- **Causal pathways**: Which features are problematic and why
- **Intervention priorities**: Most impactful areas for pre-processing

### Outputs to In-Processing Toolkit
- **Cleaned datasets**: Bias-reduced training data
- **Residual bias patterns**: Issues that require model-level intervention
- **Feature importance**: Which features still need constraints

### Outputs to Post-Processing Toolkit
- **Group definitions**: Protected attribute encodings and intersectional groups
- **Baseline fairness**: Post pre-processing fairness metrics
- **Calibration needs**: Groups requiring threshold adjustments

## Documentation and Reporting

### Pre-Processing Report Template

```markdown
# Pre-Processing Intervention Report

## Executive Summary
- **Interventions Applied**: [List of techniques used]
- **Fairness Improvements**: [Key metric improvements]
- **Data Quality Impact**: [Changes in data characteristics]

## Detailed Results

### Bias Patterns Addressed
| Pattern | Intervention | Parameters | Effectiveness |
|---------|--------------|------------|---------------|
| Gender-Income correlation | Disparate Impact Removal | repair_level=0.7 | 65% reduction |
| Minority underrepresentation | SMOTE resampling | k_neighbors=5 | Balanced representation |

### Fairness Metrics
| Metric | Before | After | Improvement |
|--------|--------|--------|-------------|
| Demographic Parity | 0.18 | 0.05 | 72% reduction |
| Equal Opportunity | 0.13 | 0.04 | 69% reduction |

### Data Quality Assessment
- **Sample Size**: Original: 50,000 → Processed: 52,500 (+5%)
- **Feature Correlations**: 95% preserved for legitimate predictors
- **Distribution Shift**: Minimal for non-protected features

## Validation and Limitations
- **Cross-validation**: All improvements statistically significant (p < 0.01)
- **Limitations**: Synthetic data may not capture all real-world complexity
- **Monitoring**: Recommend quarterly fairness metric tracking
```

This comprehensive Pre-Processing Fairness Toolkit provides the foundation for addressing bias at the data level, setting up your AI system for fair and effective model training.