            if len(group_y_true) > 0:
                fpr, tpr, thresholds = roc_curve(group_y_true, group_y_scores)
                auc_score = auc(fpr, tpr)
                
                self.group_roc_curves[group] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': thresholds,
                    'auc': auc_score
                }
        
        # Select thresholds based on fairness constraint
        if self.fairness_constraint == 'equal_opportunity':
            self._select_equal_opportunity_thresholds()
        elif self.fairness_constraint == 'equalized_odds':
            self._select_equalized_odds_thresholds()
        elif self.fairness_constraint == 'equal_tpr_fpr':
            self._select_equal_tpr_fpr_thresholds()
        
        return self
    
    def _select_equal_opportunity_thresholds(self):
        """Select thresholds for equal TPR across groups"""
        
        # Find common TPR range across all groups
        min_max_tpr = min(max(curve['tpr']) for curve in self.group_roc_curves.values())
        
        # Try different target TPR values
        target_tprs = np.arange(0.1, min_max_tpr, 0.01)
        best_solution = None
        best_accuracy = 0
        
        for target_tpr in target_tprs:
            thresholds = {}
            accuracy_sum = 0
            valid_solution = True
            
            for group, roc_data in self.group_roc_curves.items():
                # Find threshold that gives closest TPR to target
                tpr_diff = np.abs(roc_data['tpr'] - target_tpr)
                closest_idx = np.argmin(tpr_diff)
                
                if tpr_diff[closest_idx] > self.tolerance:
                    valid_solution = False
                    break
                
                threshold = roc_data['thresholds'][closest_idx]
                thresholds[group] = threshold
                
                # Estimate accuracy at this threshold
                fpr_at_threshold = roc_data['fpr'][closest_idx]
                tpr_at_threshold = roc_data['tpr'][closest_idx]
                
                # Assume balanced classes for simplicity
                accuracy = 0.5 * tpr_at_threshold + 0.5 * (1 - fpr_at_threshold)
                accuracy_sum += accuracy
            
            if valid_solution:
                avg_accuracy = accuracy_sum / len(self.group_roc_curves)
                
                if avg_accuracy > best_accuracy:
                    best_accuracy = avg_accuracy
                    best_solution = {
                        'thresholds': thresholds,
                        'target_tpr': target_tpr,
                        'accuracy': avg_accuracy
                    }
        
        if best_solution:
            self.selected_thresholds = best_solution['thresholds']
        else:
            # Fallback: use default threshold for all groups
            self.selected_thresholds = {group: 0.5 for group in self.group_roc_curves.keys()}
    
    def _select_equalized_odds_thresholds(self):
        """Select thresholds for equal TPR and FPR across groups"""
        
        # This is more complex as we need to match both TPR and FPR
        # Use optimization approach
        
        from scipy.optimize import minimize
        
        groups = list(self.group_roc_curves.keys())
        n_groups = len(groups)
        
        def objective(threshold_indices):
            """Minimize TPR and FPR differences while maximizing accuracy"""
            
            tprs = []
            fprs = []
            accuracies = []
            
            for i, group in enumerate(groups):
                idx = int(threshold_indices[i])
                roc_data = self.group_roc_curves[group]
                
                if idx >= len(roc_data['tpr']):
                    idx = len(roc_data['tpr']) - 1
                
                tpr = roc_data['tpr'][idx]
                fpr = roc_data['fpr'][idx]
                
                tprs.append(tpr)
                fprs.append(fpr)
                
                # Estimate accuracy
                accuracy = 0.5 * tpr + 0.5 * (1 - fpr)
                accuracies.append(accuracy)
            
            # Fairness penalty (difference in TPRs and FPRs)
            tpr_penalty = max(tprs) - min(tprs)
            fpr_penalty = max(fprs) - min(fprs)
            
            # Objective: minimize fairness violations, maximize accuracy
            avg_accuracy = np.mean(accuracies)
            
            return (tpr_penalty + fpr_penalty) - 0.1 * avg_accuracy
        
        # Optimize
        initial_indices = [len(roc['tpr']) // 2 for roc in self.group_roc_curves.values()]
        bounds = [(0, len(roc['tpr']) - 1) for roc in self.group_roc_curves.values()]
        
        result = minimize(
            objective,
            initial_indices,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        if result.success:
            optimal_indices = [int(idx) for idx in result.x]
            thresholds = {}
            
            for i, group in enumerate(groups):
                roc_data = self.group_roc_curves[group]
                threshold_idx = optimal_indices[i]
                thresholds[group] = roc_data['thresholds'][threshold_idx]
            
            self.selected_thresholds = thresholds
        else:
            # Fallback
            self.selected_thresholds = {group: 0.5 for group in groups}
    
    def visualize_roc_curves(self):
        """Visualize ROC curves for all groups with selected thresholds"""
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.group_roc_curves)))
        
        for i, (group, roc_data) in enumerate(self.group_roc_curves.items()):
            plt.plot(
                roc_data['fpr'], 
                roc_data['tpr'], 
                color=colors[i], 
                label=f'{group} (AUC = {roc_data["auc"]:.3f})',
                linewidth=2
            )
            
            # Mark selected threshold
            if group in self.selected_thresholds:
                threshold = self.selected_thresholds[group]
                
                # Find closest point on ROC curve
                threshold_diffs = np.abs(roc_data['thresholds'] - threshold)
                closest_idx = np.argmin(threshold_diffs)
                
                selected_fpr = roc_data['fpr'][closest_idx]
                selected_tpr = roc_data['tpr'][closest_idx]
                
                plt.scatter(
                    selected_fpr, selected_tpr, 
                    color=colors[i], s=100, marker='o', 
                    edgecolor='black', linewidth=2,
                    label=f'{group} threshold = {threshold:.3f}'
                )
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves with Fair Threshold Selection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
```

## Component 2: Calibration and Score Adjustment

### Purpose
Adjust prediction scores to ensure fair calibration across protected groups.

### 2.1 Group-Specific Calibration

```python
class FairCalibrator:
    """Calibrate predictions to ensure fairness across groups"""
    
    def __init__(self, calibration_method='platt', fairness_metric='demographic_parity'):
        self.calibration_method = calibration_method
        self.fairness_metric = fairness_metric
        self.group_calibrators = {}
        self.calibration_curves = {}
    
    def fit(self, y_true, y_scores, protected_attr):
        """Fit group-specific calibrators"""
        
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            group_y_true = y_true[group_mask]
            group_y_scores = y_scores[group_mask]
            
            if len(group_y_true) > 10:  # Minimum samples for calibration
                if self.calibration_method == 'platt':
                    # Platt scaling (logistic regression)
                    calibrator = LogisticRegression()
                    calibrator.fit(group_y_scores.reshape(-1, 1), group_y_true)
                    
                elif self.calibration_method == 'isotonic':
                    # Isotonic regression
                    calibrator = IsotonicRegression(out_of_bounds='clip')
                    calibrator.fit(group_y_scores, group_y_true)
                    
                else:
                    raise ValueError(f"Unknown calibration method: {self.calibration_method}")
                
                self.group_calibrators[group] = calibrator
                
                # Store calibration curve for analysis
                self._compute_calibration_curve(group, group_y_true, group_y_scores)
            else:
                # Not enough samples, use identity mapping
                self.group_calibrators[group] = None
    
    def _compute_calibration_curve(self, group, y_true, y_scores):
        """Compute calibration curve for analysis"""
        
        from sklearn.calibration import calibration_curve
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_scores, n_bins=10, normalize=False
        )
        
        self.calibration_curves[group] = {
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value
        }
    
    def transform(self, y_scores, protected_attr):
        """Apply group-specific calibration"""
        
        calibrated_scores = np.zeros_like(y_scores)
        
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            
            if group in self.group_calibrators and self.group_calibrators[group] is not None:
                calibrator = self.group_calibrators[group]
                
                if self.calibration_method == 'platt':
                    group_calibrated = calibrator.predict_proba(
                        y_scores[group_mask].reshape(-1, 1)
                    )[:, 1]
                elif self.calibration_method == 'isotonic':
                    group_calibrated = calibrator.predict(y_scores[group_mask])
                
                calibrated_scores[group_mask] = group_calibrated
            else:
                # No calibration available, use original scores
                calibrated_scores[group_mask] = y_scores[group_mask]
        
        return calibrated_scores
    
    def visualize_calibration(self):
        """Visualize calibration curves before and after calibration"""
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot original calibration curves
        for group, curve_data in self.calibration_curves.items():
            ax1.plot(
                curve_data['mean_predicted_value'],
                curve_data['fraction_of_positives'],
                marker='o',
                label=f'{group} (original)',
                linewidth=2
            )
        
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Original Calibration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # For calibrated curves, we'd need to recompute after transformation
        # This is a simplified visualization
        ax2.text(0.5, 0.5, 'Calibrated curves\nwould be shown here\nafter applying\ntransformation',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Calibrated Curves')
        
        plt.tight_layout()
        return fig

class EqualizingOddsPostProcessor:
    """Post-processor that equalizes odds across groups"""
    
    def __init__(self, tolerance=0.05):
        self.tolerance = tolerance
        self.mixing_rates = {}
        self.base_rates = {}
    
    def fit(self, y_true, y_scores, protected_attr, threshold=0.5):
        """Learn optimal mixing rates for equalizing odds"""
        
        # Calculate base rates and confusion matrices for each group
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            group_y_true = y_true[group_mask]
            group_y_scores = y_scores[group_mask]
            group_y_pred = (group_y_scores >= threshold).astype(int)
            
            # Calculate confusion matrix elements
            tp = ((group_y_true == 1) & (group_y_pred == 1)).sum()
            fp = ((group_y_true == 0) & (group_y_pred == 1)).sum()
            tn = ((group_y_true == 0) & (group_y_pred == 0)).sum()
            fn = ((group_y_true == 1) & (group_y_pred == 0)).sum()
            
            # Calculate rates
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            base_rate = (group_y_true == 1).mean()
            
            self.base_rates[group] = {
                'tpr': tpr,
                'fpr': fpr,
                'base_rate': base_rate,
                'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
            }
        
        # Find mixing rates to equalize odds
        self._compute_mixing_rates()
        
        return self
    
    def _compute_mixing_rates(self):
        """Compute optimal mixing rates using linear programming"""
        
        groups = list(self.base_rates.keys())
        
        if len(groups) != 2:
            raise ValueError("Equalizing odds currently supports only 2 groups")
        
        group_0, group_1 = groups
        
        # Get rates for both groups
        tpr_0 = self.base_rates[group_0]['tpr']
        fpr_0 = self.base_rates[group_0]['fpr']
        tpr_1 = self.base_rates[group_1]['tpr']
        fpr_1 = self.base_rates[group_1]['fpr']
        
        # Solve for mixing rates
        # For group 0: mixed_tpr_0 = p0 * tpr_0 + (1-p0) * base_rate_0
        # For group 1: mixed_tpr_1 = p1 * tpr_1 + (1-p1) * base_rate_1
        # Constraint: mixed_tpr_0 = mixed_tpr_1, mixed_fpr_0 = mixed_fpr_1
        
        base_rate_0 = self.base_rates[group_0]['base_rate']
        base_rate_1 = self.base_rates[group_1]['base_rate']
        
        # Simplified approach: find mixing rates that minimize differences
        from scipy.optimize import minimize
        
        def objective(params):
            p0, p1 = params
            
            # Mixed TPRs
            mixed_tpr_0 = p0 * tpr_0 + (1 - p0) * base_rate_0
            mixed_tpr_1 = p1 * tpr_1 + (1 - p1) * base_rate_1
            
            # Mixed FPRs
            mixed_fpr_0 = p0 * fpr_0 + (1 - p0) * (1 - base_rate_0)
            mixed_fpr_1 = p1 * fpr_1 + (1 - p1) * (1 - base_rate_1)
            
            # Penalty for differences
            tpr_diff = abs(mixed_tpr_0 - mixed_tpr_1)
            fpr_diff = abs(mixed_fpr_0 - mixed_fpr_1)
            
            return tpr_diff + fpr_diff
        
        # Constraints: 0 <= p0, p1 <= 1
        bounds = [(0, 1), (0, 1)]
        initial_guess = [1.0, 1.0]
        
        result = minimize(objective, initial_guess, bounds=bounds)
        
        if result.success:
            self.mixing_rates[group_0] = result.x[0]
            self.mixing_rates[group_1] = result.x[1]
        else:
            # Fallback: no mixing
            self.mixing_rates[group_0] = 1.0
            self.mixing_rates[group_1] = 1.0
    
    def transform(self, y_scores, protected_attr):
        """Apply equalized odds transformation"""
        
        adjusted_scores = y_scores.copy()
        
        for group in protected_attr.unique():
            if group in self.mixing_rates:
                group_mask = protected_attr == group
                mixing_rate = self.mixing_rates[group]
                base_rate = self.base_rates[group]['base_rate']
                
                # Apply mixing: combine original predictions with base rate
                group_scores = y_scores[group_mask]
                mixed_scores = (mixing_rate * group_scores + 
                               (1 - mixing_rate) * base_rate)
                
                adjusted_scores[group_mask] = mixed_scores
        
        return adjusted_scores
```

## Component 3: Bias Amplification Prevention

### Purpose
Prevent post-processing adjustments from amplifying existing biases or creating new ones.

### 3.1 Bias Amplification Detector

```python
class BiasAmplificationDetector:
    """Detect and prevent bias amplification in post-processing"""
    
    def __init__(self, amplification_threshold=0.1):
        self.amplification_threshold = amplification_threshold
        self.baseline_metrics = {}
        self.post_process_metrics = {}
        self.amplification_results = {}
    
    def analyze_amplification(self, y_true, y_scores_original, y_scores_processed, 
                            protected_attr, threshold=0.5):
        """Analyze if post-processing amplifies bias"""
        
        # Calculate baseline metrics
        y_pred_original = (y_scores_original >= threshold).astype(int)
        y_pred_processed = (y_scores_processed >= threshold).astype(int)
        
        self.baseline_metrics = self._calculate_group_metrics(
            y_true, y_pred_original, protected_attr
        )
        
        self.post_process_metrics = self._calculate_group_metrics(
            y_true, y_pred_processed, protected_attr
        )
        
        # Detect amplification
        self._detect_amplification()
        
        return self.amplification_results
    
    def _calculate_group_metrics(self, y_true, y_pred, protected_attr):
        """Calculate fairness metrics for each group"""
        
        metrics = {}
        
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            if len(group_y_true) > 0:
                # Basic metrics
                accuracy = (group_y_pred == group_y_true).mean()
                positive_rate = group_y_pred.mean()
                
                # Fairness-specific metrics
                if (group_y_true == 1).sum() > 0:
                    tpr = group_y_pred[group_y_true == 1].mean()
                else:
                    tpr = 0
                
                if (group_y_true == 0).sum() > 0:
                    fpr = group_y_pred[group_y_true == 0].mean()
                else:
                    fpr = 0
                
                metrics[group] = {
                    'accuracy': accuracy,
                    'positive_rate': positive_rate,
                    'tpr': tpr,
                    'fpr': fpr,
                    'sample_size': len(group_y_true)
                }
        
        return metrics
    
    def _detect_amplification(self):
        """Detect bias amplification across different metrics"""
        
        amplification_flags = {}
        
        # Check each metric for amplification
        metric_names = ['accuracy', 'positive_rate', 'tpr', 'fpr']
        
        for metric in metric_names:
            baseline_disparities = self._calculate_disparities(self.baseline_metrics, metric)
            processed_disparities = self._calculate_disparities(self.post_process_metrics, metric)
            
            amplification_flags[metric] = {}
            
            for group_pair, baseline_gap in baseline_disparities.items():
                processed_gap = processed_disparities.get(group_pair, 0)
                
                # Check if gap increased significantly
                amplification = processed_gap - baseline_gap
                is_amplified = amplification > self.amplification_threshold
                
                amplification_flags[metric][group_pair] = {
                    'baseline_gap': baseline_gap,
                    'processed_gap': processed_gap,
                    'amplification': amplification,
                    'is_amplified': is_amplified
                }
        
        self.amplification_results = amplification_flags
    
    def _calculate_disparities(self, metrics, metric_name):
        """Calculate pairwise disparities for a specific metric"""
        
        disparities = {}
        groups = list(metrics.keys())
        
        for i, group_a in enumerate(groups):
            for j, group_b in enumerate(groups):
                if i < j:
                    value_a = metrics[group_a].get(metric_name, 0)
                    value_b = metrics[group_b].get(metric_name, 0)
                    
                    disparity = abs(value_a - value_b)
                    disparities[f"{group_a}_vs_{group_b}"] = disparity
        
        return disparities
    
    def get_amplification_summary(self):
        """Get summary of bias amplification analysis"""
        
        summary = {
            'total_amplifications': 0,
            'amplified_metrics': [],
            'most_affected_groups': {},
            'recommendations': []
        }
        
        for metric, pairs in self.amplification_results.items():
            metric_amplifications = sum(
                1 for pair_data in pairs.values() if pair_data['is_amplified']
            )
            
            if metric_amplifications > 0:
                summary['amplified_metrics'].append(metric)
                summary['total_amplifications'] += metric_amplifications
                
                # Find most affected group pairs
                max_amplification = max(
                    pair_data['amplification'] for pair_data in pairs.values()
                )
                
                most_affected_pair = [
                    pair for pair, data in pairs.items() 
                    if data['amplification'] == max_amplification
                ][0]
                
                summary['most_affected_groups'][metric] = {
                    'group_pair': most_affected_pair,
                    'amplification': max_amplification
                }
        
        # Generate recommendations
        if summary['total_amplifications'] > 0:
            summary['recommendations'] = [
                "Consider adjusting post-processing parameters",
                "Implement bias-aware post-processing constraints",
                "Use different fairness objectives",
                "Apply regularization to prevent amplification"
            ]
        else:
            summary['recommendations'] = [
                "Current post-processing appears safe from bias amplification"
            ]
        
        return summary

class AmplificationAwarePostProcessor:
    """Post-processor with bias amplification prevention"""
    
    def __init__(self, base_post_processor, amplification_threshold=0.1):
        self.base_post_processor = base_post_processor
        self.amplification_threshold = amplification_threshold
        self.detector = BiasAmplificationDetector(amplification_threshold)
        self.safe_parameters = None
    
    def fit(self, y_true, y_scores, protected_attr):
        """Fit with amplification-aware parameter tuning"""
        
        # Try different parameter settings
        parameter_candidates = self._generate_parameter_candidates()
        best_params = None
        best_score = float('-inf')
        
        for params in parameter_candidates:
            # Configure base post-processor with these parameters
            self._set_post_processor_params(params)
            
            # Fit and transform
            self.base_post_processor.fit(y_true, y_scores, protected_attr)
            processed_scores = self.base_post_processor.transform(y_scores, protected_attr)
            
            # Check for amplification
            amplification_results = self.detector.analyze_amplification(
                y_true, y_scores, processed_scores, protected_attr
            )
            
            # Score this configuration
            score = self._score_configuration(amplification_results, y_true, processed_scores, protected_attr)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        # Set best parameters
        if best_params is not None:
            self._set_post_processor_params(best_params)
            self.base_post_processor.fit(y_true, y_scores, protected_attr)
            self.safe_parameters = best_params
        
        return self
    
    def _generate_parameter_candidates(self):
        """Generate parameter candidates for tuning"""
        
        # This is specific to the post-processor type
        # Example for threshold optimizer
        if hasattr(self.base_post_processor, 'tolerance'):
            tolerances = [0.01, 0.03, 0.05, 0.1]
            return [{'tolerance': tol} for tol in tolerances]
        else:
            # Default: try original parameters
            return [{}]
    
    def _set_post_processor_params(self, params):
        """Set parameters on base post-processor"""
        
        for param, value in params.items():
            if hasattr(self.base_post_processor, param):
                setattr(self.base_post_processor, param, value)
    
    def _score_configuration(self, amplification_results, y_true, y_scores, protected_attr):
        """Score a parameter configuration"""
        
        # Count amplifications
        total_amplifications = 0
        for metric_results in amplification_results.values():
            total_amplifications += sum(
                1 for result in metric_results.values() if result['is_amplified']
            )
        
        # Calculate accuracy
        y_pred = (y_scores >= 0.5).astype(int)
        accuracy = (y_pred == y_true).mean()
        
        # Combined score: penalize amplifications, reward accuracy
        amplification_penalty = total_amplifications * 0.1
        score = accuracy - amplification_penalty
        
        return score
    
    def transform(self, y_scores, protected_attr):
        """Apply safe transformation"""
        
        return self.base_post_processor.transform(y_scores, protected_attr)
```

## Component 4: Multi-Constraint Post-Processing

### Purpose
Handle multiple fairness constraints simultaneously in post-processing.

### 4.1 Multi-Objective Post-Processing

```python
class MultiConstraintPostProcessor:
    """Post-processor handling multiple fairness constraints"""
    
    def __init__(self, constraints=None, constraint_weights=None):
        self.constraints = constraints or ['demographic_parity', 'equal_opportunity']
        self.constraint_weights = constraint_weights or [1.0] * len(self.constraints)
        self.optimal_thresholds = {}
        self.constraint_violations = {}
    
    def fit(self, y_true, y_scores, protected_attr, method='weighted_optimization'):
        """Fit multi-constraint post-processor"""
        
        if method == 'weighted_optimization':
            self._fit_weighted_optimization(y_true, y_scores, protected_attr)
        elif method == 'lexicographic':
            self._fit_lexicographic_optimization(y_true, y_scores, protected_attr)
        elif method == 'pareto_optimal':
            self._fit_pareto_optimization(y_true, y_scores, protected_attr)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        return self
    
    def _fit_weighted_optimization(self, y_true, y_scores, protected_attr):
        """Weighted combination of constraints"""
        
        from scipy.optimize import minimize
        
        groups = list(protected_attr.unique())
        n_groups = len(groups)
        
        def objective(thresholds):
            """Weighted combination of constraint violations and accuracy loss"""
            
            # Calculate predictions for each group
            total_violation = 0
            total_accuracy = 0
            total_samples = 0
            
            group_predictions = {}
            group_metrics = {}
            
            for i, group in enumerate(groups):
                group_mask = protected_attr == group
                group_y_true = y_true[group_mask]
                group_y_scores = y_scores[group_mask]
                
                if len(group_y_true) == 0:
                    continue
                
                threshold = thresholds[i]
                group_pred = (group_y_scores >= threshold).astype(int)
                group_predictions[group] = group_pred
                
                # Calculate metrics
                accuracy = (group_pred == group_y_true).mean()
                total_accuracy += accuracy * len(group_y_true)
                total_samples += len(group_y_true)
                
                # Store group metrics for constraint calculation
                positive_rate = group_pred.mean()
                
                if (group_y_true == 1).sum() > 0:
                    tpr = group_pred[group_y_true == 1].mean()
                else:
                    tpr = 0
                
                group_metrics[group] = {
                    'positive_rate': positive_rate,
                    'tpr': tpr
                }
            
            # Calculate constraint violations
            for constraint, weight in zip(self.constraints, self.constraint_weights):
                violation = self._calculate_constraint_violation(constraint, group_metrics)
                total_violation += weight * violation
            
            # Overall accuracy
            overall_accuracy = total_accuracy / total_samples if total_samples > 0 else 0
            
            # Objective: minimize violations while maximizing accuracy
            return total_violation - 0.1 * overall_accuracy
        
        # Optimize thresholds
        initial_thresholds = [0.5] * n_groups
        bounds = [(0.1, 0.9)] * n_groups
        
        result = minimize(
            objective,
            initial_thresholds,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        if result.success:
            self.optimal_thresholds = dict(zip(groups, result.x))
        else:
            self.optimal_thresholds = {group: 0.5 for group in groups}
    
    def _fit_lexicographic_optimization(self, y_true, y_scores, protected_attr):
        """Lexicographic optimization (constraints in priority order)"""
        
        groups = list(protected_attr.unique())
        current_thresholds = {group: 0.5 for group in groups}
        
        # Optimize constraints in order of priority
        for i, constraint in enumerate(self.constraints):
            current_thresholds = self._optimize_single_constraint(
                constraint, y_true, y_scores, protected_attr, current_thresholds
            )
        
        self.optimal_thresholds = current_thresholds
    
    def _optimize_single_constraint(self, constraint, y_true, y_scores, protected_attr, 
                                  initial_thresholds):
        """Optimize a single constraint"""
        
        from scipy.optimize import minimize
        
        groups = list(protected_attr.unique())
        
        def single_constraint_objective(thresholds):
            group_metrics = {}
            
            for i, group in enumerate(groups):
                group_mask = protected_attr == group
                group_y_true = y_true[group_mask]
                group_y_scores = y_scores[group_mask]
                
                if len(group_y_true) == 0:
                    continue
                
                threshold = thresholds[i]
                group_pred = (group_y_scores >= threshold).astype(int)
                
                positive_rate = group_pred.mean()
                
                if (group_y_true == 1).sum() > 0:
                    tpr = group_pred[group_y_true == 1].mean()
                else:
                    tpr = 0
                
                group_metrics[group] = {
                    'positive_rate': positive_rate,
                    'tpr': tpr
                }
            
            return self._calculate_constraint_violation(constraint, group_metrics)
        
        # Optimize
        initial_values = [initial_thresholds[group] for group in groups]
        bounds = [(0.1, 0.9)] * len(groups)
        
        result = minimize(
            single_constraint_objective,
            initial_values,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        if result.success:
            return dict(zip(groups, result.x))
        else:
            return initial_thresholds
    
    def _calculate_constraint_violation(self, constraint, group_metrics):
        """Calculate violation for a specific constraint"""
        
        if constraint == 'demographic_parity':
            rates = [metrics['positive_rate'] for metrics in group_metrics.values()]
            return max(rates) - min(rates) if rates else 0
        
        elif constraint == 'equal_opportunity':
            tprs = [metrics['tpr'] for metrics in group_metrics.values()]
            return max(tprs) - min(tprs) if tprs else 0
        
        else:
            return 0
    
    def predict(self, y_scores, protected_attr):
        """Make predictions using multi-constraint thresholds"""
        
        predictions = np.zeros(len(y_scores))
        
        for group, threshold in self.optimal_thresholds.items():
            group_mask = protected_attr == group
            predictions[group_mask] = (y_scores[group_mask] >= threshold).astype(int)
        
        return predictions
    
    def get_constraint_analysis(self, y_true, y_scores, protected_attr):
        """Analyze how well constraints are satisfied"""
        
        predictions = self.predict(y_scores, protected_attr)
        
        analysis = {}
        
        for constraint in self.constraints:
            group_metrics = {}
            
            for group in protected_attr.unique():
                group_mask = protected_attr == group
                group_y_true = y_true[group_mask]
                group_pred = predictions[group_mask]
                
                if len(group_y_true) > 0:
                    positive_rate = group_pred.mean()
                    
                    if (group_y_true == 1).sum() > 0:
                        tpr = group_pred[group_y_true == 1].mean()
                    else:
                        tpr = 0
                    
                    group_metrics[group] = {
                        'positive_rate': positive_rate,
                        'tpr': tpr
                    }
            
            violation = self._calculate_constraint_violation(constraint, group_metrics)
            
            analysis[constraint] = {
                'violation': violation,
                'group_metrics': group_metrics,
                'satisfied': violation <= 0.05  # Default tolerance
            }
        
        return analysis
```

## Component 5: Integration and Validation Framework

### Purpose
Coordinate multiple post-processing techniques and validate their effectiveness.

### 5.1 Integrated Post-Processing Pipeline

```python
class PostProcessingPipeline:
    """Integrated pipeline for post-processing fairness interventions"""
    
    def __init__(self, intervention_strategy='adaptive'):
        self.intervention_strategy = intervention_strategy
        self.fitted_components = {}
        self.final_processor = None
        self.validation_results = {}
    
    def fit(self, y_true, y_scores, protected_attr, fairness_goals=None):
        """Fit integrated post-processing pipeline"""
        
        fairness_goals = fairness_goals or {
            'demographic_parity': 0.05,
            'equal_opportunity': 0.05
        }
        
        if self.intervention_strategy == 'threshold_only':
            # Use threshold optimization only
            processor = FairThresholdOptimizer(
                fairness_metric='equal_opportunity'
            )
            processor.fit(y_true, y_scores, protected_attr)
            self.final_processor = processor
            
        elif self.intervention_strategy == 'calibration_only':
            # Use calibration only
            processor = FairCalibrator(
                calibration_method='platt',
                fairness_metric='demographic_parity'
            )
            processor.fit(y_true, y_scores, protected_attr)
            self.final_processor = processor
            
        elif self.intervention_strategy == 'multi_constraint':
            # Use multi-constraint optimization
            processor = MultiConstraintPostProcessor(
                constraints=list(fairness_goals.keys()),
                constraint_weights=[1.0] * len(fairness_goals)
            )
            processor.fit(y_true, y_scores, protected_attr)
            self.final_processor = processor
            
        elif self.intervention_strategy == 'adaptive':
            # Choose best approach based on data characteristics
            self._fit_adaptive_approach(y_true, y_scores, protected_attr, fairness_goals)
            
        else:
            raise ValueError(f"Unknown intervention strategy: {self.intervention_strategy}")
        
        # Validate results
        self._validate_intervention(y_true, y_scores, protected_attr, fairness_goals)
        
        return self
    
    def _fit_adaptive_approach(self, y_true, y_scores, protected_attr, fairness_goals):
        """Adaptively choose best post-processing approach"""
        
        # Analyze data characteristics
        data_analysis = self._analyze_data_characteristics(y_true, y_scores, protected_attr)
        
        # Try different approaches
        candidates = {}
        
        # 1. Threshold optimization
        threshold_optimizer = FairThresholdOptimizer(fairness_metric='equal_opportunity')
        threshold_optimizer.fit(y_true, y_scores, protected_attr)
        candidates['threshold'] = threshold_optimizer
        
        # 2. Calibration approach
        if data_analysis['calibration_needed']:
            calibrator = FairCalibrator(calibration_method='platt')
            calibrator.fit(y_true, y_scores, protected_attr)
            candidates['calibration'] = calibrator
        
        # 3. Multi-constraint approach
        if len(fairness_goals) > 1:
            multi_processor = MultiConstraintPostProcessor(
                constraints=list(fairness_goals.keys())
            )
            multi_processor.fit(y_true, y_scores, protected_attr)
            candidates['multi_constraint'] = multi_processor
        
        # 4. Amplification-aware approach
        if data_analysis['amplification_risk']:
            base_processor = FairThresholdOptimizer()
            safe_processor = AmplificationAwarePostProcessor(base_processor)
            safe_processor.fit(y_true, y_scores, protected_attr)
            candidates['amplification_aware'] = safe_processor
        
        # Evaluate candidates
        best_processor = self._select_best_processor(
            candidates, y_true, y_scores, protected_attr, fairness_goals
        )
        
        self.final_processor = best_processor
        self.fitted_components = candidates
    
    def _analyze_data_characteristics(self, y_true, y_scores, protected_attr):
        """Analyze data to guide approach selection"""
        
        analysis = {
            'calibration_needed': False,
            'amplification_risk': False,
            'group_sizes': {},
            'baseline_fairness': {}
        }
        
        # Check calibration
        from sklearn.calibration import calibration_curve
        
        overall_fraction_pos, overall_mean_pred = calibration_curve(
            y_true, y_scores, n_bins=5
        )
        
        # Simple calibration check
        calibration_error = np.mean(np.abs(overall_fraction_pos - overall_mean_pred))
        analysis['calibration_needed'] = calibration_error > 0.1
        
        # Check group sizes
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            analysis['group_sizes'][group] = group_mask.sum()
        
        # Small groups indicate amplification risk
        min_group_size = min(analysis['group_sizes'].values())
        analysis['amplification_risk'] = min_group_size < 100
        
        # Baseline fairness assessment
        y_pred_baseline = (y_scores >= 0.5).astype(int)
        
        # Demographic parity
        group_rates = {}
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            group_rates[group] = y_pred_baseline[group_mask].mean()
        
        dp_violation = max(group_rates.values()) - min(group_rates.values())
        analysis['baseline_fairness']['demographic_parity'] = dp_violation
        
        return analysis
    
    def _select_best_processor(self, candidates, y_true, y_scores, protected_attr, fairness_goals):
        """Select best processor based on performance"""
        
        best_processor = None
        best_score = float('-inf')
        
        for name, processor in candidates.items():
            try:
                # Get processed predictions
                if hasattr(processor, 'predict'):
                    processed_pred = processor.predict(y_scores, protected_attr)
                elif hasattr(processor, 'transform'):
                    processed_scores = processor.transform(y_scores, protected_attr)
                    processed_pred = (processed_scores >= 0.5).astype(int)
                else:
                    continue
                
                # Calculate score
                score = self._score_processor(
                    y_true, processed_pred, protected_attr, fairness_goals
                )
                
                if score > best_score:
                    best_score = score
                    best_processor = processor
                    
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                continue
        
        return best_processor if best_processor is not None else candidates.get('threshold')
    
    def _score_processor(self, y_true, y_pred, protected_attr, fairness_goals):
        """Score a processor based on fairness and accuracy"""
        
        # Accuracy
        accuracy = (y_pred == y_true).mean()
        
        # Fairness violations
        total_violation = 0
        
        for goal, threshold in fairness_goals.items():
            if goal == 'demographic_parity':
                group_rates = {}
                for group in protected_attr.unique():
                    group_mask = protected_attr == group
                    group_rates[group] = y_pred[group_mask].mean()
                
                violation = max(group_rates.values()) - min(group_rates.values())
                total_violation += max(0, violation - threshold)
            
            elif goal == 'equal_opportunity':
                tprs = {}
                for group in protected_attr.unique():
                    group_mask = protected_attr == group
                    positive_mask = y_true == 1
                    group_positive_mask = group_mask & positive_mask
                    
                    if group_positive_mask.sum() > 0:
                        tpr = y_pred[group_positive_mask].mean()
                        tprs[group] = tpr
                
                if len(tprs) >= 2:
                    violation = max(tprs.values()) - min(tprs.values())
                    total_violation += max(0, violation - threshold)
        
        # Combined score: high accuracy, low violations
        score = accuracy - 2 * total_violation
        
        return score
    
    def _validate_intervention(self, y_true, y_scores, protected_attr, fairness_goals):
        """Validate the intervention results"""
        
        # Get final predictions
        if hasattr(self.final_processor, 'predict'):
            final_pred = self.final_processor.predict(y_scores, protected_attr)
        elif hasattr(self.final_processor, 'transform'):
            processed_scores = self.final_processor.transform(y_scores, protected_attr)
            final_pred = (processed_scores >= 0.5).astype(int)
        else:
            final_pred = (y_scores >= 0.5).astype(int)
        
        # Baseline predictions
        baseline_pred = (y_scores >= 0.5).astype(int)
        
        # Calculate validation metrics
        self.validation_results = {
            'accuracy': {
                'baseline': (baseline_pred == y_true).mean(),
                'final': (final_pred == y_true).mean()
            },
            'fairness_metrics': {},
            'group_performance': {}
        }
        
        # Fairness metrics
        for goal in fairness_goals.keys():
            baseline_violation = self._calculate_fairness_violation(
                goal, y_true, baseline_pred, protected_attr
            )
            final_violation = self._calculate_fairness_violation(
                goal, y_true, final_pred, protected_attr
            )
            
            self.validation_results['fairness_metrics'][goal] = {
                'baseline': baseline_violation,
                'final': final_violation,
                'improvement': baseline_violation - final_violation
            }
        
        # Group-specific performance
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            
            baseline_acc = (baseline_pred[group_mask] == y_true[group_mask]).mean()
            final_acc = (final_pred[group_mask] == y_true[group_mask]).mean()
            
            self.validation_results['group_performance'][group] = {
                'baseline_accuracy': baseline_acc,
                'final_accuracy': final_acc,
                'accuracy_change': final_acc - baseline_acc,
                'sample_size': group_mask.sum()
            }
    
    def _calculate_fairness_violation(self, metric, y_true, y_pred, protected_attr):
        """Calculate fairness violation for a specific metric"""
        
        if metric == 'demographic_parity':
            group_rates = {}
            for group in protected_attr.unique():
                group_mask = protected_attr == group
                group_rates[group] = y_pred[group_mask].mean()
            
            return max(group_rates.values()) - min(group_rates.values())
        
        elif metric == 'equal_opportunity':
            tprs = {}
            for group in protected_attr.unique():
                group_mask = protected_attr == group
                positive_mask = y_true == 1
                group_positive_mask = group_mask & positive_mask
                
                if group_positive_mask.sum() > 0:
                    tpr = y_pred[group_positive_mask].mean()
                    tprs[group] = tpr
            
            if len(tprs) >= 2:
                return max(tprs.values()) - min(tprs.values())
            else:
                return 0
        
        return 0
    
    def predict(self, y_scores, protected_attr):
        """Make final predictions"""
        
        if hasattr(self.final_processor, 'predict'):
            return self.final_processor.predict(y_scores, protected_attr)
        elif hasattr(self.final_processor, 'transform'):
            processed_scores = self.final_processor.transform(y_scores, protected_attr)
            return (processed_scores >= 0.5).astype(int)
        else:
            return (y_scores >= 0.5).astype(int)
    
    def get_intervention_summary(self):
        """Get summary of post-processing intervention"""
        
        summary = {
            'strategy_used': self.intervention_strategy,
            'final_processor_type': type(self.final_processor).__name__,
            'components_tried': list(self.fitted_components.keys()) if self.fitted_components else [],
            'validation_results': self.validation_results
        }
        
        return summary
```

## Integration with Other Toolkits

### Inputs from Previous Toolkits
- **Pre-processed data**: Clean training datasets from Pre-Processing Toolkit
- **Trained models**: Fair models from In-Processing Toolkit
- **Causal insights**: Understanding of bias mechanisms from Causal Fairness Toolkit
- **Residual bias patterns**: Issues that require post-processing adjustment

### Outputs for Monitoring and Deployment
- **Fair predictions**: Adjusted outputs meeting fairness criteria
- **Threshold configurations**: Group-specific decision rules
- **Calibration mappings**: Score adjustment functions
- **Monitoring specifications**: Key metrics to track in production

## Documentation Template

```markdown
# Post-Processing Intervention Report

## Executive Summary
- **Intervention Strategy**: [threshold/calibration/multi-constraint/adaptive]
- **Fairness Goals**: [target metrics and thresholds]
- **Achieved Results**: [fairness improvements and accuracy trade-offs]

## Intervention Details

### Approach Selection
- **Data Characteristics**: [calibration needs, group sizes, baseline fairness]
- **Rationale**: [why this approach was chosen]
- **Alternative Approaches**: [other methods considered]

### Implementation
| Component | Configuration | Purpose |
|-----------|---------------|---------|
| Threshold Optimization | Equal opportunity, tolerance=0.05 | Equalize TPR across groups |
| Calibration | Platt scaling, group-specific | Ensure fair probability estimates |

## Results Analysis

### Fairness Metrics
| Metric | Baseline | After Post-Processing | Improvement |
|--------|----------|----------------------|-------------|
| Demographic Parity | 0.15 | 0.04 | 73% reduction |
| Equal Opportunity | 0.12 | 0.03 | 75% reduction |

### Performance Impact
| Group | Baseline Accuracy | Final Accuracy | Change |
|-------|------------------|----------------|--------|
| Group A | 85.2% | 84.8% | -0.4% |
| Group B | 79.1% | 83.2% | +4.1% |

### Threshold Analysis
- **Group A Threshold**: 0.52 (↑ from 0.50)
- **Group B Threshold**: 0.41 (↓ from 0.50)
- **Rationale**: Compensate for historical bias in Group B

## Validation and Monitoring
- **Cross-validation**: Results stable across different data splits
- **Robustness**: Performance maintained under reasonable distribution shifts
- **Amplification Check**: No bias amplification detected
- **Monitoring Plan**: Track fairness metrics weekly, retune thresholds quarterly

## Deployment Considerations
- **Integration**: Threshold adjustment layer in prediction pipeline
- **Performance**: <1ms overhead per prediction
- **Maintenance**: Monthly fairness metric review, quarterly threshold optimization
```

This Post-Processing Fairness Toolkit provides comprehensive methods for achieving fairness through output adjustment, enabling rapid deployment of fair AI systems even when upstream interventions are not possible.# Post-Processing Fairness Toolkit
## Adjusting Model Outputs for Fair Outcomes

### Overview

The Post-Processing Fairness Toolkit provides methods to adjust model predictions after training to achieve fairness goals. This approach is particularly valuable when you cannot modify training data or model architecture, or when you need to quickly address fairness issues in deployed models.

### When to Use This Toolkit

- Working with pre-trained models that cannot be retrained
- Need rapid fairness fixes for production systems
- Pre-processing and in-processing interventions are insufficient
- Require different fairness criteria for different deployment contexts
- Working with third-party models or APIs
- Need to satisfy specific regulatory requirements post-deployment

### Core Components

## Component 1: Threshold Optimization

### Purpose
Optimize decision thresholds for different groups to achieve fairness while maintaining overall performance.

### 1.1 Group-Specific Threshold Optimization

```python
class FairThresholdOptimizer:
    """Optimize thresholds for fair decision making"""
    
    def __init__(self, fairness_metric='equal_opportunity'):
        self.fairness_metric = fairness_metric
        self.thresholds = {}
        self.optimization_results = {}
    
    def fit(self, y_true, y_scores, protected_attr, method='grid_search'):
        """Find optimal thresholds for each group"""
        
        if method == 'grid_search':
            self._fit_grid_search(y_true, y_scores, protected_attr)
        elif method == 'constrained_optimization':
            self._fit_constrained_optimization(y_true, y_scores, protected_attr)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        return self
    
    def _fit_grid_search(self, y_true, y_scores, protected_attr):
        """Grid search optimization for thresholds"""
        
        # Define threshold candidates
        threshold_candidates = np.arange(0.1, 0.9, 0.01)
        
        if self.fairness_metric == 'equal_opportunity':
            self._optimize_equal_opportunity(y_true, y_scores, protected_attr, threshold_candidates)
        elif self.fairness_metric == 'demographic_parity':
            self._optimize_demographic_parity(y_true, y_scores, protected_attr, threshold_candidates)
        elif self.fairness_metric == 'equalized_odds':
            self._optimize_equalized_odds(y_true, y_scores, protected_attr, threshold_candidates)
        else:
            raise ValueError(f"Unsupported fairness metric: {self.fairness_metric}")
    
    def _optimize_equal_opportunity(self, y_true, y_scores, protected_attr, candidates):
        """Optimize for equal opportunity (equal TPR across groups)"""
        
        groups = protected_attr.unique()
        best_thresholds = {}
        best_score = float('-inf')
        
        # Try all combinations of thresholds
        from itertools import product
        
        threshold_combinations = product(candidates, repeat=len(groups))
        
        for thresholds in threshold_combinations:
            group_thresholds = dict(zip(groups, thresholds))
            
            # Calculate TPRs for each group
            tprs = {}
            overall_accuracy = 0
            total_samples = 0
            
            valid_combination = True
            
            for group, threshold in group_thresholds.items():
                group_mask = protected_attr == group
                group_y_true = y_true[group_mask]
                group_y_scores = y_scores[group_mask]
                
                if len(group_y_true) == 0:
                    continue
                
                group_predictions = (group_y_scores >= threshold).astype(int)
                
                # Calculate TPR
                positive_mask = group_y_true == 1
                if positive_mask.sum() > 0:
                    tpr = group_predictions[positive_mask].mean()
                    tprs[group] = tpr
                else:
                    valid_combination = False
                    break
                
                # Accumulate for overall accuracy
                group_accuracy = (group_predictions == group_y_true).mean()
                overall_accuracy += group_accuracy * len(group_y_true)
                total_samples += len(group_y_true)
            
            if not valid_combination or len(tprs) < 2:
                continue
            
            # Calculate fairness score (negative TPR difference)
            tpr_values = list(tprs.values())
            fairness_score = -abs(max(tpr_values) - min(tpr_values))
            
            # Combined score (fairness + accuracy)
            overall_accuracy /= total_samples
            combined_score = 0.7 * fairness_score + 0.3 * overall_accuracy
            
            if combined_score > best_score:
                best_score = combined_score
                best_thresholds = group_thresholds.copy()
                
                self.optimization_results = {
                    'tprs': tprs,
                    'overall_accuracy': overall_accuracy,
                    'fairness_score': fairness_score,
                    'combined_score': combined_score
                }
        
        self.thresholds = best_thresholds
    
    def _optimize_demographic_parity(self, y_true, y_scores, protected_attr, candidates):
        """Optimize for demographic parity (equal positive prediction rates)"""
        
        groups = protected_attr.unique()
        best_thresholds = {}
        best_score = float('-inf')
        
        from itertools import product
        threshold_combinations = product(candidates, repeat=len(groups))
        
        for thresholds in threshold_combinations:
            group_thresholds = dict(zip(groups, thresholds))
            
            # Calculate positive prediction rates
            positive_rates = {}
            overall_accuracy = 0
            total_samples = 0
            
            for group, threshold in group_thresholds.items():
                group_mask = protected_attr == group
                group_y_true = y_true[group_mask]
                group_y_scores = y_scores[group_mask]
                
                if len(group_y_true) == 0:
                    continue
                
                group_predictions = (group_y_scores >= threshold).astype(int)
                positive_rates[group] = group_predictions.mean()
                
                # Accumulate for overall accuracy
                group_accuracy = (group_predictions == group_y_true).mean()
                overall_accuracy += group_accuracy * len(group_y_true)
                total_samples += len(group_y_true)
            
            if len(positive_rates) < 2:
                continue
            
            # Calculate fairness score
            rate_values = list(positive_rates.values())
            fairness_score = -abs(max(rate_values) - min(rate_values))
            
            # Combined score
            overall_accuracy /= total_samples
            combined_score = 0.7 * fairness_score + 0.3 * overall_accuracy
            
            if combined_score > best_score:
                best_score = combined_score
                best_thresholds = group_thresholds.copy()
                
                self.optimization_results = {
                    'positive_rates': positive_rates,
                    'overall_accuracy': overall_accuracy,
                    'fairness_score': fairness_score,
                    'combined_score': combined_score
                }
        
        self.thresholds = best_thresholds
    
    def _fit_constrained_optimization(self, y_true, y_scores, protected_attr):
        """Use constrained optimization for threshold finding"""
        
        from scipy.optimize import minimize
        
        groups = list(protected_attr.unique())
        n_groups = len(groups)
        
        def objective_function(thresholds):
            """Objective: maximize overall accuracy"""
            
            total_accuracy = 0
            total_samples = 0
            
            for i, group in enumerate(groups):
                group_mask = protected_attr == group
                group_y_true = y_true[group_mask]
                group_y_scores = y_scores[group_mask]
                
                if len(group_y_true) == 0:
                    continue
                
                group_predictions = (group_y_scores >= thresholds[i]).astype(int)
                group_accuracy = (group_predictions == group_y_true).mean()
                
                total_accuracy += group_accuracy * len(group_y_true)
                total_samples += len(group_y_true)
            
            return -(total_accuracy / total_samples)  # Negative for minimization
        
        def fairness_constraint(thresholds):
            """Constraint: fairness metric violation should be small"""
            
            if self.fairness_metric == 'equal_opportunity':
                tprs = []
                
                for i, group in enumerate(groups):
                    group_mask = protected_attr == group
                    group_y_true = y_true[group_mask]
                    group_y_scores = y_scores[group_mask]
                    
                    positive_mask = group_y_true == 1
                    if positive_mask.sum() > 0:
                        group_predictions = (group_y_scores >= thresholds[i]).astype(int)
                        tpr = group_predictions[positive_mask].mean()
                        tprs.append(tpr)
                
                if len(tprs) >= 2:
                    return 0.05 - abs(max(tprs) - min(tprs))  # Constraint: violation <= 0.05
                else:
                    return 0
            
            # Add other fairness metrics as needed
            return 0
        
        # Optimization
        initial_thresholds = [0.5] * n_groups
        bounds = [(0.1, 0.9)] * n_groups
        constraints = {'type': 'ineq', 'fun': fairness_constraint}
        
        result = minimize(
            objective_function,
            initial_thresholds,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            self.thresholds = dict(zip(groups, result.x))
        else:
            # Fallback to equal thresholds
            self.thresholds = {group: 0.5 for group in groups}
    
    def predict(self, y_scores, protected_attr):
        """Make fair predictions using optimized thresholds"""
        
        if not self.thresholds:
            raise ValueError("Must call fit() first")
        
        predictions = np.zeros(len(y_scores))
        
        for group, threshold in self.thresholds.items():
            group_mask = protected_attr == group
            predictions[group_mask] = (y_scores[group_mask] >= threshold).astype(int)
        
        return predictions
    
    def get_threshold_summary(self):
        """Get summary of optimized thresholds"""
        
        summary = {
            'thresholds': self.thresholds,
            'fairness_metric': self.fairness_metric,
            'optimization_results': self.optimization_results
        }
        
        return summary
```

### 1.2 ROC-Based Fair Threshold Selection

```python
class ROCFairThresholdSelector:
    """Select thresholds based on ROC analysis for fairness"""
    
    def __init__(self, fairness_constraint='equal_opportunity', tolerance=0.05):
        self.fairness_constraint = fairness_constraint
        self.tolerance = tolerance
        self.group_roc_curves = {}
        self.selected_thresholds = {}
    
    def fit(self, y_true, y_scores, protected_attr):
        """Analyze ROC curves and select fair thresholds"""
        
        # Calculate ROC curves for each group
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            group_y_true = y_true[group_mask]
            group_y_scores = y_scores[group_mask]
            
            if len(group_y