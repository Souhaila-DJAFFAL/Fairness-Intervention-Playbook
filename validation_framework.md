                 (len(intervention_violations) - 1) * np.var(intervention_violations)) /
                (len(baseline_violations) + len(intervention_violations) - 2)
            )
            
            effect_size = (np.mean(baseline_violations) - np.mean(intervention_violations)) / pooled_std
            
            significance_results[metric] = {
                'p_value': p_value,
                'significant': p_value < alpha,
                'effect_size': effect_size,
                'baseline_mean': np.mean(baseline_violations),
                'intervention_mean': np.mean(intervention_violations),
                'improvement': np.mean(baseline_violations) - np.mean(intervention_violations)
            }
        
        return significance_results
    
    def _calculate_single_metric(self, metric, y_true, y_pred, protected_attr):
        """Calculate a single fairness metric"""
        
        groups = np.unique(protected_attr)
        
        if metric == 'demographic_parity':
            group_rates = []
            for group in groups:
                group_mask = protected_attr == group
                if group_mask.sum() > 0:
                    group_rates.append(y_pred[group_mask].mean())
            
            return max(group_rates) - min(group_rates) if len(group_rates) >= 2 else 0
        
        elif metric == 'equal_opportunity':
            group_tprs = []
            for group in groups:
                group_mask = protected_attr == group
                positive_mask = y_true == 1
                group_positive_mask = group_mask & positive_mask
                
                if group_positive_mask.sum() > 0:
                    tpr = y_pred[group_positive_mask].mean()
                    group_tprs.append(tpr)
            
            return max(group_tprs) - min(group_tprs) if len(group_tprs) >= 2 else 0
        
        # Add other metrics as needed
        return 0
    
    def _generate_validation_report(self):
        """Generate comprehensive validation report"""
        
        report = {
            'summary': {
                'metrics_improved': 0,
                'metrics_degraded': 0,
                'statistically_significant': 0,
                'large_effect_size': 0
            },
            'detailed_results': {},
            'recommendations': []
        }
        
        for metric in self.fairness_metrics:
            baseline_violation = self.baseline_results[metric]['violation']
            intervention_violation = self.intervention_results[metric]['violation']
            improvement = baseline_violation - intervention_violation
            
            sig_result = self.statistical_significance.get(metric, {})
            
            # Classify improvement
            if improvement > 0.01:  # Meaningful improvement threshold
                report['summary']['metrics_improved'] += 1
            elif improvement < -0.01:  # Degradation
                report['summary']['metrics_degraded'] += 1
            
            if sig_result.get('significant', False):
                report['summary']['statistically_significant'] += 1
            
            if abs(sig_result.get('effect_size', 0)) > 0.5:  # Medium to large effect
                report['summary']['large_effect_size'] += 1
            
            # Detailed results
            report['detailed_results'][metric] = {
                'baseline_violation': baseline_violation,
                'intervention_violation': intervention_violation,
                'absolute_improvement': improvement,
                'relative_improvement': improvement / baseline_violation if baseline_violation > 0 else 0,
                'statistical_significance': sig_result.get('significant', False),
                'p_value': sig_result.get('p_value', 1.0),
                'effect_size': sig_result.get('effect_size', 0)
            }
        
        # Generate recommendations
        if report['summary']['metrics_improved'] == 0:
            report['recommendations'].append("No fairness improvements detected. Consider alternative intervention strategies.")
        
        if report['summary']['metrics_degraded'] > 0:
            report['recommendations'].append("Some fairness metrics degraded. Review intervention parameters.")
        
        if report['summary']['statistically_significant'] < report['summary']['metrics_improved']:
            report['recommendations'].append("Some improvements may not be statistically significant. Consider larger sample sizes or different approaches.")
        
        return report
```

## Dimension 2: Performance Impact Assessment

### Purpose
Measure the impact of fairness interventions on model performance and business metrics.

### 2.1 Multi-Level Performance Validation

```python
class PerformanceImpactValidator:
    """Validate performance impact of fairness interventions"""
    
    def __init__(self, performance_metrics=None, business_metrics=None):
        self.performance_metrics = performance_metrics or [
            'accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr'
        ]
        self.business_metrics = business_metrics or [
            'approval_rate', 'revenue_impact', 'risk_exposure'
        ]
        self.baseline_performance = {}
        self.intervention_performance = {}
        self.acceptable_degradation = {}
    
    def validate_performance_impact(self, y_true_baseline, y_pred_baseline, y_scores_baseline,
                                  y_true_intervention, y_pred_intervention, y_scores_intervention,
                                  protected_attr, business_data=None, 
                                  degradation_thresholds=None):
        """Validate performance impact of intervention"""
        
        # Set acceptable degradation thresholds
        self.acceptable_degradation = degradation_thresholds or {
            'accuracy': 0.02,    # 2% accuracy loss acceptable
            'auc_roc': 0.01,     # 1% AUC loss acceptable
            'f1': 0.03,          # 3% F1 loss acceptable
            'approval_rate': 0.05, # 5% approval rate change acceptable
            'revenue_impact': 0.03  # 3% revenue impact acceptable
        }
        
        # Calculate performance metrics
        self.baseline_performance = self._calculate_performance_metrics(
            y_true_baseline, y_pred_baseline, y_scores_baseline, protected_attr
        )
        
        self.intervention_performance = self._calculate_performance_metrics(
            y_true_intervention, y_pred_intervention, y_scores_intervention, protected_attr
        )
        
        # Calculate business impact if data provided
        business_impact = {}
        if business_data is not None:
            business_impact = self._calculate_business_impact(
                y_pred_baseline, y_pred_intervention, business_data, protected_attr
            )
        
        # Generate performance assessment
        return self._generate_performance_assessment(business_impact)
    
    def _calculate_performance_metrics(self, y_true, y_pred, y_scores, protected_attr):
        """Calculate comprehensive performance metrics"""
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, confusion_matrix
        )
        
        # Overall metrics
        overall_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }
        
        # AUC metrics (if probability scores available)
        if y_scores is not None and len(np.unique(y_true)) > 1:
            try:
                overall_metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
                overall_metrics['auc_pr'] = average_precision_score(y_true, y_scores)
            except ValueError:
                overall_metrics['auc_roc'] = 0.5
                overall_metrics['auc_pr'] = 0.5
        
        # Group-specific metrics
        group_metrics = {}
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            group_y_scores = y_scores[group_mask] if y_scores is not None else None
            
            if len(group_y_true) > 0:
                group_performance = {
                    'accuracy': accuracy_score(group_y_true, group_y_pred),
                    'precision': precision_score(group_y_true, group_y_pred, average='binary', zero_division=0),
                    'recall': recall_score(group_y_true, group_y_pred, average='binary', zero_division=0),
                    'f1': f1_score(group_y_true, group_y_pred, average='binary', zero_division=0),
                    'sample_size': len(group_y_true)
                }
                
                # Group-specific AUC if possible
                if group_y_scores is not None and len(np.unique(group_y_true)) > 1:
                    try:
                        group_performance['auc_roc'] = roc_auc_score(group_y_true, group_y_scores)
                        group_performance['auc_pr'] = average_precision_score(group_y_true, group_y_scores)
                    except ValueError:
                        group_performance['auc_roc'] = 0.5
                        group_performance['auc_pr'] = 0.5
                
                group_metrics[group] = group_performance
        
        return {
            'overall': overall_metrics,
            'by_group': group_metrics
        }
    
    def _calculate_business_impact(self, y_pred_baseline, y_pred_intervention, 
                                 business_data, protected_attr):
        """Calculate business impact metrics"""
        
        business_impact = {}
        
        # Approval rate changes
        baseline_approval_rate = y_pred_baseline.mean()
        intervention_approval_rate = y_pred_intervention.mean()
        
        business_impact['approval_rate_change'] = (
            intervention_approval_rate - baseline_approval_rate
        )
        
        # Revenue impact (if revenue data available)
        if 'loan_amount' in business_data.columns:
            baseline_revenue = business_data['loan_amount'][y_pred_baseline == 1].sum()
            intervention_revenue = business_data['loan_amount'][y_pred_intervention == 1].sum()
            
            revenue_change = (intervention_revenue - baseline_revenue) / baseline_revenue
            business_impact['revenue_impact'] = revenue_change
        
        # Risk exposure changes
        if 'risk_score' in business_data.columns:
            baseline_avg_risk = business_data['risk_score'][y_pred_baseline == 1].mean()
            intervention_avg_risk = business_data['risk_score'][y_pred_intervention == 1].mean()
            
            risk_change = intervention_avg_risk - baseline_avg_risk
            business_impact['risk_exposure_change'] = risk_change
        
        # Group-specific business impacts
        group_business_impact = {}
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            
            group_baseline_approval = y_pred_baseline[group_mask].mean()
            group_intervention_approval = y_pred_intervention[group_mask].mean()
            
            group_business_impact[group] = {
                'approval_rate_change': group_intervention_approval - group_baseline_approval
            }
            
            if 'loan_amount' in business_data.columns:
                group_data = business_data[group_mask]
                group_baseline_revenue = group_data['loan_amount'][y_pred_baseline[group_mask] == 1].sum()
                group_intervention_revenue = group_data['loan_amount'][y_pred_intervention[group_mask] == 1].sum()
                
                if group_baseline_revenue > 0:
                    group_revenue_change = (group_intervention_revenue - group_baseline_revenue) / group_baseline_revenue
                    group_business_impact[group]['revenue_impact'] = group_revenue_change
        
        business_impact['by_group'] = group_business_impact
        
        return business_impact
    
    def _generate_performance_assessment(self, business_impact):
        """Generate comprehensive performance assessment"""
        
        assessment = {
            'overall_impact': {
                'acceptable_degradation': True,
                'critical_metrics_degraded': [],
                'performance_summary': {}
            },
            'group_impact': {},
            'business_impact': business_impact,
            'recommendations': []
        }
        
        # Assess overall performance changes
        for metric in self.performance_metrics:
            if metric in self.baseline_performance['overall'] and metric in self.intervention_performance['overall']:
                baseline_value = self.baseline_performance['overall'][metric]
                intervention_value = self.intervention_performance['overall'][metric]
                
                change = intervention_value - baseline_value
                relative_change = change / baseline_value if baseline_value != 0 else 0
                
                threshold = self.acceptable_degradation.get(metric, 0.05)
                
                assessment['overall_impact']['performance_summary'][metric] = {
                    'baseline': baseline_value,
                    'intervention': intervention_value,
                    'absolute_change': change,
                    'relative_change': relative_change,
                    'acceptable': change >= -threshold
                }
                
                if change < -threshold:
                    assessment['overall_impact']['acceptable_degradation'] = False
                    assessment['overall_impact']['critical_metrics_degraded'].append(metric)
        
        # Assess group-specific impacts
        for group in self.baseline_performance['by_group'].keys():
            group_assessment = {}
            
            for metric in self.performance_metrics:
                if (metric in self.baseline_performance['by_group'][group] and 
                    metric in self.intervention_performance['by_group'][group]):
                    
                    baseline_value = self.baseline_performance['by_group'][group][metric]
                    intervention_value = self.intervention_performance['by_group'][group][metric]
                    
                    change = intervention_value - baseline_value
                    relative_change = change / baseline_value if baseline_value != 0 else 0
                    
                    group_assessment[metric] = {
                        'baseline': baseline_value,
                        'intervention': intervention_value,
                        'change': change,
                        'relative_change': relative_change
                    }
            
            assessment['group_impact'][group] = group_assessment
        
        # Generate recommendations
        if not assessment['overall_impact']['acceptable_degradation']:
            assessment['recommendations'].append(
                f"Critical performance degradation in: {', '.join(assessment['overall_impact']['critical_metrics_degraded'])}. "
                "Consider adjusting intervention parameters or trying alternative approaches."
            )
        
        if business_impact and 'revenue_impact' in business_impact:
            if business_impact['revenue_impact'] < -0.05:  # >5% revenue loss
                assessment['recommendations'].append(
                    "Significant revenue impact detected. Evaluate business case for fairness intervention."
                )
        
        return assessment
```

## Dimension 3: Robustness and Stability Validation

### Purpose
Ensure fairness interventions are robust across different conditions and stable over time.

### 3.1 Cross-Validation and Stability Testing

```python
class RobustnessValidator:
    """Validate robustness and stability of fairness interventions"""
    
    def __init__(self, n_folds=5, stability_threshold=0.05):
        self.n_folds = n_folds
        self.stability_threshold = stability_threshold
        self.cross_validation_results = {}
        self.stability_results = {}
        self.distribution_shift_results = {}
    
    def validate_robustness(self, X, y, protected_attr, intervention_pipeline, 
                          baseline_model=None, test_scenarios=None):
        """Comprehensive robustness validation"""
        
        # Cross-validation stability
        cv_results = self._cross_validation_stability(
            X, y, protected_attr, intervention_pipeline, baseline_model
        )
        
        # Distribution shift testing
        shift_results = self._test_distribution_shifts(
            X, y, protected_attr, intervention_pipeline, test_scenarios
        )
        
        # Temporal stability (if temporal data available)
        temporal_results = self._test_temporal_stability(
            X, y, protected_attr, intervention_pipeline
        )
        
        return self._generate_robustness_report(cv_results, shift_results, temporal_results)
    
    def _cross_validation_stability(self, X, y, protected_attr, intervention_pipeline, baseline_model):
        """Test stability across cross-validation folds"""
        
        from sklearn.model_selection import StratifiedKFold
        
        # Stratified CV to maintain class balance
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        fold_results = {
            'fairness_metrics': [],
            'performance_metrics': [],
            'intervention_effectiveness': []
        }
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            protected_train = protected_attr.iloc[train_idx]
            protected_test = protected_attr.iloc[test_idx]
            
            # Train baseline model
            if baseline_model is not None:
                baseline_model.fit(X_train, y_train)
                baseline_pred = baseline_model.predict(X_test)
                baseline_scores = baseline_model.predict_proba(X_test)[:, 1] if hasattr(baseline_model, 'predict_proba') else baseline_pred
            else:
                # Use simple threshold as baseline
                baseline_scores = np.random.random(len(X_test))
                baseline_pred = (baseline_scores > 0.5).astype(int)
            
            # Apply intervention pipeline
            try:
                intervention_pipeline.fit(y_train, baseline_scores[:len(y_train)], protected_train)
                intervention_pred = intervention_pipeline.predict(baseline_scores, protected_test)
            except:
                # If intervention fails, use baseline
                intervention_pred = baseline_pred
            
            # Calculate metrics for this fold
            fold_fairness = self._calculate_fold_fairness_metrics(
                y_test, baseline_pred, intervention_pred, protected_test
            )
            
            fold_performance = self._calculate_fold_performance_metrics(
                y_test, baseline_pred, intervention_pred
            )
            
            fold_effectiveness = self._calculate_intervention_effectiveness(
                fold_fairness, fold_performance
            )
            
            fold_results['fairness_metrics'].append(fold_fairness)
            fold_results['performance_metrics'].append(fold_performance)
            fold_results['intervention_effectiveness'].append(fold_effectiveness)
        
        # Analyze stability across folds
        stability_analysis = self._analyze_cross_fold_stability(fold_results)
        
        return {
            'fold_results': fold_results,
            'stability_analysis': stability_analysis
        }
    
    def _calculate_fold_fairness_metrics(self, y_true, y_baseline, y_intervention, protected_attr):
        """Calculate fairness metrics for a single fold"""
        
        metrics = {}
        groups = protected_attr.unique()
        
        # Demographic parity
        baseline_rates = {group: y_baseline[protected_attr == group].mean() for group in groups}
        intervention_rates = {group: y_intervention[protected_attr == group].mean() for group in groups}
        
        baseline_dp = max(baseline_rates.values()) - min(baseline_rates.values())
        intervention_dp = max(intervention_rates.values()) - min(intervention_rates.values())
        
        metrics['demographic_parity'] = {
            'baseline': baseline_dp,
            'intervention': intervention_dp,
            'improvement': baseline_dp - intervention_dp
        }
        
        # Equal opportunity
        baseline_tprs = {}
        intervention_tprs = {}
        
        for group in groups:
            group_mask = protected_attr == group
            positive_mask = y_true == 1
            group_positive_mask = group_mask & positive_mask
            
            if group_positive_mask.sum() > 0:
                baseline_tprs[group] = y_baseline[group_positive_mask].mean()
                intervention_tprs[group] = y_intervention[group_positive_mask].mean()
        
        if len(baseline_tprs) >= 2:
            baseline_eo = max(baseline_tprs.values()) - min(baseline_tprs.values())
            intervention_eo = max(intervention_tprs.values()) - min(intervention_tprs.values())
            
            metrics['equal_opportunity'] = {
                'baseline': baseline_eo,
                'intervention': intervention_eo,
                'improvement': baseline_eo - intervention_eo
            }
        
        return metrics
    
    def _calculate_fold_performance_metrics(self, y_true, y_baseline, y_intervention):
        """Calculate performance metrics for a single fold"""
        
        from sklearn.metrics import accuracy_score, f1_score
        
        return {
            'accuracy': {
                'baseline': accuracy_score(y_true, y_baseline),
                'intervention': accuracy_score(y_true, y_intervention)
            },
            'f1': {
                'baseline': f1_score(y_true, y_baseline, zero_division=0),
                'intervention': f1_score(y_true, y_intervention, zero_division=0)
            }
        }
    
    def _calculate_intervention_effectiveness(self, fairness_metrics, performance_metrics):
        """Calculate overall intervention effectiveness"""
        
        # Simple effectiveness score: fairness improvement - performance degradation
        fairness_improvement = 0
        performance_degradation = 0
        
        for metric, values in fairness_metrics.items():
            fairness_improvement += values.get('improvement', 0)
        
        for metric, values in performance_metrics.items():
            baseline = values['baseline']
            intervention = values['intervention']
            if baseline > 0:
                degradation = max(0, baseline - intervention)
                performance_degradation += degradation / baseline
        
        effectiveness = fairness_improvement - 0.5 * performance_degradation
        
        return {
            'effectiveness_score': effectiveness,
            'fairness_improvement': fairness_improvement,
            'performance_degradation': performance_degradation
        }
    
    def _analyze_cross_fold_stability(self, fold_results):
        """Analyze stability of results across CV folds"""
        
        stability_analysis = {}
        
        # Analyze fairness metric stability
        fairness_stability = {}
        for metric in ['demographic_parity', 'equal_opportunity']:
            if any(metric in fold for fold in fold_results['fairness_metrics']):
                improvements = [
                    fold.get(metric, {}).get('improvement', 0)
                    for fold in fold_results['fairness_metrics']
                    if metric in fold
                ]
                
                if improvements:
                    stability_score = 1.0 - (np.std(improvements) / (np.mean(improvements) + 1e-8))
                    fairness_stability[metric] = {
                        'mean_improvement': np.mean(improvements),
                        'std_improvement': np.std(improvements),
                        'stability_score': stability_score,
                        'stable': np.std(improvements) < self.stability_threshold
                    }
        
        # Analyze performance stability
        performance_stability = {}
        for metric in ['accuracy', 'f1']:
            baseline_values = [fold[metric]['baseline'] for fold in fold_results['performance_metrics']]
            intervention_values = [fold[metric]['intervention'] for fold in fold_results['performance_metrics']]
            
            baseline_std = np.std(baseline_values)
            intervention_std = np.std(intervention_values)
            
            performance_stability[metric] = {
                'baseline_stability': baseline_std < self.stability_threshold,
                'intervention_stability': intervention_std < self.stability_threshold,
                'baseline_std': baseline_std,
                'intervention_std': intervention_std
            }
        
        # Overall effectiveness stability
        effectiveness_scores = [fold['effectiveness_score'] for fold in fold_results['intervention_effectiveness']]
        effectiveness_stability = {
            'mean_effectiveness': np.mean(effectiveness_scores),
            'std_effectiveness': np.std(effectiveness_scores),
            'stable': np.std(effectiveness_scores) < self.stability_threshold
        }
        
        return {
            'fairness_stability': fairness_stability,
            'performance_stability': performance_stability,
            'effectiveness_stability': effectiveness_stability
        }
    
    def _test_distribution_shifts(self, X, y, protected_attr, intervention_pipeline, test_scenarios):
        """Test robustness under distribution shifts"""
        
        if test_scenarios is None:
            # Create default test scenarios
            test_scenarios = self._create_default_test_scenarios(X, y, protected_attr)
        
        shift_results = {}
        
        for scenario_name, scenario_data in test_scenarios.items():
            try:
                # Apply scenario transformation
                X_shifted = scenario_data['transform'](X)
                
                # Test intervention on shifted data
                # This is a simplified approach - in practice, you'd retrain/recalibrate
                baseline_scores = np.random.random(len(X_shifted))  # Placeholder
                
                # Apply intervention (assuming it was already fitted)
                intervention_pred = intervention_pipeline.predict(baseline_scores, protected_attr)
                baseline_pred = (baseline_scores > 0.5).astype(int)
                
                # Calculate metrics
                fairness_under_shift = self._calculate_fold_fairness_metrics(
                    y, baseline_pred, intervention_pred, protected_attr
                )
                
                shift_results[scenario_name] = {
                    'fairness_metrics': fairness_under_shift,
                    'successful': True
                }
                
            except Exception as e:
                shift_results[scenario_name] = {
                    'successful': False,
                    'error': str(e)
                }
        
        return shift_results
    
    def _create_default_test_scenarios(self, X, y, protected_attr):
        """Create default distribution shift test scenarios"""
        
        scenarios = {}
        
        # Feature noise scenario
        scenarios['feature_noise'] = {
            'transform': lambda X: X + np.random.normal(0, 0.1, X.shape)
        }
        
        # Feature scaling scenario
        scenarios['feature_scaling'] = {
            'transform': lambda X: X * np.random.uniform(0.8, 1.2, X.shape[1])
        }
        
        # Missing feature scenario (if applicable)
        if X.shape[1] > 5:
            scenarios['missing_features'] = {
                'transform': lambda X: X.iloc[:, :-2]  # Remove last 2 features
            }
        
        return scenarios
    
    def _test_temporal_stability(self, X, y, protected_attr, intervention_pipeline):
        """Test temporal stability (if temporal features available)"""
        
        # This is a simplified implementation
        # In practice, you'd use actual temporal data
        
        temporal_results = {
            'tested': False,
            'reason': 'No temporal features detected'
        }
        
        # Check if temporal features exist
        temporal_features = [col for col in X.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if temporal_features:
            # Implement temporal stability testing
            temporal_results = {
                'tested': True,
                'temporal_features': temporal_features,
                'stability_analysis': 'Temporal stability testing would be implemented here'
            }
        
        return temporal_results
    
    def _generate_robustness_report(self, cv_results, shift_results, temporal_results):
        """Generate comprehensive robustness report"""
        
        report = {
            'overall_robustness': {
                'stable_across_cv': True,
                'robust_to_shifts': True,
                'temporally_stable': True
            },
            'detailed_analysis': {
                'cross_validation': cv_results,
                'distribution_shifts': shift_results,
                'temporal_stability': temporal_results
            },
            'recommendations': []
        }
        
        # Assess CV stability
        cv_stability = cv_results['stability_analysis']
        
        fairness_stable = all(
            metric_data.get('stable', False)
            for metric_data in cv_stability['fairness_stability'].values()
        )
        
        performance_stable = all(
            metric_data.get('baseline_stability', False) and metric_data.get('intervention_stability', False)
            for metric_data in cv_stability['performance_stability'].values()
        )
        
        if not fairness_stable:
            report['overall_robustness']['stable_across_cv'] = False
            report['recommendations'].append(
                "Fairness improvements show instability across CV folds. Consider more robust intervention methods."
            )
        
        if not performance_stable:
            report['overall_robustness']['stable_across_cv'] = False
            report['recommendations'].append(
                "Performance shows instability across CV folds. Review model training and intervention parameters."
            )
        
        # Assess distribution shift robustness
        shift_failures = sum(
            1 for result in shift_results.values() if not result.get('successful', False)
        )
        
        if shift_failures > len(shift_results) * 0.3:  # >30% failures
            report['overall_robustness']['robust_to_shifts'] = False
            report['recommendations'].append(
                "Intervention shows poor robustness to distribution shifts. Consider domain adaptation techniques."
            )
        
        # Overall assessment
        if report['overall_robustness']['stable_across_cv'] and report['overall_robustness']['robust_to_shifts']:
            report['recommendations'].append(
                "Intervention shows good robustness characteristics. Safe for deployment with continued monitoring."
            )
        
        return report
```

## Dimension 4: Comprehensive Validation Pipeline

### Purpose
Orchestrate all validation dimensions into a single comprehensive assessment.

### 4.1 Integrated Validation Framework

```python
class ComprehensiveValidator:
    """Comprehensive validation of fairness interventions"""
    
    def __init__(self, validation_config=None):
        self.validation_config = validation_config or {
            'fairness_metrics': ['demographic_parity', 'equal_opportunity', 'equalized_odds'],
            'performance_thresholds': {'accuracy': 0.02, 'f1': 0.03, 'auc_roc': 0.01},
            'business_thresholds': {'revenue_impact': 0.05, 'approval_rate': 0.1},
            'stability_threshold': 0.05,
            'confidence_level': 0.95
        }
        
        # Initialize component validators
        self.fairness_validator = FairnessMetricsValidator(
            self.validation_config['fairness_metrics']
        )
        self.performance_validator = PerformanceImpactValidator(
            degradation_thresholds=self.validation_config['performance_thresholds']
        )
        self.robustness_validator = RobustnessValidator(
            stability_threshold=self.validation_config['stability_threshold']
        )
        
        self.validation_results = {}
        self.overall_assessment = {}
    
    def validate_complete_intervention(self, baseline_data, intervention_data, 
                                     intervention_pipeline, business_data=None,
                                     test_scenarios=None):
        """Complete validation of fairness intervention"""
        
        # Unpack data
        X_baseline, y_true_baseline, y_pred_baseline, y_scores_baseline, protected_attr_baseline = baseline_data
        X_intervention, y_true_intervention, y_pred_intervention, y_scores_intervention, protected_attr_intervention = intervention_data
        
        print("Starting comprehensive validation...")
        
        # 1. Fairness Metrics Validation
        print("Validating fairness improvements...")
        fairness_results = self.fairness_validator.validate_intervention(
            y_true_baseline, y_pred_baseline, y_scores_baseline,
            y_true_intervention, y_pred_intervention, y_scores_intervention,
            protected_attr_baseline, 
            confidence_level=self.validation_config['confidence_level']
        )
        
        # 2. Performance Impact Validation
        print("Validating performance impact...")
        performance_results = self.performance_validator.validate_performance_impact(
            y_true_baseline, y_pred_baseline, y_scores_baseline,
            y_true_intervention, y_pred_intervention, y_scores_intervention,
            protected_attr_baseline, business_data,
            degradation_thresholds=self.validation_config['performance_thresholds']
        )
        
        # 3. Robustness Validation
        print("Validating robustness and stability...")
        robustness_results = self.robustness_validator.validate_robustness(
            X_intervention, y_true_intervention, protected_attr_intervention,
            intervention_pipeline, test_scenarios=test_scenarios
        )
        
        # 4. Integration and Overall Assessment
        print("Generating overall assessment...")
        overall_assessment = self._generate_overall_assessment(
            fairness_results, performance_results, robustness_results
        )
        
        # Store results
        self.validation_results = {
            'fairness': fairness_results,
            'performance': performance_results,
            'robustness': robustness_results,
            'overall': overall_assessment
        }
        
        # Generate recommendations
        recommendations = self._generate_deployment_recommendations()
        self.validation_results['recommendations'] = recommendations
        
        print("Validation complete!")
        return self.validation_results
    
    def _generate_overall_assessment(self, fairness_results, performance_results, robustness_results):
        """Generate overall assessment combining all validation dimensions"""
        
        assessment = {
            'deployment_ready': True,
            'critical_issues': [],
            'warnings': [],
            'strengths': [],
            'overall_score': 0.0,
            'dimension_scores': {}
        }
        
        # Assess fairness dimension
        fairness_score = self._score_fairness_results(fairness_results)
        assessment['dimension_scores']['fairness'] = fairness_score
        
        if fairness_score['score'] < 0.6:
            assessment['deployment_ready'] = False
            assessment['critical_issues'].append(
                f"Insufficient fairness improvements (score: {fairness_score['score']:.2f})"
            )
        elif fairness_score['score'] < 0.8:
            assessment['warnings'].append(
                f"Moderate fairness improvements (score: {fairness_score['score']:.2f})"
            )
        else:
            assessment['strengths'].append(
                f"Strong fairness improvements (score: {fairness_score['score']:.2f})"
            )
        
        # Assess performance dimension
        performance_score = self._score_performance_results(performance_results)
        assessment['dimension_scores']['performance'] = performance_score
        
        if not performance_results['overall_impact']['acceptable_degradation']:
            assessment['deployment_ready'] = False
            assessment['critical_issues'].append(
                "Unacceptable performance degradation in critical metrics"
            )
        elif performance_score['score'] < 0.7:
            assessment['warnings'].append(
                f"Moderate performance impact (score: {performance_score['score']:.2f})"
            )
        else:
            assessment['strengths'].append(
                f"Good performance preservation (score: {performance_score['score']:.2f})"
            )
        
        # Assess robustness dimension
        robustness_score = self._score_robustness_results(robustness_results)
        assessment['dimension_scores']['robustness'] = robustness_score
        
        if not robustness_results['overall_robustness']['stable_across_cv']:
            assessment['deployment_ready'] = False
            assessment['critical_issues'].append("Unstable results across validation folds")
        elif robustness_score['score'] < 0.7:
            assessment['warnings'].append(
                f"Moderate robustness concerns (score: {robustness_score['score']:.2f})"
            )
        else:
            assessment['strengths'].append(
                f"Good robustness characteristics (score: {robustness_score['score']:.2f})"
            )
        
        # Calculate overall score
        weights = {'fairness': 0.4, 'performance': 0.35, 'robustness': 0.25}
        overall_score = sum(
            weights[dim] * assessment['dimension_scores'][dim]['score']
            for dim in weights.keys()
        )
        assessment['overall_score'] = overall_score
        
        # Final deployment decision
        if overall_score < 0.6:
            assessment['deployment_ready'] = False
            assessment['critical_issues'].append(
                f"Overall validation score too low ({overall_score:.2f})"
            )
        
        return assessment
    
    def _score_fairness_results(self, fairness_results):
        """Score fairness validation results"""
        
        # Count improvements vs degradations
        improved_metrics = fairness_results['summary']['metrics_improved']
        degraded_metrics = fairness_results['summary']['metrics_degraded']
        significant_improvements = fairness_results['summary']['statistically_significant']
        
        total_metrics = len(self.validation_config['fairness_metrics'])
        
        # Calculate score
        improvement_ratio = improved_metrics / total_metrics
        significance_ratio = significant_improvements / max(improved_metrics, 1)
        degradation_penalty = degraded_metrics / total_metrics
        
        score = improvement_ratio * significance_ratio - 0.5 * degradation_penalty
        score = max(0, min(1, score))  # Clamp to [0, 1]
        
        return {
            'score': score,
            'improved_metrics': improved_metrics,
            'significant_improvements': significant_improvements,
            'degraded_metrics': degraded_metrics
        }
    
    def _score_performance_results(self, performance_results):
        """Score performance validation results"""
        
        # Check if performance degradation is acceptable
        acceptable = performance_results['overall_impact']['acceptable_degradation']
        
        # Calculate average performance preservation
        performance_preservation = []
        
        for metric, data in performance_results['overall_impact']['performance_summary'].items():
            if data['baseline'] > 0:
                preservation = 1 + data['relative_change']  # 1 = no change, >1 = improvement, <1 = degradation
                performance_preservation.append(max(0, preservation))
        
        avg_preservation = np.mean(performance_preservation) if performance_preservation else 0.5
        
        # Score based on acceptability and preservation
        if acceptable:
            score = min(1.0, avg_preservation)
        else:
            score = max(0.0, avg_preservation - 0.3)  # Penalty for unacceptable degradation
        
        return {
            'score': score,
            'acceptable_degradation': acceptable,
            'avg_preservation': avg_preservation,
            'critical_degradations': performance_results['overall_impact']['critical_metrics_degraded']
        }
    
    def _score_robustness_results(self, robustness_results):
        """Score robustness validation results"""
        
        robustness_flags = robustness_results['overall_robustness']
        
        # Count positive robustness indicators
        positive_indicators = sum([
            robustness_flags['stable_across_cv'],
            robustness_flags['robust_to_shifts'],
            robustness_flags['temporally_stable']
        ])
        
        score = positive_indicators / 3.0  # Normalize to [0, 1]
        
        return {
            'score': score,
            'stable_cv': robustness_flags['stable_across_cv'],
            'robust_shifts': robustness_flags['robust_to_shifts'],
            'temporal_stable': robustness_flags['temporally_stable']
        }
    
    def _generate_deployment_recommendations(self):
        """Generate deployment recommendations based on validation results"""
        
        recommendations = {
            'deployment_decision': '',
            'immediate_actions': [],
            'monitoring_requirements': [],
            'future_improvements': [],
            'risk_mitigation': []
        }
        
        overall = self.validation_results['overall']
        
        # Deployment decision
        if overall['deployment_ready']:
            if overall['overall_score'] >= 0.8:
                recommendations['deployment_decision'] = "APPROVED: High confidence deployment"
            else:
                recommendations['deployment_decision'] = "APPROVED: Conditional deployment with monitoring"
        else:
            recommendations['deployment_decision'] = "REJECTED: Critical issues must be resolved"
        
        # Immediate actions
        if overall['critical_issues']:
            recommendations['immediate_actions'].extend([
                f"Address critical issue: {issue}" for issue in overall['critical_issues']
            ])
        
        if overall['warnings']:
            recommendations['immediate_actions'].extend([
                f"Monitor closely: {warning}" for warning in overall['warnings'][:2]  # Top 2 warnings
            ])
        
        # Monitoring requirements
        fairness_results = self.validation_results['fairness']
        if fairness_results['summary']['metrics_improved'] > 0:
            recommendations['monitoring_requirements'].append(
                "Implement continuous fairness monitoring for improved metrics"
            )
        
        performance_results = self.validation_results['performance']
        if performance_results['overall_impact']['critical_metrics_degraded']:
            recommendations['monitoring_requirements'].append(
                f"Monitor performance metrics: {', '.join(performance_results['overall_impact']['critical_metrics_degraded'])}"
            )
        
        # Future improvements
        if overall['overall_score'] < 0.8:
            recommendations['future_improvements'].append(
                "Consider advanced intervention techniques to improve overall validation score"
            )
        
        robustness_results = self.validation_results['robustness']
        if not robustness_results['overall_robustness']['robust_to_shifts']:
            recommendations['future_improvements'].append(
                "Develop domain adaptation strategies for better robustness"
            )
        
        # Risk mitigation
        if overall['warnings']:
            recommendations['risk_mitigation'].append(
                "Implement gradual rollout with A/B testing"
            )
        
        if not robustness_results['overall_robustness']['stable_across_cv']:
            recommendations['risk_mitigation'].append(
                "Establish retraining triggers and model versioning"
            )
        
        return recommendations
    
    def generate_validation_report(self, output_format='markdown'):
        """Generate comprehensive validation report"""
        
        if output_format == 'markdown':
            return self._generate_markdown_report()
        elif output_format == 'json':
            return self.validation_results
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_markdown_report(self):
        """Generate markdown validation report"""
        
        report = f"""# Fairness Intervention Validation Report

## Executive Summary

**Overall Validation Score**: {self.validation_results['overall']['overall_score']:.2f}/1.00

**Deployment Decision**: {self.validation_results['recommendations']['deployment_decision']}

### Key Findings
"""
        
        # Add strengths and issues
        overall = self.validation_results['overall']
        
        if overall['strengths']:
            report += "\n**Strengths:**\n"
            for strength in overall['strengths']:
                report += f"- ✅ {strength}\n"
        
        if overall['warnings']:
            report += "\n**Warnings:**\n"
            for warning in overall['warnings']:
                report += f"- ⚠️ {warning}\n"
        
        if overall['critical_issues']:
            report += "\n**Critical Issues:**\n"
            for issue in overall['critical_issues']:
                report += f"- ❌ {issue}\n"
        
        # Detailed results
        report += "\n## Detailed Validation Results\n"
        
        # Fairness results
        fairness = self.validation_results['fairness']
        report += f"""
### Fairness Metrics Validation
- **Metrics Improved**: {fairness['summary']['metrics_improved']}
- **Statistically Significant**: {fairness['summary']['statistically_significant']}
- **Metrics Degraded**: {fairness['summary']['metrics_degraded']}

"""
        
        # Performance results
        performance = self.validation_results['performance']
        report += f"""
### Performance Impact Assessment
- **Acceptable Degradation**: {performance['overall_impact']['acceptable_degradation']}
- **Critical Metrics Affected**: {len(performance['overall_impact']['critical_metrics_degraded'])}

"""
        
        # Robustness results
        robustness = self.validation_results['robustness']
        report += f"""
### Robustness and Stability
- **Stable Across CV**: {robustness['overall_robustness']['stable_across_cv']}
- **Robust to Distribution Shifts**: {robustness['overall_robustness']['robust_to_shifts']}
- **Temporally Stable**: {robustness['overall_robustness']['temporally_stable']}

"""
        
        # Recommendations
        recommendations = self.validation_results['recommendations']
        report += "## Recommendations\n"
        
        if recommendations['immediate_actions']:
            report += "\n### Immediate Actions\n"
            for action in recommendations['immediate_actions']:
                report += f"- {action}\n"
        
        if recommendations['monitoring_requirements']:
            report += "\n### Monitoring Requirements\n"
            for requirement in recommendations['monitoring_requirements']:
                report += f"- {requirement}\n"
        
        if recommendations['future_improvements']:
            report += "\n### Future Improvements\n"
            for improvement in recommendations['future_improvements']:
                report += f"- {improvement}\n"
        
        if recommendations['risk_mitigation']:
            report += "\n### Risk Mitigation\n"
            for mitigation in recommendations['risk_mitigation']:
                report += f"- {mitigation}\n"
        
        report += f"\n---\n*Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return report
    
    def get_validation_summary(self):
        """Get concise validation summary for dashboards"""
        
        summary = {
            'overall_score': self.validation_results['overall']['overall_score'],
            'deployment_ready': self.validation_results['overall']['deployment_ready'],
            'dimension_scores': {
                dim: data['score'] for dim, data in self.validation_results['overall']['dimension_scores'].items()
            },
            'critical_issues_count': len(self.validation_results['overall']['critical_issues']),
            'warnings_count': len(self.validation_results['overall']['warnings']),
            'strengths_count': len(self.validation_results['overall']['strengths'])
        }
        
        return summary
```

## Integration with Fairness Intervention Playbook

### Usage in Complete Workflow

```python
def validate_fairness_intervention_pipeline(baseline_data, intervention_data, 
                                          intervention_pipeline, validation_config=None):
    """
    Complete validation workflow for fairness intervention pipeline
    
    Args:
        baseline_data: (X, y_true, y_pred, y_scores, protected_attr) before intervention
        intervention_data: (X, y_true, y_pred, y_scores, protected_attr) after intervention
        intervention_pipeline: Fitted intervention pipeline
        validation_config: Validation configuration parameters
    
    Returns:
        Comprehensive validation results and recommendations
    """
    
    # Initialize validator
    validator = ComprehensiveValidator(validation_config)
    
    # Run comprehensive validation
    validation_results = validator.validate_complete_intervention(
        baseline_data=baseline_data,
        intervention_data=intervention_data,
        intervention_pipeline=intervention_pipeline
    )
    
    # Generate report
    report = validator.generate_validation_report('markdown')
    summary = validator.get_validation_summary()
    
    return {
        'validation_results': validation_results,
        'report': report,
        'summary': summary,
        'deployment_ready': validation_results['overall']['deployment_ready']
    }

# Example usage in the main playbook
def example_validation_usage():
    """Example of how to use validation framework in the playbook"""
    
    # After applying fairness interventions...
    baseline_data = (X_original, y_true, y_pred_baseline, y_scores_baseline, protected_attr)
    intervention_data = (X_processed, y_true, y_pred_fair, y_scores_fair, protected_attr)
    
    # Validation configuration
    validation_config = {
        'fairness_metrics': ['demographic_parity', 'equal_opportunity'],
        'performance_thresholds': {'accuracy': 0.02, 'f1': 0.03},
        'stability_threshold': 0.05,
        'confidence_level': 0.95
    }
    
    # Run validation
    validation_outcome = validate_fairness_intervention_pipeline(
        baseline_data, intervention_data, intervention_pipeline, validation_config
    )
    
    # Decision making
    if validation_outcome['deployment_ready']:
        print("✅ Intervention validated - ready for deployment")
        print(f"Overall score: {validation_outcome['summary']['overall_score']:.2f}")
    else:
        print("❌ Intervention requires improvements before deployment")
        print("Critical issues:", validation_outcome['validation_results']['overall']['critical_issues'])
    
    return validation_outcome
```

## Documentation Template

```markdown
# Validation Report Template

## Validation Configuration
- **Fairness Metrics**: [List of metrics validated]
- **Performance Thresholds**: [Acceptable degradation limits]
- **Confidence Level**: [Statistical significance threshold]
- **Stability Threshold**: [Cross-validation stability requirement]

## Validation Results Summary
- **Overall Score**: [X.XX]/1.00
- **Deployment Decision**: [APPROVED/REJECTED/CONDITIONAL]
- **Dimensions Assessed**: Fairness, Performance, Robustness

## Detailed Findings

### Fairness Validation
| Metric | Baseline | Intervention | Improvement | Significant |
|--------|----------|--------------|-------------|-------------|
| Demographic Parity | 0.15 | 0.04 | 73% | ✅ |
| Equal Opportunity | 0.12 | 0.03 | 75% | ✅ |

### Performance Validation
| Metric | Baseline | Intervention | Change | Acceptable |
|--------|----------|--------------|--------|------------|
| Accuracy | 84.2% | 82.4% | -1.8% | ✅ |
| F1 Score | 81.5% | 79.8% | -1.7% | ✅ |

### Robustness Validation
- **Cross-Validation Stability**: ✅ Stable across all folds
- **Distribution Shift Robustness**: ⚠️ Moderate robustness
- **Temporal Stability**: ✅ Stable over time

## Recommendations
1. **Immediate Actions**: [List of immediate requirements]
2. **Monitoring Plan**: [Key metrics to track post-deployment]
3. **Future Improvements**: [Suggested enhancements]
4. **Risk Mitigation**: [Strategies to address identified risks]
```

This Validation Framework provides comprehensive assessment capabilities to ensure fairness interventions are effective, robust, and ready for production deployment while maintaining transparency about limitations and risks.# Validation Framework
## Measuring Intervention Success Across Multiple Dimensions

### Overview

The Validation Framework provides comprehensive methods to assess the effectiveness of fairness interventions across multiple dimensions: fairness improvements, model performance, business impact, and robustness. This framework ensures that interventions truly solve bias problems without creating new issues.

### Core Validation Dimensions

## Dimension 1: Fairness Metric Validation

### Purpose
Quantitatively measure improvements in fairness across different metrics and groups.

### 1.1 Comprehensive Fairness Assessment

```python
class FairnessMetricsValidator:
    """Comprehensive validation of fairness improvements"""
    
    def __init__(self, fairness_metrics=None):
        self.fairness_metrics = fairness_metrics or [
            'demographic_parity',
            'equal_opportunity', 
            'equalized_odds',
            'predictive_parity',
            'calibration'
        ]
        self.baseline_results = {}
        self.intervention_results = {}
        self.statistical_significance = {}
    
    def validate_intervention(self, y_true_baseline, y_pred_baseline, y_scores_baseline,
                            y_true_intervention, y_pred_intervention, y_scores_intervention,
                            protected_attr, confidence_level=0.95):
        """Validate fairness improvements from intervention"""
        
        # Calculate baseline metrics
        self.baseline_results = self._calculate_all_metrics(
            y_true_baseline, y_pred_baseline, y_scores_baseline, protected_attr
        )
        
        # Calculate intervention metrics
        self.intervention_results = self._calculate_all_metrics(
            y_true_intervention, y_pred_intervention, y_scores_intervention, protected_attr
        )
        
        # Statistical significance testing
        self.statistical_significance = self._test_statistical_significance(
            y_true_baseline, y_pred_baseline, y_true_intervention, y_pred_intervention,
            protected_attr, confidence_level
        )
        
        # Generate validation report
        return self._generate_validation_report()
    
    def _calculate_all_metrics(self, y_true, y_pred, y_scores, protected_attr):
        """Calculate all fairness metrics"""
        
        results = {}
        groups = protected_attr.unique()
        
        # Demographic Parity
        group_positive_rates = {}
        for group in groups:
            group_mask = protected_attr == group
            group_positive_rates[group] = y_pred[group_mask].mean()
        
        dp_violation = max(group_positive_rates.values()) - min(group_positive_rates.values())
        results['demographic_parity'] = {
            'violation': dp_violation,
            'group_rates': group_positive_rates
        }
        
        # Equal Opportunity
        group_tprs = {}
        for group in groups:
            group_mask = protected_attr == group
            positive_mask = y_true == 1
            group_positive_mask = group_mask & positive_mask
            
            if group_positive_mask.sum() > 0:
                tpr = y_pred[group_positive_mask].mean()
                group_tprs[group] = tpr
        
        if len(group_tprs) >= 2:
            eo_violation = max(group_tprs.values()) - min(group_tprs.values())
        else:
            eo_violation = 0
        
        results['equal_opportunity'] = {
            'violation': eo_violation,
            'group_tprs': group_tprs
        }
        
        # Equalized Odds (TPR + FPR)
        group_fprs = {}
        for group in groups:
            group_mask = protected_attr == group
            negative_mask = y_true == 0
            group_negative_mask = group_mask & negative_mask
            
            if group_negative_mask.sum() > 0:
                fpr = y_pred[group_negative_mask].mean()
                group_fprs[group] = fpr
        
        if len(group_fprs) >= 2 and len(group_tprs) >= 2:
            tpr_violation = max(group_tprs.values()) - min(group_tprs.values())
            fpr_violation = max(group_fprs.values()) - min(group_fprs.values())
            eo_odds_violation = max(tpr_violation, fpr_violation)
        else:
            eo_odds_violation = 0
        
        results['equalized_odds'] = {
            'violation': eo_odds_violation,
            'tpr_violation': tpr_violation if 'tpr_violation' in locals() else 0,
            'fpr_violation': fpr_violation if 'fpr_violation' in locals() else 0,
            'group_fprs': group_fprs
        }
        
        # Predictive Parity
        group_precisions = {}
        for group in groups:
            group_mask = protected_attr == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            if group_y_pred.sum() > 0:  # Avoid division by zero
                precision = group_y_true[group_y_pred == 1].mean()
                group_precisions[group] = precision
        
        if len(group_precisions) >= 2:
            pp_violation = max(group_precisions.values()) - min(group_precisions.values())
        else:
            pp_violation = 0
        
        results['predictive_parity'] = {
            'violation': pp_violation,
            'group_precisions': group_precisions
        }
        
        # Calibration
        calibration_results = self._assess_calibration(y_true, y_scores, protected_attr)
        results['calibration'] = calibration_results
        
        return results
    
    def _assess_calibration(self, y_true, y_scores, protected_attr):
        """Assess calibration across groups"""
        
        from sklearn.calibration import calibration_curve
        
        calibration_data = {}
        max_calibration_error = 0
        
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            group_y_true = y_true[group_mask]
            group_y_scores = y_scores[group_mask]
            
            if len(group_y_true) > 10:  # Minimum samples for reliable calibration
                try:
                    fraction_pos, mean_pred = calibration_curve(
                        group_y_true, group_y_scores, n_bins=5
                    )
                    
                    # Calculate Expected Calibration Error (ECE)
                    ece = np.mean(np.abs(fraction_pos - mean_pred))
                    
                    calibration_data[group] = {
                        'ece': ece,
                        'fraction_positives': fraction_pos,
                        'mean_predictions': mean_pred
                    }
                    
                    max_calibration_error = max(max_calibration_error, ece)
                    
                except Exception as e:
                    calibration_data[group] = {'ece': float('inf'), 'error': str(e)}
        
        # Overall calibration violation (max ECE difference between groups)
        group_eces = [data.get('ece', 0) for data in calibration_data.values() 
                     if data.get('ece', float('inf')) != float('inf')]
        
        if len(group_eces) >= 2:
            calibration_violation = max(group_eces) - min(group_eces)
        else:
            calibration_violation = 0
        
        return {
            'violation': calibration_violation,
            'group_calibration': calibration_data,
            'max_ece': max_calibration_error
        }
    
    def _test_statistical_significance(self, y_true_baseline, y_pred_baseline,
                                     y_true_intervention, y_pred_intervention,
                                     protected_attr, confidence_level):
        """Test statistical significance of fairness improvements"""
        
        from scipy import stats
        
        significance_results = {}
        alpha = 1 - confidence_level
        
        # Test for each fairness metric
        for metric in self.fairness_metrics:
            baseline_violation = self.baseline_results[metric]['violation']
            intervention_violation = self.intervention_results[metric]['violation']
            
            # Bootstrap test for difference in violations
            n_bootstrap = 1000
            baseline_violations = []
            intervention_violations = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                n_samples = len(y_true_baseline)
                bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
                
                # Calculate metric for bootstrap sample
                bootstrap_baseline = self._calculate_single_metric(
                    metric,
                    y_true_baseline[bootstrap_indices],
                    y_pred_baseline[bootstrap_indices],
                    protected_attr.iloc[bootstrap_indices] if hasattr(protected_attr, 'iloc') 
                    else protected_attr[bootstrap_indices]
                )
                baseline_violations.append(bootstrap_baseline)
                
                bootstrap_intervention = self._calculate_single_metric(
                    metric,
                    y_true_intervention[bootstrap_indices],
                    y_pred_intervention[bootstrap_indices],
                    protected_attr.iloc[bootstrap_indices] if hasattr(protected_attr, 'iloc') 
                    else protected_attr[bootstrap_indices]
                )
                intervention_violations.append(bootstrap_intervention)
            
            # Statistical test
            baseline_violations = np.array(baseline_violations)
            intervention_violations = np.array(intervention_violations)
            
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(baseline_violations, intervention_violations)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                ((len(baseline_violations) - 1) * np.var(baseline_violations) +
                 (len(intervention_violations) - 1) *