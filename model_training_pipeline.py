"""
Production Model Training Pipeline
===================================

Proper ML workflow for energy price forecasting:
- Time-series cross-validation
- Hyperparameter optimization with Optuna
- Model versioning and persistence
- Performance monitoring

Usage:
    python model_training_pipeline.py --country ES --optimize
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')


class EnergyPriceForecaster:
    """
    Production-grade energy price forecasting model.
    
    Handles:
    - Feature engineering
    - Time-series CV
    - Hyperparameter optimization
    - Model persistence
    """
    
    def __init__(self, model_type='hist_gradient_boosting'):
        """
        Initialize forecaster.
        
        Args:
            model_type: 'gradient_boosting' or 'hist_gradient_boosting'
        """
        self.model_type = model_type
        self.stat_model = None
        self.phys_model = None
        self.best_params_stat = None
        self.best_params_phys = None
        self.feature_names_stat = None
        self.feature_names_phys = None
        self.metadata = {}
        
    def engineer_features(self, df):
        """
        Feature engineering for energy price forecasting.
        
        Args:
            df: DataFrame with raw ENTSO-E data
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # === LAG FEATURES ===
        # Price lags (momentum)
        for lag in [1, 24, 168]:  # 1h, 1day, 1week
            df[f'price_lag_{lag}h'] = df['price'].shift(lag)
        
        # Rolling statistics
        for window in [24, 168]:  # 1day, 1week
            df[f'price_rolling_mean_{window}h'] = df['price'].rolling(window).mean()
            df[f'price_rolling_std_{window}h'] = df['price'].rolling(window).std()
        
        # Demand lags
        for lag in [1, 24]:
            df[f'demand_lag_{lag}h'] = df['demand'].shift(lag)
        
        # === PHYSICS FEATURES ===
        # Generation features
        gen_cols = ['nuclear', 'coal', 'gas', 'wind_onshore', 'solar']
        available_gen = [c for c in gen_cols if c in df.columns]
        
        if available_gen:
            df['total_generation'] = df[available_gen].sum(axis=1)
        
        # Residual demand (demand after renewables)
        renewable_cols = [c for c in ['wind_onshore', 'wind_offshore', 'solar'] if c in df.columns]
        if renewable_cols and 'demand' in df.columns:
            df['residual_demand'] = df['demand'] - df[renewable_cols].sum(axis=1)
            df['residual_demand_lag_1h'] = df['residual_demand'].shift(1)
        
        # Renewable penetration
        if renewable_cols and 'demand' in df.columns:
            df['renewable_penetration'] = df[renewable_cols].sum(axis=1) / (df['demand'] + 1)
        
        # Utilization (if we know total capacity - assume 68000 MW for now)
        if 'total_generation' in df.columns:
            # This should ideally come from ENTSO-E capacity data
            estimated_capacity = df['total_generation'].quantile(0.95) * 1.2
            df['utilization'] = df['total_generation'] / estimated_capacity
            df['utilization_squared'] = df['utilization'] ** 2
        
        # Ramp rates (physics: how fast things change)
        if 'demand' in df.columns:
            df['demand_ramp_1h'] = df['demand'].diff(1)
            df['demand_ramp_3h'] = df['demand'].diff(3)
        
        if 'residual_demand' in df.columns:
            df['residual_ramp'] = df['residual_demand'].diff(1)
        
        # Evening ramp stress (duck curve effect)
        if 'solar' in df.columns and 'hour' in df.columns:
            is_evening = (df['hour'] >= 17) & (df['hour'] <= 20)
            df['evening_solar_drop'] = 0.0
            df.loc[is_evening, 'evening_solar_drop'] = -df['solar'].diff().fillna(0)
        
        # Gas/Coal interaction with hour (merit order effects)
        if 'gas' in df.columns and 'hour' in df.columns:
            df['gas_hour_interaction'] = df['gas'] * df['hour']
        
        # Time features
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        if 'day_of_week' in df.columns:
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def prepare_features(self, df, include_physics=True):
        """
        Select and prepare features for modeling.
        
        Args:
            df: DataFrame with engineered features
            include_physics: If True, include physics features
            
        Returns:
            X, y, feature_names
        """
        # Base features (statistical model)
        base_features = ['hour', 'day_of_week', 'demand']
        
        # Add time features
        for feat in ['hour_sin', 'hour_cos', 'is_weekend', 'day_of_year', 'month']:
            if feat in df.columns:
                base_features.append(feat)
        
        # Add generation features
        for feat in ['nuclear', 'coal', 'gas', 'wind_onshore', 'solar']:
            if feat in df.columns:
                base_features.append(feat)
        
        # Add lag features
        lag_features = [c for c in df.columns if 'lag' in c or 'rolling' in c]
        base_features.extend(lag_features)
        
        # Physics features (additional to base)
        physics_features = []
        if include_physics:
            physics_candidates = [
                'residual_demand', 'residual_ramp', 'renewable_penetration',
                'utilization', 'utilization_squared', 'demand_ramp_1h', 'demand_ramp_3h',
                'evening_solar_drop', 'gas_hour_interaction', 'total_generation'
            ]
            physics_features = [f for f in physics_candidates if f in df.columns]
        
        # Combine
        if include_physics:
            all_features = base_features + physics_features
        else:
            all_features = base_features
        
        # Remove duplicates, preserve order
        all_features = list(dict.fromkeys(all_features))
        
        # Filter to available features
        available_features = [f for f in all_features if f in df.columns]
        
        # Extract X and y
        X = df[available_features].copy()
        y = df['price'].copy() if 'price' in df.columns else None
        
        # Handle NaNs (from lag features)
        # Fill with forward-fill, then backward-fill, then 0
        X = X.ffill().bfill().fillna(0)
        
        # Drop rows where target is NaN
        if y is not None:
            valid_idx = y.notna()
            X = X[valid_idx]
            y = y[valid_idx]
        
        return X, y, available_features
    
    def time_series_cv_score(self, X, y, model_class, params, n_splits=5):
        """
        Time-series cross-validation score.
        
        Args:
            X: Features
            y: Target
            model_class: Model class (GradientBoostingRegressor, etc.)
            params: Hyperparameters dict
            n_splits: Number of CV splits
            
        Returns:
            mean_mae, std_mae
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        mae_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = model_class(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            mae = mean_absolute_error(y_val, y_pred)
            mae_scores.append(mae)
        
        return np.mean(mae_scores), np.std(mae_scores)
    
    def optimize_hyperparameters(self, X, y, model_type='stat', n_trials=50, n_splits=5):
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X: Features
            y: Target
            model_type: 'stat' or 'phys' (for logging)
            n_trials: Number of Optuna trials
            n_splits: Number of CV splits
            
        Returns:
            best_params dict
        """
        
        def objective(trial):
            if self.model_type == 'hist_gradient_boosting':
                params = {
                    'max_iter': trial.suggest_int('max_iter', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
                    'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 10.0),
                    'random_state': 42
                }
                model_class = HistGradientBoostingRegressor
            else:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'random_state': 42
                }
                model_class = GradientBoostingRegressor
            
            mean_mae, _ = self.time_series_cv_score(X, y, model_class, params, n_splits)
            return mean_mae
        
        print(f"Optimizing {model_type} model hyperparameters...")
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\n{model_type.upper()} Model - Best MAE: {study.best_value:.2f} EUR/MWh")
        print(f"Best params: {study.best_params}")
        
        return study.best_params
    
    def train(self, df, optimize=True, n_trials=50, test_size_days=30, quantiles=[0.1, 0.5, 0.9]):
        """
        Train statistical and physics-informed models with uncertainty quantification.
        
        Args:
            df: DataFrame with raw data
            optimize: Whether to optimize hyperparameters
            n_trials: Number of Optuna trials
            test_size_days: Size of test set in days
            quantiles: Quantiles to predict for uncertainty bands
            
        Returns:
            dict with training results
        """
        print("="*70)
        print("ENERGY PRICE FORECASTING - MODEL TRAINING")
        print("="*70)
        
        # Feature engineering
        print("\n[1/6] Feature engineering...")
        df = self.engineer_features(df)
        
        # Prepare features
        print("\n[2/6] Preparing features...")
        X_stat, y, self.feature_names_stat = self.prepare_features(df, include_physics=False)
        X_phys, _, self.feature_names_phys = self.prepare_features(df, include_physics=True)
        
        print(f"  Statistical features: {len(self.feature_names_stat)}")
        print(f"  Physics features: {len(self.feature_names_phys)}")
        print(f"  Total samples: {len(X_stat)}")
        
        # Train/test split (time-based)
        test_size = test_size_days * 24
        split_idx = len(X_stat) - test_size
        
        X_train_stat = X_stat.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test_stat = X_stat.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        X_train_phys = X_phys.iloc[:split_idx]
        X_test_phys = X_phys.iloc[split_idx:]
        
        print(f"  Train samples: {len(X_train_stat)}")
        print(f"  Test samples: {len(X_test_stat)}")
        
        # Hyperparameter optimization
        if optimize:
            print("\n[3/6] Hyperparameter optimization...")
            self.best_params_stat = self.optimize_hyperparameters(
                X_train_stat, y_train, model_type='stat', n_trials=n_trials
            )
            self.best_params_phys = self.optimize_hyperparameters(
                X_train_phys, y_train, model_type='phys', n_trials=n_trials
            )
        else:
            print("\n[3/6] Using default hyperparameters...")
            if self.model_type == 'hist_gradient_boosting':
                self.best_params_stat = {'max_iter': 200, 'max_depth': 8, 'learning_rate': 0.05, 'random_state': 42}
                self.best_params_phys = {'max_iter': 200, 'max_depth': 8, 'learning_rate': 0.05, 'random_state': 42}
            else:
                self.best_params_stat = {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'random_state': 42}
                self.best_params_phys = {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'random_state': 42}

        # Run CV once on final params to get std for error bars
        print("\n   Computing CV std for error bar estimates...")
        model_class = HistGradientBoostingRegressor if self.model_type == 'hist_gradient_boosting' else GradientBoostingRegressor
        _, cv_std_stat = self.time_series_cv_score(X_train_stat, y_train, model_class, self.best_params_stat)
        _, cv_std_phys = self.time_series_cv_score(X_train_phys, y_train, model_class, self.best_params_phys)
        print(f"   CV std - Statistical: {cv_std_stat:.2f}, Physics: {cv_std_phys:.2f}")
        
        # Train point prediction models
        print("\n[4/6] Training point prediction models...")
        
        if self.model_type == 'hist_gradient_boosting':
            self.stat_model = HistGradientBoostingRegressor(**self.best_params_stat)
            self.phys_model = HistGradientBoostingRegressor(**self.best_params_phys)
        else:
            self.stat_model = GradientBoostingRegressor(**self.best_params_stat)
            self.phys_model = GradientBoostingRegressor(**self.best_params_phys)
        
        self.stat_model.fit(X_train_stat, y_train)
        self.phys_model.fit(X_train_phys, y_train)
        
        # Train quantile models for uncertainty
        print("\n[5/6] Training quantile models for uncertainty...")
        self.quantile_models_stat = {}
        self.quantile_models_phys = {}
        
        for q in quantiles:
            if q == 0.5:
                continue  # Use main model for median
            
            print(f"  Training quantile {q:.1f}...")
            
            # Statistical model quantiles
            if self.model_type == 'hist_gradient_boosting':
                model_stat_q = HistGradientBoostingRegressor(
                    **{**self.best_params_stat, 'loss': 'quantile', 'quantile': q}
                )
                model_phys_q = HistGradientBoostingRegressor(
                    **{**self.best_params_phys, 'loss': 'quantile', 'quantile': q}
                )
            else:
                model_stat_q = GradientBoostingRegressor(
                    **{**self.best_params_stat, 'loss': 'quantile', 'alpha': q}
                )
                model_phys_q = GradientBoostingRegressor(
                    **{**self.best_params_phys, 'loss': 'quantile', 'alpha': q}
                )
            
            model_stat_q.fit(X_train_stat, y_train)
            model_phys_q.fit(X_train_phys, y_train)
            
            self.quantile_models_stat[q] = model_stat_q
            self.quantile_models_phys[q] = model_phys_q
        
        # Evaluate
        print("\n[6/6] Evaluating on test set...")
        y_pred_stat = self.stat_model.predict(X_test_stat)
        y_pred_phys = self.phys_model.predict(X_test_phys)
        
        # Get quantile predictions
        quantile_preds_stat = {q: model.predict(X_test_stat) 
                               for q, model in self.quantile_models_stat.items()}
        quantile_preds_phys = {q: model.predict(X_test_phys) 
                               for q, model in self.quantile_models_phys.items()}
        
        results = self._compute_metrics(
            y_test, y_pred_stat, y_pred_phys,
            quantile_preds_stat, quantile_preds_phys, quantiles
        )
        
        # Store metadata
        self.metadata = {
            'train_date': datetime.now().isoformat(),
            'n_train_samples': len(X_train_stat),
            'n_test_samples': len(X_test_stat),
            'test_size_days': test_size_days,
            'model_type': self.model_type,
            'optimized': optimize,
            'n_trials': n_trials if optimize else 0,
            'quantiles': quantiles,
            'mae_std_stat': cv_std_stat,
            'mae_std_phys': cv_std_phys,
            **results
        }
        
        self._print_results(results)
        
        return results
    
    def _compute_metrics(self, y_true, y_pred_stat, y_pred_phys, 
                        quantile_preds_stat=None, quantile_preds_phys=None, quantiles=None):
        """Compute evaluation metrics including performance by regime."""
        metrics = {}
        
        # Overall metrics
        for name, y_pred in [('stat', y_pred_stat), ('phys', y_pred_phys)]:
            metrics[f'mae_{name}'] = mean_absolute_error(y_true, y_pred)
            metrics[f'rmse_{name}'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics[f'r2_{name}'] = r2_score(y_true, y_pred)
        
        metrics['improvement_pct'] = (metrics['mae_stat'] - metrics['mae_phys']) / metrics['mae_stat'] * 100
        
        # Performance by price regime
        p90 = y_true.quantile(0.90)
        p10 = y_true.quantile(0.10)
        
        # Normal prices (10th - 90th percentile)
        normal_mask = (y_true >= p10) & (y_true <= p90)
        metrics['mae_stat_normal'] = mean_absolute_error(y_true[normal_mask], y_pred_stat[normal_mask])
        metrics['mae_phys_normal'] = mean_absolute_error(y_true[normal_mask], y_pred_phys[normal_mask])
        metrics['improvement_normal'] = (metrics['mae_stat_normal'] - metrics['mae_phys_normal']) / metrics['mae_stat_normal'] * 100
        
        # Price spikes (>90th percentile)
        spike_mask = y_true > p90
        if spike_mask.sum() > 10:  # Need enough samples
            metrics['mae_stat_spike'] = mean_absolute_error(y_true[spike_mask], y_pred_stat[spike_mask])
            metrics['mae_phys_spike'] = mean_absolute_error(y_true[spike_mask], y_pred_phys[spike_mask])
            metrics['improvement_spike'] = (metrics['mae_stat_spike'] - metrics['mae_phys_spike']) / metrics['mae_stat_spike'] * 100
        
        # Low prices (<10th percentile)
        low_mask = y_true < p10
        if low_mask.sum() > 10:
            metrics['mae_stat_low'] = mean_absolute_error(y_true[low_mask], y_pred_stat[low_mask])
            metrics['mae_phys_low'] = mean_absolute_error(y_true[low_mask], y_pred_phys[low_mask])
            metrics['improvement_low'] = (metrics['mae_stat_low'] - metrics['mae_phys_low']) / metrics['mae_stat_low'] * 100
        
        # Quantile coverage (if quantiles provided)
        if quantile_preds_stat and quantiles:
            for q in quantiles:
                if q == 0.5:
                    continue
                if q in quantile_preds_stat:
                    # Check coverage: what % of actuals fall below the q-quantile prediction?
                    coverage_stat = (y_true <= quantile_preds_stat[q]).mean()
                    coverage_phys = (y_true <= quantile_preds_phys[q]).mean()
                    metrics[f'coverage_stat_{int(q*100)}'] = coverage_stat
                    metrics[f'coverage_phys_{int(q*100)}'] = coverage_phys
        
        return metrics
    
    def _print_results(self, results):
        """Print training results including performance by regime."""
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"\nOverall Performance:")
        print(f"  Statistical Model MAE:  {results['mae_stat']:.2f} EUR/MWh")
        print(f"  Physics Model MAE:      {results['mae_phys']:.2f} EUR/MWh")
        print(f"  Improvement:            {results['improvement_pct']:.1f}%")
        
        print(f"\nPerformance by Price Regime:")
        print(f"  Normal Prices (10th-90th percentile):")
        print(f"    Statistical MAE: {results.get('mae_stat_normal', 0):.2f} EUR/MWh")
        print(f"    Physics MAE:     {results.get('mae_phys_normal', 0):.2f} EUR/MWh")
        print(f"    Improvement:     {results.get('improvement_normal', 0):.1f}%")
        
        if 'mae_stat_spike' in results:
            print(f"  Price Spikes (>90th percentile):")
            print(f"    Statistical MAE: {results['mae_stat_spike']:.2f} EUR/MWh")
            print(f"    Physics MAE:     {results['mae_phys_spike']:.2f} EUR/MWh")
            print(f"    Improvement:     {results['improvement_spike']:.1f}% ***")
        
        if 'mae_stat_low' in results:
            print(f"  Low Prices (<10th percentile):")
            print(f"    Statistical MAE: {results['mae_stat_low']:.2f} EUR/MWh")
            print(f"    Physics MAE:     {results['mae_phys_low']:.2f} EUR/MWh")
            print(f"    Improvement:     {results['improvement_low']:.1f}%")
        
        print("="*70)
    
    def save(self, filepath):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model (without extension)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save models
        with open(f"{filepath}_stat.pkl", 'wb') as f:
            pickle.dump(self.stat_model, f)
        
        with open(f"{filepath}_phys.pkl", 'wb') as f:
            pickle.dump(self.phys_model, f)
        
        # Save quantile models
        with open(f"{filepath}_quantile_stat.pkl", 'wb') as f:
            pickle.dump(self.quantile_models_stat, f)
        
        with open(f"{filepath}_quantile_phys.pkl", 'wb') as f:
            pickle.dump(self.quantile_models_phys, f)
        
        # Save metadata
        metadata_full = {
            'best_params_stat': self.best_params_stat,
            'best_params_phys': self.best_params_phys,
            'feature_names_stat': self.feature_names_stat,
            'feature_names_phys': self.feature_names_phys,
            'model_type': self.model_type,
            **self.metadata
        }
        
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata_full, f, indent=2)
        
        print(f"\nModel saved to {filepath}_[stat|phys].pkl")
    
    @classmethod
    def load(cls, filepath):
        """
        Load model from disk.
        
        Args:
            filepath: Path to model (without extension)
            
        Returns:
            EnergyPriceForecaster instance
        """
        filepath = Path(filepath)
        
        # Load metadata
        with open(f"{filepath}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        forecaster = cls(model_type=metadata['model_type'])
        
        # Load models
        with open(f"{filepath}_stat.pkl", 'rb') as f:
            forecaster.stat_model = pickle.load(f)
        
        with open(f"{filepath}_phys.pkl", 'rb') as f:
            forecaster.phys_model = pickle.load(f)
        
        # Load quantile models
        try:
            with open(f"{filepath}_quantile_stat.pkl", 'rb') as f:
                forecaster.quantile_models_stat = pickle.load(f)
            
            with open(f"{filepath}_quantile_phys.pkl", 'rb') as f:
                forecaster.quantile_models_phys = pickle.load(f)
        except FileNotFoundError:
            # Older models without quantile support
            forecaster.quantile_models_stat = {}
            forecaster.quantile_models_phys = {}
        
        # Restore metadata
        forecaster.best_params_stat = metadata['best_params_stat']
        forecaster.best_params_phys = metadata['best_params_phys']
        forecaster.feature_names_stat = metadata['feature_names_stat']
        forecaster.feature_names_phys = metadata['feature_names_phys']
        forecaster.metadata = metadata
        
        print(f"Model loaded from {filepath}")
        print(f"  Trained: {metadata['train_date']}")
        print(f"  Test MAE (stat): {metadata['mae_stat']:.2f}")
        print(f"  Test MAE (phys): {metadata['mae_phys']:.2f}")
        print(f"  Improvement: {metadata['improvement_pct']:.1f}%")
        
        return forecaster


def main():
    """Example usage."""
    import argparse
    from entsoe_data_fetcher import EntsoeDataFetcher
    
    parser = argparse.ArgumentParser(description='Train energy price forecasting models')
    parser.add_argument('--country', type=str, default='ES', help='Country code')
    parser.add_argument('--start', type=str, default='2023-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2023-12-31', help='End date')
    parser.add_argument('--optimize', action='store_true', help='Optimize hyperparameters')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--model-type', type=str, default='hist_gradient_boosting',
                       choices=['gradient_boosting', 'hist_gradient_boosting'])
    parser.add_argument('--output', type=str, default='models/energy_forecaster',
                       help='Output path for model')
    
    args = parser.parse_args()
    
    # Fetch data
    print(f"Fetching {args.country} data from {args.start} to {args.end}...")
    fetcher = EntsoeDataFetcher()
    df = fetcher.fetch_complete_dataset(
        country=args.country,
        start_date=args.start,
        end_date=args.end,
        include_forecasts=False
    )
    df = fetcher.compute_physics_features(df)
    
    # Train model
    forecaster = EnergyPriceForecaster(model_type=args.model_type)
    results = forecaster.train(
        df,
        optimize=args.optimize,
        n_trials=args.n_trials
    )
    
    # Save model
    forecaster.save(args.output)
    
    print(f"\nDone! Model ready for deployment.")


if __name__ == "__main__":
    main()
