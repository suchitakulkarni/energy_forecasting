"""
Model Drift Monitoring
======================

Automated monitoring of model performance for drift detection.
Logs predictions vs actuals and alerts when performance degrades.

Usage:
    python drift_monitor.py --country ES --alert-threshold 0.20

Can be run as:
- Cron job (daily monitoring)
- GitHub Action (automated checks)
- Standalone script
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class DriftMonitor:
    """
    Monitor model performance and detect drift.
    """
    
    def __init__(self, country, log_dir='monitoring_logs'):
        """
        Initialize drift monitor.
        
        Args:
            country: Country code (ES, DE, etc.)
            log_dir: Directory to store monitoring logs
        """
        self.country = country
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.log_file = self.log_dir / f'{country}_predictions.csv'
        self.metrics_file = self.log_dir / f'{country}_metrics.json'
    
    def fetch_and_predict(self):
        """
        Fetch latest data and make predictions.
        
        Returns:
            DataFrame with predictions and actuals
        """
        from entsoe_data_fetcher import EntsoeDataFetcher
        from model_training_pipeline import EnergyPriceForecaster
        
        # Load model
        model_path = Path(f"models/{self.country}_forecaster")
        
        if not model_path.with_name(f"{self.country}_forecaster_stat.pkl").exists():
            raise FileNotFoundError(f"No model found for {self.country}")
        
        forecaster = EnergyPriceForecaster.load(model_path)
        
        # Fetch yesterday's data
        fetcher = EntsoeDataFetcher()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Get week for context
        
        df = fetcher.fetch_complete_dataset(
            country=self.country,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            include_forecasts=False
        )
        
        # Engineer features
        df_feat = forecaster.engineer_features(df)
        
        # Get yesterday's data
        yesterday = end_date - timedelta(days=1)
        yesterday_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday_end = yesterday_start + timedelta(hours=24)
        
        mask = (df.index >= yesterday_start) & (df.index < yesterday_end)
        df_yesterday = df_feat[mask]
        
        if len(df_yesterday) == 0:
            print(f"No data available for yesterday ({yesterday_start.date()})")
            return None
        
        # Prepare features
        X_stat = df_yesterday[forecaster.feature_names_stat].copy()
        X_phys = df_yesterday[forecaster.feature_names_phys].copy()
        
        X_stat = X_stat.ffill().bfill().fillna(0)
        X_phys = X_phys.ffill().bfill().fillna(0)
        
        # Predictions
        y_pred_stat = forecaster.stat_model.predict(X_stat)
        y_pred_phys = forecaster.phys_model.predict(X_phys)
        y_actual = df.loc[df_yesterday.index, 'price'].values
        
        # Create results dataframe
        results = pd.DataFrame({
            'timestamp': df_yesterday.index,
            'actual': y_actual,
            'pred_stat': y_pred_stat,
            'pred_phys': y_pred_phys,
            'error_stat': np.abs(y_actual - y_pred_stat),
            'error_phys': np.abs(y_actual - y_pred_phys),
        })
        
        return results
    
    def log_predictions(self, results):
        """
        Log predictions to CSV file.
        
        Args:
            results: DataFrame with predictions
        """
        if results is None:
            return
        
        # Append to log file
        if self.log_file.exists():
            existing = pd.read_csv(self.log_file, index_col=0, parse_dates=True)
            combined = pd.concat([existing, results.set_index('timestamp')])
            # Remove duplicates (keep latest)
            combined = combined[~combined.index.duplicated(keep='last')]
            combined.to_csv(self.log_file)
        else:
            results.set_index('timestamp').to_csv(self.log_file)
        
        print(f"Logged {len(results)} predictions to {self.log_file}")
    
    def compute_metrics(self, window_days=7):
        """
        Compute performance metrics over recent window.
        
        Args:
            window_days: Number of days to compute metrics over
            
        Returns:
            dict with metrics
        """
        if not self.log_file.exists():
            return None
        
        # Load logs
        logs = pd.read_csv(self.log_file, index_col=0, parse_dates=True)
        
        # Filter to recent window
        cutoff = datetime.now() - timedelta(days=window_days)
        recent = logs[logs.index >= cutoff]
        
        if len(recent) == 0:
            return None
        
        # Compute metrics
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'window_days': window_days,
            'n_samples': len(recent),
            'mae_stat': recent['error_stat'].mean(),
            'mae_phys': recent['error_phys'].mean(),
            'rmse_stat': np.sqrt((recent['error_stat'] ** 2).mean()),
            'rmse_phys': np.sqrt((recent['error_phys'] ** 2).mean()),
            'mae_stat_median': recent['error_stat'].median(),
            'mae_phys_median': recent['error_phys'].median(),
            'mae_stat_p95': recent['error_stat'].quantile(0.95),
            'mae_phys_p95': recent['error_phys'].quantile(0.95),
        }
        
        # Load model baseline
        model_path = Path(f"models/{self.country}_forecaster_metadata.json")
        if model_path.exists():
            with open(model_path, 'r') as f:
                baseline = json.load(f)
            
            metrics['baseline_mae_stat'] = baseline.get('mae_stat', 0)
            metrics['baseline_mae_phys'] = baseline.get('mae_phys', 0)
            
            # Drift calculation
            if metrics['baseline_mae_stat'] > 0:
                metrics['drift_stat'] = (metrics['mae_stat'] - metrics['baseline_mae_stat']) / metrics['baseline_mae_stat']
                metrics['drift_phys'] = (metrics['mae_phys'] - metrics['baseline_mae_phys']) / metrics['baseline_mae_phys']
            else:
                metrics['drift_stat'] = 0
                metrics['drift_phys'] = 0
        
        return metrics
    
    def check_drift(self, threshold=0.20):
        """
        Check if drift exceeds threshold.
        
        Args:
            threshold: Drift threshold (e.g., 0.20 = 20% increase in MAE)
            
        Returns:
            tuple: (alert_triggered, drift_pct, message)
        """
        metrics = self.compute_metrics(window_days=7)
        
        if metrics is None:
            return False, 0, "Insufficient data for drift detection"
        
        drift_phys = metrics.get('drift_phys', 0)
        drift_stat = metrics.get('drift_stat', 0)
        
        if abs(drift_phys) > threshold:
            alert = True
            message = f"ALERT: Physics model drift {drift_phys*100:.1f}% (threshold: {threshold*100:.0f}%)"
        elif abs(drift_stat) > threshold:
            alert = True
            message = f"ALERT: Statistical model drift {drift_stat*100:.1f}% (threshold: {threshold*100:.0f}%)"
        else:
            alert = False
            message = f"OK: Drift within acceptable range (Physics: {drift_phys*100:.1f}%, Stat: {drift_stat*100:.1f}%)"
        
        return alert, drift_phys * 100, message
    
    def save_metrics(self):
        """Save current metrics to JSON."""
        metrics = self.compute_metrics(window_days=7)
        
        if metrics:
            # Append to metrics history
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            history.append(metrics)
            
            # Keep last 90 days
            cutoff = datetime.now() - timedelta(days=90)
            history = [m for m in history 
                      if datetime.fromisoformat(m['timestamp']) >= cutoff]
            
            with open(self.metrics_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            print(f"Saved metrics to {self.metrics_file}")
    
    def generate_report(self):
        """Generate drift monitoring report."""
        metrics = self.compute_metrics(window_days=7)
        
        if metrics is None:
            print("No data available for report")
            return
        
        print("\n" + "="*70)
        print(f"DRIFT MONITORING REPORT - {self.country}")
        print("="*70)
        print(f"\nTime: {metrics['timestamp']}")
        print(f"Window: Last {metrics['window_days']} days ({metrics['n_samples']} hours)")
        
        print(f"\nCurrent Performance:")
        print(f"  Statistical MAE:  {metrics['mae_stat']:.2f} EUR/MWh")
        print(f"  Physics MAE:      {metrics['mae_phys']:.2f} EUR/MWh")
        
        if 'baseline_mae_stat' in metrics:
            print(f"\nBaseline (from training):")
            print(f"  Statistical MAE:  {metrics['baseline_mae_stat']:.2f} EUR/MWh")
            print(f"  Physics MAE:      {metrics['baseline_mae_phys']:.2f} EUR/MWh")
            
            print(f"\nDrift:")
            print(f"  Statistical:      {metrics['drift_stat']*100:+.1f}%")
            print(f"  Physics:          {metrics['drift_phys']*100:+.1f}%")
        
        print(f"\nRobustness Metrics:")
        print(f"  Median Error (Stat):  {metrics['mae_stat_median']:.2f} EUR/MWh")
        print(f"  Median Error (Phys):  {metrics['mae_phys_median']:.2f} EUR/MWh")
        print(f"  95th Percentile (Stat): {metrics['mae_stat_p95']:.2f} EUR/MWh")
        print(f"  95th Percentile (Phys): {metrics['mae_phys_p95']:.2f} EUR/MWh")
        
        print("="*70)


def main():
    """Main monitoring script."""
    parser = argparse.ArgumentParser(description='Monitor model drift')
    parser.add_argument('--country', type=str, default='ES', help='Country code')
    parser.add_argument('--alert-threshold', type=float, default=0.20,
                       help='Drift threshold for alerts (default: 0.20 = 20%%)')
    parser.add_argument('--report-only', action='store_true',
                       help='Only generate report, do not fetch new data')
    
    args = parser.parse_args()
    
    monitor = DriftMonitor(args.country)
    
    if not args.report_only:
        # Fetch and log predictions
        print(f"Fetching predictions for {args.country}...")
        results = monitor.fetch_and_predict()
        
        if results is not None:
            monitor.log_predictions(results)
            monitor.save_metrics()
    
    # Generate report
    monitor.generate_report()
    
    # Check drift
    alert, drift_pct, message = monitor.check_drift(threshold=args.alert_threshold)
    
    print(f"\n{message}")
    
    if alert:
        print("\n*** RETRAIN RECOMMENDED ***")
    
    print(f"\nMonitoring logs: {monitor.log_dir}")


if __name__ == "__main__":
    main()
