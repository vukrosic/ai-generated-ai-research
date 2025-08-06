"""
Experiment results storage and retrieval system.

This module provides comprehensive storage, querying, and filtering capabilities
for experiment results in the AI curve fitting research project.
"""

import json
import sqlite3
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from enum import Enum

try:
    from .config import ExperimentConfig
    from .runner import ExperimentResults
except ImportError:
    from config import ExperimentConfig
    # Create minimal stub for ExperimentResults
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, Any, Optional, List
    
    @dataclass
    class ExperimentResults:
        experiment_id: str
        timestamp: datetime
        config: 'ExperimentConfig'
        status: str = "completed"
        final_train_loss: float = 0.0
        final_val_loss: float = 0.0
        best_val_loss: float = 0.0
        training_time: float = 0.0
        duration_seconds: float = 0.0
        convergence_epoch: Optional[int] = None
        model_size: Optional[int] = None
        error_message: Optional[str] = None
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'experiment_id': self.experiment_id,
                'timestamp': self.timestamp.isoformat(),
                'config': self.config.to_dict(),
                'status': self.status,
                'final_train_loss': self.final_train_loss,
                'final_val_loss': self.final_val_loss,
                'best_val_loss': self.best_val_loss,
                'training_time': self.training_time,
                'duration_seconds': self.duration_seconds,
                'convergence_epoch': self.convergence_epoch,
                'model_size': self.model_size,
                'error_message': self.error_message
            }
        
        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentResults':
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            data['config'] = ExperimentConfig.from_dict(data['config'])
            return cls(**data)


class StorageBackend(Enum):
    """Available storage backends."""
    JSON = "json"
    SQLITE = "sqlite"
    PICKLE = "pickle"


@dataclass
class QueryFilter:
    """Filter criteria for experiment queries."""
    
    # Configuration filters
    polynomial_degrees: Optional[List[int]] = None
    model_architectures: Optional[List[str]] = None
    optimizers: Optional[List[str]] = None
    noise_levels: Optional[List[float]] = None
    
    # Performance filters
    min_final_train_loss: Optional[float] = None
    max_final_train_loss: Optional[float] = None
    min_final_val_loss: Optional[float] = None
    max_final_val_loss: Optional[float] = None
    min_best_val_loss: Optional[float] = None
    max_best_val_loss: Optional[float] = None
    
    # Time filters
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    
    # Status filters
    statuses: Optional[List[str]] = None
    
    # Custom filter function
    custom_filter: Optional[Callable[[ExperimentResults], bool]] = None
    
    def matches(self, result: ExperimentResults) -> bool:
        """Check if an experiment result matches this filter."""
        
        # Configuration filters
        if self.polynomial_degrees and result.config.polynomial_degree not in self.polynomial_degrees:
            return False
        
        if self.model_architectures and result.config.model_architecture not in self.model_architectures:
            return False
        
        if self.optimizers and result.config.optimizer not in self.optimizers:
            return False
        
        if self.noise_levels and result.config.noise_level not in self.noise_levels:
            return False
        
        # Performance filters
        if self.min_final_train_loss is not None and result.final_train_loss < self.min_final_train_loss:
            return False
        
        if self.max_final_train_loss is not None and result.final_train_loss > self.max_final_train_loss:
            return False
        
        if self.min_final_val_loss is not None and result.final_val_loss < self.min_final_val_loss:
            return False
        
        if self.max_final_val_loss is not None and result.final_val_loss > self.max_final_val_loss:
            return False
        
        if self.min_best_val_loss is not None and result.best_val_loss < self.min_best_val_loss:
            return False
        
        if self.max_best_val_loss is not None and result.best_val_loss > self.max_best_val_loss:
            return False
        
        # Time filters
        if self.start_date and result.timestamp < self.start_date:
            return False
        
        if self.end_date and result.timestamp > self.end_date:
            return False
        
        if self.min_duration is not None and result.duration_seconds < self.min_duration:
            return False
        
        if self.max_duration is not None and result.duration_seconds > self.max_duration:
            return False
        
        # Status filters
        if self.statuses and result.status not in self.statuses:
            return False
        
        # Custom filter
        if self.custom_filter and not self.custom_filter(result):
            return False
        
        return True


class ExperimentStorage:
    """
    Comprehensive experiment results storage and retrieval system.
    
    Supports multiple storage backends (JSON, SQLite, Pickle) with advanced
    querying, filtering, and analysis capabilities.
    """
    
    def __init__(self, 
                 storage_dir: str = "experiments",
                 backend: StorageBackend = StorageBackend.JSON,
                 db_name: str = "experiments.db"):
        """
        Initialize the experiment storage system.
        
        Args:
            storage_dir: Directory for storing experiment data
            backend: Storage backend to use
            db_name: Database name for SQLite backend
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.backend = backend
        self.db_path = self.storage_dir / db_name
        
        if backend == StorageBackend.SQLITE:
            self._init_sqlite_db()
    
    def _init_sqlite_db(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create experiments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    status TEXT NOT NULL,
                    
                    -- Configuration
                    polynomial_degree INTEGER,
                    noise_level REAL,
                    train_val_split REAL,
                    model_architecture TEXT,
                    optimizer TEXT,
                    learning_rate REAL,
                    batch_size INTEGER,
                    epochs INTEGER,
                    random_seed INTEGER,
                    
                    -- Performance metrics
                    final_train_loss REAL,
                    final_val_loss REAL,
                    best_val_loss REAL,
                    training_time REAL,
                    convergence_epoch INTEGER,
                    model_size INTEGER,
                    
                    -- Metadata
                    error_message TEXT,
                    
                    -- Full data (JSON)
                    full_data TEXT NOT NULL
                )
            ''')
            
            # Create indices for common queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON experiments(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON experiments(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_polynomial_degree ON experiments(polynomial_degree)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_architecture ON experiments(model_architecture)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_optimizer ON experiments(optimizer)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_final_val_loss ON experiments(final_val_loss)')
            
            conn.commit()
    
    def store_experiment(self, result: ExperimentResults) -> bool:
        """
        Store an experiment result.
        
        Args:
            result: ExperimentResults to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.backend == StorageBackend.JSON:
                return self._store_json(result)
            elif self.backend == StorageBackend.SQLITE:
                return self._store_sqlite(result)
            elif self.backend == StorageBackend.PICKLE:
                return self._store_pickle(result)
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
        except Exception as e:
            print(f"Error storing experiment {result.experiment_id}: {e}")
            return False
    
    def _store_json(self, result: ExperimentResults) -> bool:
        """Store experiment result as JSON."""
        exp_dir = self.storage_dir / result.experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        result_path = exp_dir / "result.json"
        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        return True
    
    def _store_sqlite(self, result: ExperimentResults) -> bool:
        """Store experiment result in SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO experiments VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            ''', (
                result.experiment_id,
                result.timestamp.isoformat(),
                result.duration_seconds,
                result.status,
                result.config.polynomial_degree,
                result.config.noise_level,
                result.config.train_val_split,
                result.config.model_architecture,
                result.config.optimizer,
                result.config.learning_rate,
                result.config.batch_size,
                result.config.epochs,
                result.config.random_seed,
                result.final_train_loss,
                result.final_val_loss,
                result.best_val_loss,
                result.training_time,
                result.convergence_epoch,
                result.model_size,
                result.error_message,
                json.dumps(result.to_dict(), default=str)
            ))
            
            conn.commit()
        
        return True
    
    def _store_pickle(self, result: ExperimentResults) -> bool:
        """Store experiment result as pickle."""
        exp_dir = self.storage_dir / result.experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        result_path = exp_dir / "result.pkl"
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        
        return True
    
    def load_experiment(self, experiment_id: str) -> Optional[ExperimentResults]:
        """
        Load a specific experiment result.
        
        Args:
            experiment_id: ID of the experiment to load
            
        Returns:
            ExperimentResults if found, None otherwise
        """
        try:
            if self.backend == StorageBackend.JSON:
                return self._load_json(experiment_id)
            elif self.backend == StorageBackend.SQLITE:
                return self._load_sqlite(experiment_id)
            elif self.backend == StorageBackend.PICKLE:
                return self._load_pickle(experiment_id)
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
        except Exception as e:
            print(f"Error loading experiment {experiment_id}: {e}")
            return None
    
    def _load_json(self, experiment_id: str) -> Optional[ExperimentResults]:
        """Load experiment result from JSON."""
        result_path = self.storage_dir / experiment_id / "result.json"
        
        if not result_path.exists():
            return None
        
        with open(result_path, 'r') as f:
            data = json.load(f)
        
        return ExperimentResults.from_dict(data)
    
    def _load_sqlite(self, experiment_id: str) -> Optional[ExperimentResults]:
        """Load experiment result from SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT full_data FROM experiments WHERE experiment_id = ?', (experiment_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            data = json.loads(row[0])
            return ExperimentResults.from_dict(data)
    
    def _load_pickle(self, experiment_id: str) -> Optional[ExperimentResults]:
        """Load experiment result from pickle."""
        result_path = self.storage_dir / experiment_id / "result.pkl"
        
        if not result_path.exists():
            return None
        
        with open(result_path, 'rb') as f:
            return pickle.load(f)
    
    def query_experiments(self, 
                         filter_criteria: Optional[QueryFilter] = None,
                         sort_by: str = "timestamp",
                         ascending: bool = False,
                         limit: Optional[int] = None) -> List[ExperimentResults]:
        """
        Query experiments with filtering and sorting.
        
        Args:
            filter_criteria: Filter criteria to apply
            sort_by: Field to sort by
            ascending: Sort order
            limit: Maximum number of results
            
        Returns:
            List of matching ExperimentResults
        """
        if self.backend == StorageBackend.SQLITE:
            return self._query_sqlite(filter_criteria, sort_by, ascending, limit)
        else:
            return self._query_file_based(filter_criteria, sort_by, ascending, limit)
    
    def _query_sqlite(self, 
                     filter_criteria: Optional[QueryFilter],
                     sort_by: str,
                     ascending: bool,
                     limit: Optional[int]) -> List[ExperimentResults]:
        """Query experiments from SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT full_data FROM experiments"
            params = []
            
            if filter_criteria:
                conditions = []
                
                if filter_criteria.polynomial_degrees:
                    placeholders = ','.join('?' * len(filter_criteria.polynomial_degrees))
                    conditions.append(f"polynomial_degree IN ({placeholders})")
                    params.extend(filter_criteria.polynomial_degrees)
                
                if filter_criteria.model_architectures:
                    placeholders = ','.join('?' * len(filter_criteria.model_architectures))
                    conditions.append(f"model_architecture IN ({placeholders})")
                    params.extend(filter_criteria.model_architectures)
                
                if filter_criteria.optimizers:
                    placeholders = ','.join('?' * len(filter_criteria.optimizers))
                    conditions.append(f"optimizer IN ({placeholders})")
                    params.extend(filter_criteria.optimizers)
                
                if filter_criteria.min_final_val_loss is not None:
                    conditions.append("final_val_loss >= ?")
                    params.append(filter_criteria.min_final_val_loss)
                
                if filter_criteria.max_final_val_loss is not None:
                    conditions.append("final_val_loss <= ?")
                    params.append(filter_criteria.max_final_val_loss)
                
                if filter_criteria.start_date:
                    conditions.append("timestamp >= ?")
                    params.append(filter_criteria.start_date.isoformat())
                
                if filter_criteria.end_date:
                    conditions.append("timestamp <= ?")
                    params.append(filter_criteria.end_date.isoformat())
                
                if filter_criteria.statuses:
                    placeholders = ','.join('?' * len(filter_criteria.statuses))
                    conditions.append(f"status IN ({placeholders})")
                    params.extend(filter_criteria.statuses)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            # Add sorting
            order = "ASC" if ascending else "DESC"
            query += f" ORDER BY {sort_by} {order}"
            
            # Add limit
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                data = json.loads(row[0])
                result = ExperimentResults.from_dict(data)
                
                # Apply additional filters that can't be done in SQL
                if filter_criteria and not filter_criteria.matches(result):
                    continue
                
                results.append(result)
            
            return results
    
    def _query_file_based(self, 
                         filter_criteria: Optional[QueryFilter],
                         sort_by: str,
                         ascending: bool,
                         limit: Optional[int]) -> List[ExperimentResults]:
        """Query experiments from file-based storage."""
        results = []
        
        # Load all experiments
        for exp_dir in self.storage_dir.iterdir():
            if exp_dir.is_dir():
                result = self.load_experiment(exp_dir.name)
                if result:
                    results.append(result)
        
        # Apply filters
        if filter_criteria:
            results = [r for r in results if filter_criteria.matches(r)]
        
        # Sort results
        if sort_by == "timestamp":
            results.sort(key=lambda x: x.timestamp, reverse=not ascending)
        elif sort_by == "final_val_loss":
            results.sort(key=lambda x: x.final_val_loss, reverse=not ascending)
        elif sort_by == "duration_seconds":
            results.sort(key=lambda x: x.duration_seconds, reverse=not ascending)
        elif sort_by == "experiment_id":
            results.sort(key=lambda x: x.experiment_id, reverse=not ascending)
        
        # Apply limit
        if limit:
            results = results[:limit]
        
        return results
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all experiments."""
        all_results = self.query_experiments()
        
        if not all_results:
            return {"total_experiments": 0}
        
        # Basic counts
        total = len(all_results)
        completed = sum(1 for r in all_results if r.status == "completed")
        failed = sum(1 for r in all_results if r.status == "failed")
        
        # Performance statistics (only for completed experiments)
        completed_results = [r for r in all_results if r.status == "completed"]
        
        if completed_results:
            val_losses = [r.final_val_loss for r in completed_results]
            train_losses = [r.final_train_loss for r in completed_results]
            durations = [r.duration_seconds for r in completed_results]
            
            summary = {
                "total_experiments": total,
                "completed": completed,
                "failed": failed,
                "success_rate": completed / total if total > 0 else 0,
                "performance": {
                    "mean_val_loss": np.mean(val_losses),
                    "std_val_loss": np.std(val_losses),
                    "min_val_loss": np.min(val_losses),
                    "max_val_loss": np.max(val_losses),
                    "mean_train_loss": np.mean(train_losses),
                    "std_train_loss": np.std(train_losses),
                },
                "timing": {
                    "mean_duration": np.mean(durations),
                    "std_duration": np.std(durations),
                    "min_duration": np.min(durations),
                    "max_duration": np.max(durations),
                },
                "configurations": {
                    "polynomial_degrees": list(set(r.config.polynomial_degree for r in completed_results)),
                    "model_architectures": list(set(r.config.model_architecture for r in completed_results)),
                    "optimizers": list(set(r.config.optimizer for r in completed_results)),
                }
            }
        else:
            summary = {
                "total_experiments": total,
                "completed": completed,
                "failed": failed,
                "success_rate": 0,
                "performance": {},
                "timing": {},
                "configurations": {}
            }
        
        return summary
    
    def get_best_experiments(self, 
                           metric: str = "final_val_loss",
                           n: int = 10,
                           filter_criteria: Optional[QueryFilter] = None) -> List[ExperimentResults]:
        """
        Get the best performing experiments.
        
        Args:
            metric: Metric to optimize ("final_val_loss", "final_train_loss", "best_val_loss")
            n: Number of top experiments to return
            filter_criteria: Optional filter criteria
            
        Returns:
            List of best performing experiments
        """
        results = self.query_experiments(filter_criteria)
        completed_results = [r for r in results if r.status == "completed"]
        
        if metric == "final_val_loss":
            completed_results.sort(key=lambda x: x.final_val_loss)
        elif metric == "final_train_loss":
            completed_results.sort(key=lambda x: x.final_train_loss)
        elif metric == "best_val_loss":
            completed_results.sort(key=lambda x: x.best_val_loss)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        return completed_results[:n]
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment.
        
        Args:
            experiment_id: ID of experiment to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.backend == StorageBackend.SQLITE:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM experiments WHERE experiment_id = ?', (experiment_id,))
                    conn.commit()
            
            # Also delete file-based storage
            exp_dir = self.storage_dir / experiment_id
            if exp_dir.exists():
                import shutil
                shutil.rmtree(exp_dir)
            
            return True
        except Exception as e:
            print(f"Error deleting experiment {experiment_id}: {e}")
            return False
    
    def export_to_csv(self, 
                     filepath: str,
                     filter_criteria: Optional[QueryFilter] = None) -> bool:
        """
        Export experiment results to CSV.
        
        Args:
            filepath: Path to save CSV file
            filter_criteria: Optional filter criteria
            
        Returns:
            True if successful, False otherwise
        """
        try:
            results = self.query_experiments(filter_criteria)
            
            # Convert to DataFrame
            data = []
            for result in results:
                row = {
                    'experiment_id': result.experiment_id,
                    'timestamp': result.timestamp,
                    'duration_seconds': result.duration_seconds,
                    'status': result.status,
                    'polynomial_degree': result.config.polynomial_degree,
                    'noise_level': result.config.noise_level,
                    'model_architecture': result.config.model_architecture,
                    'optimizer': result.config.optimizer,
                    'learning_rate': result.config.learning_rate,
                    'batch_size': result.config.batch_size,
                    'epochs': result.config.epochs,
                    'final_train_loss': result.final_train_loss,
                    'final_val_loss': result.final_val_loss,
                    'best_val_loss': result.best_val_loss,
                    'training_time': result.training_time,
                    'convergence_epoch': result.convergence_epoch,
                    'model_size': result.model_size,
                    'error_message': result.error_message
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            
            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False