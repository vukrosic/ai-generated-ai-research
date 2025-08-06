"""
Batch experiment execution with queue management and progress monitoring.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import queue
import logging

try:
    from .config import ExperimentConfig
    from .runner import ExperimentRunner, ExperimentResults
except ImportError:
    from config import ExperimentConfig
    from runner import ExperimentRunner, ExperimentResults


@dataclass
class BatchJob:
    """Represents a single job in the batch queue."""
    job_id: str
    config: ExperimentConfig
    config_file: Optional[str] = None
    priority: int = 0
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "queued"  # queued, running, completed, failed
    result: Optional[ExperimentResults] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class BatchProgressMonitor:
    """Thread-safe progress monitoring for batch experiments."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[str, BatchJob] = {}
        self._callbacks: List[Callable[[str, BatchJob], None]] = []
    
    def add_job(self, job: BatchJob):
        """Add a job to monitoring."""
        with self._lock:
            self._jobs[job.job_id] = job
    
    def update_job_status(self, job_id: str, status: str, **kwargs):
        """Update job status and optional fields."""
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.status = status
                
                # Update timestamps
                if status == "running" and job.started_at is None:
                    job.started_at = datetime.now()
                elif status in ["completed", "failed"] and job.completed_at is None:
                    job.completed_at = datetime.now()
                
                # Update other fields
                for key, value in kwargs.items():
                    if hasattr(job, key):
                        setattr(job, key, value)
                
                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(job_id, job)
                    except Exception as e:
                        logging.error(f"Error in progress callback: {e}")
    
    def get_job_status(self, job_id: str) -> Optional[BatchJob]:
        """Get status of a specific job."""
        with self._lock:
            return self._jobs.get(job_id)
    
    def get_all_jobs(self) -> Dict[str, BatchJob]:
        """Get all jobs."""
        with self._lock:
            return self._jobs.copy()
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary statistics."""
        with self._lock:
            summary = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
            for job in self._jobs.values():
                summary[job.status] = summary.get(job.status, 0) + 1
            return summary
    
    def add_callback(self, callback: Callable[[str, BatchJob], None]):
        """Add a progress callback function."""
        with self._lock:
            self._callbacks.append(callback)


class BatchExperimentRunner:
    """
    Enhanced batch experiment runner with queue management and progress monitoring.
    """
    
    def __init__(self, 
                 base_runner: ExperimentRunner,
                 max_concurrent: Optional[int] = None,
                 use_processes: bool = False):
        """
        Initialize batch runner.
        
        Args:
            base_runner: Base ExperimentRunner instance
            max_concurrent: Maximum concurrent experiments (defaults to CPU count)
            use_processes: Whether to use processes instead of threads
        """
        self.base_runner = base_runner
        self.max_concurrent = max_concurrent or 4
        self.use_processes = use_processes
        self.progress_monitor = BatchProgressMonitor()
        self.logger = logging.getLogger(__name__)
        
        # Setup progress callback for console output
        self.progress_monitor.add_callback(self._console_progress_callback)
    
    def _console_progress_callback(self, job_id: str, job: BatchJob):
        """Default console progress callback."""
        if job.status == "running":
            self.logger.info(f"Started job {job_id}: {job.config}")
        elif job.status == "completed":
            duration = (job.completed_at - job.started_at).total_seconds()
            val_loss = job.result.final_val_loss if job.result else "N/A"
            self.logger.info(f"Completed job {job_id} in {duration:.1f}s (val_loss: {val_loss})")
        elif job.status == "failed":
            self.logger.error(f"Failed job {job_id}: {job.error}")
    
    def load_configs_from_directory(self, config_dir: str) -> List[BatchJob]:
        """
        Load experiment configurations from directory.
        
        Args:
            config_dir: Directory containing JSON config files
            
        Returns:
            List of BatchJob instances
        """
        config_path = Path(config_dir)
        if not config_path.is_dir():
            raise ValueError(f"Config directory not found: {config_dir}")
        
        jobs = []
        config_files = list(config_path.glob("*.json"))
        
        if not config_files:
            self.logger.warning(f"No JSON files found in {config_dir}")
            return jobs
        
        for i, config_file in enumerate(config_files):
            try:
                config = ExperimentConfig.from_json(str(config_file))
                job_id = f"batch_{config_file.stem}_{i:03d}"
                
                job = BatchJob(
                    job_id=job_id,
                    config=config,
                    config_file=str(config_file),
                    priority=i
                )
                
                jobs.append(job)
                self.progress_monitor.add_job(job)
                
                self.logger.info(f"Loaded config from {config_file.name}")
                
            except Exception as e:
                self.logger.error(f"Error loading config from {config_file}: {e}")
        
        return jobs
    
    def create_jobs_from_configs(self, configs: List[ExperimentConfig]) -> List[BatchJob]:
        """
        Create batch jobs from a list of configurations.
        
        Args:
            configs: List of ExperimentConfig instances
            
        Returns:
            List of BatchJob instances
        """
        jobs = []
        
        for i, config in enumerate(configs):
            job_id = f"batch_config_{i:03d}"
            
            job = BatchJob(
                job_id=job_id,
                config=config,
                priority=i
            )
            
            jobs.append(job)
            self.progress_monitor.add_job(job)
        
        return jobs
    
    def run_batch_jobs(self, jobs: List[BatchJob]) -> List[ExperimentResults]:
        """
        Execute batch jobs with progress monitoring.
        
        Args:
            jobs: List of BatchJob instances to execute
            
        Returns:
            List of ExperimentResults
        """
        if not jobs:
            self.logger.warning("No jobs to execute")
            return []
        
        self.logger.info(f"Starting batch execution of {len(jobs)} jobs with {self.max_concurrent} workers")
        
        # Sort jobs by priority
        jobs.sort(key=lambda j: j.priority)
        
        results = []
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_concurrent) as executor:
            # Submit all jobs
            future_to_job = {}
            for job in jobs:
                future = executor.submit(self._execute_single_job, job)
                future_to_job[future] = job
            
            # Process completed jobs
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                
                try:
                    result = future.result()
                    self.progress_monitor.update_job_status(
                        job.job_id, 
                        "completed", 
                        result=result
                    )
                    results.append(result)
                    
                except Exception as e:
                    self.progress_monitor.update_job_status(
                        job.job_id, 
                        "failed", 
                        error=str(e)
                    )
                    self.logger.error(f"Job {job.job_id} failed: {e}")
        
        # Print final summary
        summary = self.progress_monitor.get_summary()
        self.logger.info(f"Batch execution completed: {summary}")
        
        return results
    
    def _execute_single_job(self, job: BatchJob) -> ExperimentResults:
        """Execute a single batch job."""
        self.progress_monitor.update_job_status(job.job_id, "running")
        
        try:
            # Run the experiment
            result = self.base_runner.run_single_experiment(
                config=job.config,
                experiment_id=job.job_id
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing job {job.job_id}: {e}")
            raise
    
    def run_from_directory(self, config_dir: str) -> List[ExperimentResults]:
        """
        Convenience method to run batch experiments from a directory.
        
        Args:
            config_dir: Directory containing JSON config files
            
        Returns:
            List of ExperimentResults
        """
        jobs = self.load_configs_from_directory(config_dir)
        return self.run_batch_jobs(jobs)
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get detailed progress summary."""
        summary = self.progress_monitor.get_summary()
        jobs = self.progress_monitor.get_all_jobs()
        
        # Calculate timing statistics
        completed_jobs = [j for j in jobs.values() if j.status == "completed" and j.started_at and j.completed_at]
        
        if completed_jobs:
            durations = [(j.completed_at - j.started_at).total_seconds() for j in completed_jobs]
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
        else:
            avg_duration = min_duration = max_duration = 0
        
        return {
            "status_counts": summary,
            "total_jobs": len(jobs),
            "timing": {
                "average_duration": avg_duration,
                "min_duration": min_duration,
                "max_duration": max_duration
            },
            "jobs": {job_id: {
                "status": job.status,
                "config_file": job.config_file,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "error": job.error
            } for job_id, job in jobs.items()}
        }
    
    def save_progress_report(self, output_path: str):
        """Save detailed progress report to file."""
        report = self.get_progress_summary()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Progress report saved to {output_path}")