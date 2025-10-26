#!/usr/bin/env python3
"""
Multi-GPU Training Orchestrator

Manages 100 experiments (v3-v102) across 4 GPUs with 2 concurrent jobs per GPU
"""
import os
import sys
import json
import subprocess
import time
import signal
from pathlib import Path
from queue import Queue
from threading import Thread, Lock
import argparse
from datetime import datetime


class GPUWorker(Thread):
    """
    Worker thread that runs training jobs on a specific GPU
    """
    def __init__(self, gpu_id, job_queue, results_lock, results, failed_jobs, base_cmd):
        Thread.__init__(self)
        self.gpu_id = gpu_id
        self.job_queue = job_queue
        self.results_lock = results_lock
        self.results = results
        self.failed_jobs = failed_jobs
        self.base_cmd = base_cmd
        self.daemon = True
        self.running = True
        self.current_process = None

    def run(self):
        """Process jobs from queue"""
        while self.running:
            try:
                # Get next job (with timeout to allow shutdown)
                try:
                    job = self.job_queue.get(timeout=1)
                except:
                    continue

                if job is None:  # Poison pill
                    break

                # Run job
                self.run_job(job)

                # Mark job as done
                self.job_queue.task_done()

            except Exception as e:
                print(f"[GPU {self.gpu_id}] Worker error: {e}")
                sys.stdout.flush()

    def run_job(self, job):
        """
        Run a single training job

        Args:
            job: dict with version, techniques, description
        """
        version = job['version']
        techniques = job['techniques']
        description = job['description']

        technique_str = ','.join(map(str, techniques)) if techniques else ''

        print(f"\n[GPU {self.gpu_id}] 🚀 Starting v{version}: {description}")
        print(f"[GPU {self.gpu_id}]    Techniques: {techniques}")
        sys.stdout.flush()

        # Build command
        # Note: When CUDA_VISIBLE_DEVICES is set, always use gpu_id=0
        # because the environment only exposes one GPU
        cmd = self.base_cmd + [
            '--techniques', technique_str,
            '--version', str(version),
            '--gpu_id', '0',  # Always 0 when CUDA_VISIBLE_DEVICES is set
        ]

        # Log file
        log_dir = Path('logs_improvements')
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f'v{version}_gpu{self.gpu_id}.log'

        start_time = time.time()

        try:
            # Run training with CUDA_VISIBLE_DEVICES set to actual GPU
            with open(log_file, 'w') as f:
                self.current_process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(self.gpu_id)}
                )

                # Wait for completion
                return_code = self.current_process.wait()

            elapsed = time.time() - start_time

            if return_code == 0:
                # Success: load results
                results_path = Path('checkpoints_improvements') / f'v{version}' / 'results.json'

                if results_path.exists():
                    with open(results_path, 'r') as f:
                        result = json.load(f)

                    with self.results_lock:
                        self.results[f'v{version}'] = result

                    psnr = result.get('final_psnr', 0.0)
                    print(f"[GPU {self.gpu_id}] ✅ v{version} complete: "
                          f"PSNR={psnr:.2f} dB, time={elapsed/60:.1f}min")
                else:
                    print(f"[GPU {self.gpu_id}] ⚠️  v{version} finished but no results found")
                    with self.results_lock:
                        self.failed_jobs.append({
                            'version': version,
                            'reason': 'No results file',
                            'log': str(log_file)
                        })
            else:
                # Failed
                print(f"[GPU {self.gpu_id}] ❌ v{version} failed (code {return_code})")
                with self.results_lock:
                    self.failed_jobs.append({
                        'version': version,
                        'return_code': return_code,
                        'log': str(log_file)
                    })

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[GPU {self.gpu_id}] ❌ v{version} crashed: {e}")
            with self.results_lock:
                self.failed_jobs.append({
                    'version': version,
                    'error': str(e),
                    'log': str(log_file)
                })

        finally:
            self.current_process = None
            sys.stdout.flush()

    def stop(self):
        """Stop worker and kill current process"""
        self.running = False
        if self.current_process is not None:
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=5)
            except:
                self.current_process.kill()


class Orchestrator:
    """
    Manages multi-GPU training orchestration
    """
    def __init__(self, combinations, num_gpus=4, jobs_per_gpu=2, base_args=None):
        self.combinations = combinations
        self.num_gpus = num_gpus
        self.jobs_per_gpu = jobs_per_gpu
        self.base_args = base_args or {}

        self.job_queue = Queue()
        self.results_lock = Lock()
        self.results = {}
        self.failed_jobs = []
        self.workers = []

        # Statistics
        self.total_jobs = len(combinations)
        self.start_time = None

    def build_base_command(self):
        """Build base training command"""
        script_path = Path(__file__).parent / 'train_improved.py'

        cmd = [
            sys.executable,
            str(script_path),
        ]

        # Add base arguments
        for key, value in self.base_args.items():
            cmd.extend([f'--{key}', str(value)])

        return cmd

    def populate_queue(self):
        """Add all jobs to queue"""
        for combo in self.combinations:
            self.job_queue.put(combo)

        # Add poison pills for workers
        for _ in range(self.num_gpus * self.jobs_per_gpu):
            self.job_queue.put(None)

    def start_workers(self):
        """Start worker threads"""
        base_cmd = self.build_base_command()

        for gpu_id in range(self.num_gpus):
            for slot in range(self.jobs_per_gpu):
                worker = GPUWorker(
                    gpu_id, self.job_queue, self.results_lock,
                    self.results, self.failed_jobs, base_cmd
                )
                worker.start()
                self.workers.append(worker)

        print(f"\n🚀 Started {len(self.workers)} workers across {self.num_gpus} GPUs")
        print(f"   Total jobs: {self.total_jobs}")
        print(f"   Parallel slots: {len(self.workers)}\n")
        sys.stdout.flush()

    def monitor_progress(self):
        """Monitor and report progress"""
        while True:
            time.sleep(30)  # Report every 30 seconds

            with self.results_lock:
                completed = len(self.results)
                failed = len(self.failed_jobs)

            remaining = self.total_jobs - completed - failed
            progress = (completed + failed) / self.total_jobs * 100

            elapsed = time.time() - self.start_time
            if completed > 0:
                avg_time = elapsed / completed
                eta = avg_time * remaining
            else:
                eta = 0

            print(f"\n📊 Progress: {completed + failed}/{self.total_jobs} ({progress:.1f}%)")
            print(f"   ✅ Completed: {completed}")
            print(f"   ❌ Failed: {failed}")
            print(f"   ⏳ Remaining: {remaining}")
            print(f"   ⏱️  ETA: {eta/60:.1f} minutes\n")
            sys.stdout.flush()

            if remaining == 0:
                break

    def run(self):
        """Run orchestration"""
        print(f"\n{'='*70}")
        print(f"Multi-GPU Training Orchestrator")
        print(f"{'='*70}\n")

        self.start_time = time.time()

        # Populate queue
        self.populate_queue()

        # Start workers
        self.start_workers()

        # Monitor progress
        monitor_thread = Thread(target=self.monitor_progress, daemon=True)
        monitor_thread.start()

        # Wait for completion
        self.job_queue.join()

        # Stop workers
        for worker in self.workers:
            worker.stop()

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)

        elapsed = time.time() - self.start_time

        print(f"\n{'='*70}")
        print(f"Orchestration Complete")
        print(f"{'='*70}")
        print(f"Total time: {elapsed/3600:.1f} hours")
        print(f"Completed: {len(self.results)}/{self.total_jobs}")
        print(f"Failed: {len(self.failed_jobs)}/{self.total_jobs}")
        print(f"{'='*70}\n")

        # Save results
        self.save_results()

    def save_results(self):
        """Save final results"""
        output_dir = Path('results_improvements')
        output_dir.mkdir(exist_ok=True)

        # All results
        results_path = output_dir / 'all_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"💾 Saved all results to: {results_path}")

        # Failed jobs
        if self.failed_jobs:
            failed_path = output_dir / 'failed_jobs.json'
            with open(failed_path, 'w') as f:
                json.dump(self.failed_jobs, f, indent=2)

            print(f"⚠️  Saved failed jobs to: {failed_path}")

        # Performance matrix (CSV)
        self.save_performance_matrix()

        # Technique ranking
        self.save_technique_ranking()

    def save_performance_matrix(self):
        """Create CSV performance matrix"""
        import csv

        output_path = Path('results_improvements') / 'performance_matrix.csv'

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Version', 'Techniques', 'Best_PSNR', 'Final_PSNR',
                'Training_Time', 'Complexity'
            ])

            # Sort by version
            sorted_versions = sorted(self.results.keys(), key=lambda x: int(x[1:]))

            for version in sorted_versions:
                result = self.results[version]
                writer.writerow([
                    version,
                    ' '.join(map(str, result.get('techniques', []))),
                    f"{result.get('best_psnr', 0):.2f}",
                    f"{result.get('final_psnr', 0):.2f}",
                    f"{result.get('training_time', 0)/60:.1f}",
                    result.get('complexity', 'simple'),
                ])

        print(f"📊 Saved performance matrix to: {output_path}")

    def save_technique_ranking(self):
        """Analyze and rank techniques by impact"""
        from collections import defaultdict

        # Baseline (v2) PSNR - assume ~35 dB for simple
        baseline_psnr = 35.0

        technique_impacts = defaultdict(list)

        # Analyze each result
        for version, result in self.results.items():
            techniques = result.get('techniques', [])
            psnr = result.get('final_psnr', 0)
            improvement = psnr - baseline_psnr

            # Single technique analysis
            if len(techniques) == 1:
                tid = techniques[0]
                technique_impacts[tid].append(improvement)

        # Compute average impact
        technique_rankings = []
        for tid in range(1, 11):
            if tid in technique_impacts:
                impacts = technique_impacts[tid]
                avg_impact = sum(impacts) / len(impacts)
                technique_rankings.append({
                    'technique_id': tid,
                    'technique_name': f"Technique {tid}",
                    'avg_improvement_db': avg_impact,
                    'num_samples': len(impacts),
                })

        # Sort by impact
        technique_rankings.sort(key=lambda x: x['avg_improvement_db'], reverse=True)

        # Save
        output_path = Path('results_improvements') / 'technique_ranking.json'
        with open(output_path, 'w') as f:
            json.dump(technique_rankings, f, indent=2)

        print(f"🏆 Saved technique ranking to: {output_path}")

        # Print top 5
        print(f"\n🏆 Top 5 Techniques:")
        for i, tech in enumerate(technique_rankings[:5], 1):
            print(f"   {i}. Technique {tech['technique_id']}: "
                  f"+{tech['avg_improvement_db']:.2f} dB "
                  f"(n={tech['num_samples']})")
        print()


def main():
    parser = argparse.ArgumentParser()

    # Data params
    parser.add_argument('--complexity', type=str, default='simple')
    parser.add_argument('--resolution', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=500)

    # Model params
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dim_feedforward', type=int, default=512)

    # Training params
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)

    # Orchestration
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--jobs_per_gpu', type=int, default=2)
    parser.add_argument('--combinations_file', type=str,
                       default='combinations.json')

    args = parser.parse_args()

    # Load combinations
    script_dir = Path(__file__).parent
    combinations_path = script_dir / args.combinations_file

    if not combinations_path.exists():
        print(f"❌ Combinations file not found: {combinations_path}")
        print(f"   Run generate_combinations.py first")
        sys.exit(1)

    with open(combinations_path, 'r') as f:
        combinations = json.load(f)

    print(f"📋 Loaded {len(combinations)} combinations from {combinations_path}")

    # Base arguments for all training runs
    base_args = {
        'complexity': args.complexity,
        'resolution': args.resolution,
        'num_samples': args.num_samples,
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'dim_feedforward': args.dim_feedforward,
        'epochs': args.epochs,
        'lr': args.lr,
    }

    # Create orchestrator
    orchestrator = Orchestrator(
        combinations,
        num_gpus=args.num_gpus,
        jobs_per_gpu=args.jobs_per_gpu,
        base_args=base_args,
    )

    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n\n🛑 Interrupt received, stopping workers...")
        for worker in orchestrator.workers:
            worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run orchestration
    orchestrator.run()


if __name__ == '__main__':
    main()
