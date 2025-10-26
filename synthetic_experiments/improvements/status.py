#!/usr/bin/env python3
"""
Quick Status Dashboard for Running Experiments

Shows real-time progress, top results, and failures
"""
import json
import sys
from pathlib import Path
from datetime import datetime


def load_results():
    """Load current results"""
    results_file = Path('results_improvements/all_results.json')
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return {}


def load_failed_jobs():
    """Load failed jobs"""
    failed_file = Path('results_improvements/failed_jobs.json')
    if failed_file.exists():
        with open(failed_file) as f:
            return json.load(f)
    return []


def count_log_files():
    """Count active log files (running + completed)"""
    log_dir = Path('logs_improvements')
    if not log_dir.exists():
        return 0
    return len(list(log_dir.glob('v*_gpu*.log')))


def main():
    print("=" * 70)
    print("📊 Improvement Experiments Status Dashboard")
    print("=" * 70)
    print()

    # Load data
    results = load_results()
    failed_jobs = load_failed_jobs()
    num_logs = count_log_files()

    # Progress
    total_versions = 100
    completed = len(results)
    failed = len(failed_jobs)
    running_or_queued = total_versions - completed - failed

    print(f"📈 Overall Progress: {completed + failed}/{total_versions} "
          f"({(completed + failed)/total_versions*100:.1f}%)")
    print()
    print(f"  ✅ Completed: {completed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  🔄 Running/Queued: {running_or_queued}")
    print(f"  📝 Log files: {num_logs}")
    print()

    # Performance statistics
    if completed > 0:
        psnrs = [r.get('final_psnr', 0) for r in results.values()]
        times = [r.get('training_time', 0) for r in results.values()]

        baseline_psnr = 35.0  # Assume V2 baseline
        improvements = [p - baseline_psnr for p in psnrs]

        print("=" * 70)
        print("📊 Performance Statistics")
        print("=" * 70)
        print()
        print(f"  Baseline (V2): {baseline_psnr:.2f} dB")
        print(f"  Best so far: {max(psnrs):.2f} dB (+{max(improvements):.2f} dB)")
        print(f"  Mean: {sum(psnrs)/len(psnrs):.2f} dB "
              f"(+{sum(improvements)/len(improvements):.2f} dB)")
        print(f"  Worst: {min(psnrs):.2f} dB (+{min(improvements):.2f} dB)")
        print()
        print(f"  Avg training time: {sum(times)/len(times)/60:.1f} minutes")
        print()

        # Top 10 results
        sorted_results = sorted(results.items(),
                              key=lambda x: x[1].get('final_psnr', 0),
                              reverse=True)

        print("=" * 70)
        print("🏆 Top 10 Results So Far")
        print("=" * 70)
        print()

        for i, (version, result) in enumerate(sorted_results[:10], 1):
            psnr = result.get('final_psnr', 0)
            improvement = psnr - baseline_psnr
            techniques = result.get('techniques', [])
            time_min = result.get('training_time', 0) / 60

            print(f"  {i:2d}. {version:5s}: {psnr:5.2f} dB "
                  f"(+{improvement:5.2f} dB) | "
                  f"T{techniques} | {time_min:.1f}min")
        print()

    else:
        print("⏳ No completed results yet...")
        print("   Experiments are starting or still running")
        print()

    # Failed jobs summary
    if failed > 0:
        print("=" * 70)
        print(f"❌ Failed Jobs ({failed} total)")
        print("=" * 70)
        print()

        for i, job in enumerate(failed_jobs[:10], 1):
            version = job.get('version', '?')
            reason = job.get('reason', job.get('error', 'Unknown error'))
            log_file = job.get('log', 'N/A')

            print(f"  {i:2d}. v{version}: {reason}")
            if i == 1:  # Show log path for first failure
                print(f"      Log: {log_file}")

        if failed > 10:
            print(f"  ... and {failed - 10} more")
        print()
        print("  💡 Check logs: cat results_improvements/failed_jobs.json")
        print()

    # Recommendations
    print("=" * 70)
    print("💡 Quick Commands")
    print("=" * 70)
    print()
    print("  # Monitor overall progress")
    print("  tail -f logs_improvements/orchestrator.log")
    print()
    print("  # Watch specific version")
    print("  tail -f logs_improvements/v42_gpu2.log")
    print()
    print("  # GPU utilization")
    print("  watch -n 1 nvidia-smi")
    print()
    print("  # After completion, generate full analysis")
    print("  python3 analyze_results.py")
    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
