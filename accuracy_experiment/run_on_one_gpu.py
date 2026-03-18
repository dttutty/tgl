#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import yaml


ASSIGNED_GPU_TOKEN = "__RUN_ON_ONE_GPU_ASSIGNED_GPU__"


@dataclass
class SchedulerConfig:
    max_concurrent_jobs: int = 16
    est_mem_per_job_mb: int = 5000
    min_free_mem_mb: int = 1500
    sched_poll_secs: int = 30
    fallback_gpus: str = "0,1"


@dataclass
class JobSpec:
    desc: str
    log_file: str
    command: list[str]


@dataclass
class RunningJob:
    spec: JobSpec
    gpu: str
    process: subprocess.Popen[str]
    log_handle: object
    relay_thread: threading.Thread


def load_config(config_path: Path) -> SchedulerConfig:
    if not config_path.is_file():
        return SchedulerConfig()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return SchedulerConfig(
        max_concurrent_jobs=int(raw.get("max_concurrent_jobs", 16)),
        est_mem_per_job_mb=int(raw.get("est_mem_per_job_mb", 5000)),
        min_free_mem_mb=int(raw.get("min_free_mem_mb", 1500)),
        sched_poll_secs=int(raw.get("sched_poll_secs", 30)),
        fallback_gpus=str(raw.get("fallback_gpus", "0,1")),
    )


def detect_gpus(fallback_gpus: str) -> list[str]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        gpu_ids = [line.strip() for line in output.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpu_ids = []
    if gpu_ids:
        return gpu_ids
    return [gpu.strip() for gpu in fallback_gpus.split(",") if gpu.strip()]


def gpu_mem_free_mb(gpu_id: str) -> int:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                gpu_id,
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return -1
    if not output:
        return -1
    total_str, used_str = [part.strip() for part in output.split(",", 1)]
    try:
        return int(total_str) - int(used_str)
    except ValueError:
        return -1


def emit_output(pipe, log_handle) -> None:
    try:
        assert pipe is not None
        for line in pipe:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
            log_handle.flush()
    finally:
        if pipe is not None:
            pipe.close()


def parse_jobs(script_path: str, script_args: list[str]) -> list[JobSpec]:
    cmd = [script_path, "--emit-jobs", *script_args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        raise RuntimeError(f"Job emitter failed with exit code {proc.returncode}: {script_path}")
    jobs: list[JobSpec] = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t", 2)
        if len(parts) != 3:
            raise RuntimeError(f"Invalid job line: {raw_line}")
        desc, log_file, command_escaped = parts
        jobs.append(JobSpec(desc=desc, log_file=log_file, command=shlex.split(command_escaped)))
    return jobs


def choose_gpu(
    gpus: list[str],
    gpu_active_count: dict[str, int],
    max_concurrent_jobs: int,
    est_mem_per_job_mb: int,
    min_free_mem_mb: int,
) -> str | None:
    max_per_gpu = (max_concurrent_jobs + len(gpus) - 1) // len(gpus)
    need_mem = est_mem_per_job_mb + min_free_mem_mb
    best_gpu = None
    best_count = sys.maxsize
    for gpu in gpus:
        active_count = gpu_active_count.get(gpu, 0)
        if active_count >= max_per_gpu or active_count >= best_count:
            continue
        free_mem = gpu_mem_free_mb(gpu)
        if free_mem != -1 and free_mem < need_mem:
            continue
        best_gpu = gpu
        best_count = active_count
    return best_gpu


def launch_job(spec: JobSpec, gpu: str) -> RunningJob:
    command = [gpu if token == ASSIGNED_GPU_TOKEN else token for token in spec.command]
    log_path = Path(spec.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    relay_thread = threading.Thread(target=emit_output, args=(process.stdout, log_handle), daemon=True)
    relay_thread.start()
    return RunningJob(spec=spec, gpu=gpu, process=process, log_handle=log_handle, relay_thread=relay_thread)


def reap_finished_jobs(running_jobs: list[RunningJob], gpu_active_count: dict[str, int]) -> tuple[int, int]:
    still_running: list[RunningJob] = []
    finished = 0
    failed = 0
    for job in running_jobs:
        return_code = job.process.poll()
        if return_code is None:
            still_running.append(job)
            continue
        job.relay_thread.join()
        job.log_handle.close()
        gpu_active_count[job.gpu] = gpu_active_count.get(job.gpu, 1) - 1
        finished += 1
        if return_code == 0:
            print(f"[DONE] gpu={job.gpu} {job.spec.desc}")
        else:
            failed += 1
            print(f"[FAILED] gpu={job.gpu} {job.spec.desc} (see {job.spec.log_file})", file=sys.stderr)
    running_jobs[:] = still_running
    return finished, failed


def resolve_scheduler_settings(args, config: SchedulerConfig) -> tuple[list[str], int, int, int, int]:
    gpus_arg = args.gpus if args.gpus else ",".join(detect_gpus(config.fallback_gpus))
    gpus = [gpu.strip() for gpu in gpus_arg.split(",") if gpu.strip()]
    if not gpus:
        raise RuntimeError("No GPUs available for scheduling.")
    max_concurrent_jobs = int(os.environ.get("MAX_CONCURRENT_JOBS", config.max_concurrent_jobs))
    est_mem_per_job_mb = int(os.environ.get("EST_MEM_PER_JOB_MB", config.est_mem_per_job_mb))
    min_free_mem_mb = int(os.environ.get("MIN_FREE_MEM_MB", config.min_free_mem_mb))
    sched_poll_secs = int(os.environ.get("SCHED_POLL_SECS", config.sched_poll_secs))
    return gpus, max_concurrent_jobs, est_mem_per_job_mb, min_free_mem_mb, sched_poll_secs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True, help="Task script that supports --emit-jobs")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU list; default is auto-detect")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    script_args = args.script_args
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    script_path = Path(args.script)
    if not script_path.is_file():
        raise RuntimeError(f"Script not found: {script_path}")

    config = load_config(Path(__file__).with_name("run_on_one_gpu.yaml"))
    gpus, max_concurrent_jobs, est_mem_per_job_mb, min_free_mem_mb, sched_poll_secs = resolve_scheduler_settings(
        args, config
    )
    jobs = parse_jobs(str(script_path), script_args)

    print(f"Task script: {script_path}")
    print(f"GPUs: {' '.join(gpus)}")
    print(
        f"Max concurrent jobs: {max_concurrent_jobs} "
        f"(up to {(max_concurrent_jobs + len(gpus) - 1) // len(gpus)} per GPU)"
    )
    print(f"Estimated mem/job: {est_mem_per_job_mb}MB")
    print(f"Min free mem reserve: {min_free_mem_mb}MB")

    gpu_active_count = {gpu: 0 for gpu in gpus}
    running_jobs: list[RunningJob] = []
    started_jobs = 0
    finished_jobs = 0
    failed_jobs = 0

    for job in jobs:
        while True:
            finished_delta, failed_delta = reap_finished_jobs(running_jobs, gpu_active_count)
            finished_jobs += finished_delta
            failed_jobs += failed_delta
            if len(running_jobs) < max_concurrent_jobs:
                chosen_gpu = choose_gpu(
                    gpus,
                    gpu_active_count,
                    max_concurrent_jobs,
                    est_mem_per_job_mb,
                    min_free_mem_mb,
                )
                if chosen_gpu is not None:
                    running = launch_job(job, chosen_gpu)
                    running_jobs.append(running)
                    gpu_active_count[chosen_gpu] += 1
                    started_jobs += 1
                    print(
                        f"[LAUNCH] gpu={chosen_gpu} pid={running.process.pid} "
                        f"active={len(running_jobs)}/{max_concurrent_jobs} {job.desc}"
                    )
                    break
            print(
                f"[SCHED] active={len(running_jobs)}/{max_concurrent_jobs}, "
                f"no GPU slot available; waiting {sched_poll_secs}s"
            )
            time.sleep(sched_poll_secs)

    while running_jobs:
        finished_delta, failed_delta = reap_finished_jobs(running_jobs, gpu_active_count)
        finished_jobs += finished_delta
        failed_jobs += failed_delta
        if running_jobs:
            time.sleep(sched_poll_secs)

    print("All runs finished.")
    print(f"Started jobs: {started_jobs}")
    print(f"Finished jobs: {finished_jobs}")
    print(f"Failed jobs: {failed_jobs}")
    return 1 if failed_jobs else 0


if __name__ == "__main__":
    raise SystemExit(main())
