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


ASSIGNED_GPU_PAIR_TOKEN = "__RUN_ON_GPU_PAIR_ASSIGNED_GPUS__"


@dataclass
class SchedulerConfig:
    max_concurrent_jobs: int = 16
    max_jobs_per_gpu_pair: int = 8
    est_mem_per_job_mb: int = 5000
    min_free_mem_mb: int = 1500
    sched_poll_secs: int = 30
    post_launch_wait_secs: int = 300
    fallback_gpus: str = "0,1"


@dataclass
class JobSpec:
    desc: str
    log_file: str
    command: list[str]


@dataclass
class RunningJob:
    spec: JobSpec
    gpu_pair: str
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
        max_jobs_per_gpu_pair=int(raw.get("max_jobs_per_gpu_pair", 8)),
        est_mem_per_job_mb=int(raw.get("est_mem_per_job_mb", 5000)),
        min_free_mem_mb=int(raw.get("min_free_mem_mb", 1500)),
        sched_poll_secs=int(raw.get("sched_poll_secs", 30)),
        post_launch_wait_secs=int(raw.get("post_launch_wait_secs", 300)),
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


def build_gpu_pairs(gpus: list[str]) -> list[tuple[str, str]]:
    if len(gpus) == 0:
        raise RuntimeError("No GPUs available for scheduling.")
    if len(gpus) % 2 != 0:
        raise RuntimeError(
            "GPU count must be an even multiple of 2 so pairs can be formed, got: "
            + ",".join(gpus)
        )

    gpu_pairs: list[tuple[str, str]] = []
    for idx in range(0, len(gpus), 2):
        left = gpus[idx]
        right = gpus[idx + 1]
        try:
            left_idx = int(left)
            right_idx = int(right)
        except ValueError as exc:
            raise RuntimeError(f"GPU ids must be integers, got pair: {left},{right}") from exc
        if right_idx != left_idx + 1:
            raise RuntimeError(
                f"GPU pairs must be adjacent like 0,1 or 2,3, got pair: {left},{right}"
            )
        gpu_pairs.append((left, right))
    return gpu_pairs


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
    script = Path(script_path)
    if script.suffix == ".sh":
        cmd = ["bash", script_path, "--emit-jobs", *script_args]
    elif script.suffix == ".py":
        cmd = [sys.executable, script_path, "--emit-jobs", *script_args]
    else:
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


def pair_free_mems(pair: tuple[str, str]) -> tuple[int, int]:
    return gpu_mem_free_mb(pair[0]), gpu_mem_free_mb(pair[1])


def choose_gpu_pair(
    gpu_pairs: list[tuple[str, str]],
    pair_active_count: dict[str, int],
    max_concurrent_jobs: int,
    max_jobs_per_gpu_pair: int,
    est_mem_per_job_mb: int,
    min_free_mem_mb: int,
    running_jobs: list[RunningJob],
) -> str | None:
    if len(running_jobs) >= max_concurrent_jobs:
        return None

    need_mem = est_mem_per_job_mb + min_free_mem_mb
    best_pair = None
    best_min_free = -1

    for pair in gpu_pairs:
        pair_key = ",".join(pair)
        if pair_active_count.get(pair_key, 0) >= max_jobs_per_gpu_pair:
            continue

        free_left, free_right = pair_free_mems(pair)
        if free_left != -1 and free_left < need_mem:
            continue
        if free_right != -1 and free_right < need_mem:
            continue

        if free_left == -1 or free_right == -1:
            min_free = sys.maxsize
        else:
            min_free = min(free_left, free_right)

        if best_pair is None or min_free > best_min_free:
            best_pair = pair_key
            best_min_free = min_free

    return best_pair


def launch_job(spec: JobSpec, gpu_pair: str) -> RunningJob:
    command = [gpu_pair if token == ASSIGNED_GPU_PAIR_TOKEN else token for token in spec.command]
    log_path = Path(spec.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_pair
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
    return RunningJob(spec=spec, gpu_pair=gpu_pair, process=process, log_handle=log_handle, relay_thread=relay_thread)


def reap_finished_jobs(running_jobs: list[RunningJob], pair_active_count: dict[str, int]) -> tuple[int, int]:
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
        pair_active_count[job.gpu_pair] = pair_active_count.get(job.gpu_pair, 1) - 1
        finished += 1
        if return_code == 0:
            print(f"[DONE] gpus={job.gpu_pair} {job.spec.desc}")
        else:
            failed += 1
            print(f"[FAILED] gpus={job.gpu_pair} {job.spec.desc} (see {job.spec.log_file})", file=sys.stderr)
    running_jobs[:] = still_running
    return finished, failed


def wait_after_launch(
    gpu_pair: str,
    post_launch_wait_secs: int,
    poll_secs: int,
    on_poll,
) -> None:
    sleep_step = max(1, poll_secs)
    remaining = max(0, post_launch_wait_secs)
    if remaining <= 0:
        return
    print(f"[WAIT] gpus={gpu_pair} hold scheduler for {remaining}s before allowing next launch")
    while remaining > 0:
        current_sleep = min(sleep_step, remaining)
        time.sleep(current_sleep)
        remaining -= current_sleep
        on_poll()
    print(f"[WAIT-DONE] gpus={gpu_pair} fixed post-launch wait completed")


def resolve_scheduler_settings(args, config: SchedulerConfig) -> tuple[list[str], int, int, int, int, int, int]:
    gpus_arg = args.gpus if args.gpus else ",".join(detect_gpus(config.fallback_gpus))
    gpus = [gpu.strip() for gpu in gpus_arg.split(",") if gpu.strip()]
    max_concurrent_jobs = int(os.environ.get("MAX_CONCURRENT_JOBS", config.max_concurrent_jobs))
    max_jobs_per_gpu_pair = int(os.environ.get("MAX_JOBS_PER_GPU_PAIR", config.max_jobs_per_gpu_pair))
    est_mem_per_job_mb = int(os.environ.get("EST_MEM_PER_JOB_MB", config.est_mem_per_job_mb))
    min_free_mem_mb = int(os.environ.get("MIN_FREE_MEM_MB", config.min_free_mem_mb))
    sched_poll_secs = int(os.environ.get("SCHED_POLL_SECS", config.sched_poll_secs))
    post_launch_wait_secs = int(os.environ.get("POST_LAUNCH_WAIT_SECS", config.post_launch_wait_secs))
    return (
        gpus,
        max_concurrent_jobs,
        max_jobs_per_gpu_pair,
        est_mem_per_job_mb,
        min_free_mem_mb,
        sched_poll_secs,
        post_launch_wait_secs,
    )


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

    config = load_config(Path(__file__).with_name("run_on_gpu_pairs.yaml"))
    (
        gpus,
        max_concurrent_jobs,
        max_jobs_per_gpu_pair,
        est_mem_per_job_mb,
        min_free_mem_mb,
        sched_poll_secs,
        post_launch_wait_secs,
    ) = resolve_scheduler_settings(args, config)
    gpu_pairs = build_gpu_pairs(gpus)
    jobs = parse_jobs(str(script_path), script_args)

    print(f"Task script: {script_path}")
    print(f"GPU pairs: {' | '.join(','.join(pair) for pair in gpu_pairs)}")
    print(f"Max concurrent jobs: {max_concurrent_jobs} (up to {max_jobs_per_gpu_pair} per GPU pair)")
    print(f"Estimated mem/job per GPU: {est_mem_per_job_mb}MB")
    print(f"Min free mem reserve per GPU: {min_free_mem_mb}MB")
    print(f"Post-launch wait: {post_launch_wait_secs}s")

    pair_active_count = {",".join(pair): 0 for pair in gpu_pairs}
    running_jobs: list[RunningJob] = []
    started_jobs = 0
    finished_jobs = 0
    failed_jobs = 0

    def poll_running_jobs() -> None:
        nonlocal finished_jobs, failed_jobs
        finished_delta, failed_delta = reap_finished_jobs(running_jobs, pair_active_count)
        finished_jobs += finished_delta
        failed_jobs += failed_delta

    for job in jobs:
        while True:
            poll_running_jobs()
            chosen_pair = choose_gpu_pair(
                gpu_pairs,
                pair_active_count,
                max_concurrent_jobs,
                max_jobs_per_gpu_pair,
                est_mem_per_job_mb,
                min_free_mem_mb,
                running_jobs,
            )
            if chosen_pair is not None:
                running_jobs.append(launch_job(job, chosen_pair))
                pair_active_count[chosen_pair] = pair_active_count.get(chosen_pair, 0) + 1
                started_jobs += 1
                print(f"[START] gpus={chosen_pair} {job.desc}")
                wait_after_launch(
                    chosen_pair,
                    post_launch_wait_secs,
                    sched_poll_secs,
                    poll_running_jobs,
                )
                break
            time.sleep(sched_poll_secs)

    while running_jobs:
        finished_delta, failed_delta = reap_finished_jobs(running_jobs, pair_active_count)
        finished_jobs += finished_delta
        failed_jobs += failed_delta
        if running_jobs:
            time.sleep(sched_poll_secs)

    print(
        f"Scheduler finished: started={started_jobs} finished={finished_jobs} failed={failed_jobs}"
    )
    return 0 if failed_jobs == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
