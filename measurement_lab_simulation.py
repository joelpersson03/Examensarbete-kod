from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


PRIORITY_MAP = {
    "forstabit": 1,
    "frekvensbit": 2,
    "normalbit": 3,
    "underhall": 4,
}

JOB_TYPE_SHARES = {
    "forstabit": 0.10,
    "frekvensbit": 0.20,
    "normalbit": 0.60,
    "underhall": 0.10,
}

JOB_TYPE_TIME_FACTORS = {
    "forstabit": 1.15,
    "frekvensbit": 1.00,
    "normalbit": 1.00,
    "underhall": 1.35,
}


@dataclass
class MachineConfig:
    name: str
    measurements: int
    total_measure_time_hours: float
    avg_measure_time_minutes: float
    available_hours: float
    historical_utilization_pct: float


@dataclass
class Job:
    job_id: int
    machine: str
    job_type: str
    priority: int
    arrival_hour: float
    service_hours: float


@dataclass
class JobResult:
    job_id: int
    machine: str
    job_type: str
    priority: int
    arrival_hour: float
    service_start_hour: float
    finish_hour: float
    wait_hours: float
    service_hours: float
    total_time_hours: float


class PriorityMachineQueue:
    def __init__(self, config: MachineConfig):
        self.config = config
        self.jobs: List[JobResult] = []
        self._queue: List[Tuple[float, int, int, Job]] = []
        self._busy_until: float = 0.0
        self._sequence: int = 0

    def add_job(self, job: Job) -> None:
        self._sequence += 1
        heapq.heappush(self._queue, (job.arrival_hour, job.priority, self._sequence, job))

    def run_until_empty(self) -> None:
        while self._queue:
            arrival_hour, _, _, first_seen_job = heapq.heappop(self._queue)
            current_time = max(self._busy_until, arrival_hour)

            ready_jobs: List[Tuple[int, int, Job]] = []
            ready_jobs.append((first_seen_job.priority, first_seen_job.job_id, first_seen_job))

            while self._queue and self._queue[0][0] <= current_time:
                _, _, _, next_job = heapq.heappop(self._queue)
                ready_jobs.append((next_job.priority, next_job.job_id, next_job))

            ready_jobs.sort(key=lambda x: (x[0], x[1]))
            _, _, selected_job = ready_jobs.pop(0)

            for _, _, postponed_job in ready_jobs:
                self.add_job(postponed_job)

            service_start = max(selected_job.arrival_hour, self._busy_until)
            finish = service_start + selected_job.service_hours
            self._busy_until = finish

            result = JobResult(
                job_id=selected_job.job_id,
                machine=selected_job.machine,
                job_type=selected_job.job_type,
                priority=selected_job.priority,
                arrival_hour=selected_job.arrival_hour,
                service_start_hour=service_start,
                finish_hour=finish,
                wait_hours=service_start - selected_job.arrival_hour,
                service_hours=selected_job.service_hours,
                total_time_hours=finish - selected_job.arrival_hour,
            )
            self.jobs.append(result)

    def summary(self, horizon_hours: float) -> Dict[str, float]:
        busy_hours = sum(job.service_hours for job in self.jobs)
        job_count = len(self.jobs)
        avg_wait = sum(job.wait_hours for job in self.jobs) / job_count if job_count else 0.0
        max_wait = max((job.wait_hours for job in self.jobs), default=0.0)
        avg_total = sum(job.total_time_hours for job in self.jobs) / job_count if job_count else 0.0
        return {
            "machine": self.config.name,
            "jobs_processed": job_count,
            "avg_wait_hours": round(avg_wait, 3),
            "max_wait_hours": round(max_wait, 3),
            "avg_total_time_hours": round(avg_total, 3),
            "simulated_busy_hours": round(busy_hours, 3),
            "simulated_utilization_pct": round((100.0 * busy_hours / horizon_hours) if horizon_hours else 0.0, 2),
            "historical_utilization_pct": self.config.historical_utilization_pct,
        }


class MeasurementLabSimulation:
    def __init__(
        self,
        machine_configs: List[MachineConfig],
        job_type_shares: Optional[Dict[str, float]] = None,
        priority_map: Optional[Dict[str, int]] = None,
        seed: int = 42,
    ):
        self.machine_configs = machine_configs
        self.random = random.Random(seed)
        self.job_type_shares = job_type_shares or JOB_TYPE_SHARES
        self.priority_map = priority_map or PRIORITY_MAP
        self.queues = {cfg.name: PriorityMachineQueue(cfg) for cfg in machine_configs}

    def _generate_arrivals(self, horizon_hours: float, count: int) -> List[float]:
        arrivals = [self.random.uniform(0, horizon_hours) for _ in range(count)]
        arrivals.sort()
        return arrivals

    def _sample_job_type(self) -> str:
        labels = list(self.job_type_shares.keys())
        weights = list(self.job_type_shares.values())
        return self.random.choices(labels, weights=weights, k=1)[0]

    def _sample_service_time_hours(self, avg_minutes: float, job_type: str) -> float:
        mean_hours = max(avg_minutes / 60.0, 0.01)
        factor = JOB_TYPE_TIME_FACTORS.get(job_type, 1.0)
        adjusted_mean = mean_hours * factor
        return max(self.random.triangular(adjusted_mean * 0.5, adjusted_mean * 1.8, adjusted_mean), 0.01)

    def run(self, horizon_days: float = 90.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        horizon_hours = horizon_days * 22.0
        global_job_id = 1

        for cfg in self.machine_configs:
            arrivals = self._generate_arrivals(horizon_hours=horizon_hours, count=cfg.measurements)
            for arrival_hour in arrivals:
                job_type = self._sample_job_type()
                priority = self.priority_map[job_type]
                service_hours = self._sample_service_time_hours(cfg.avg_measure_time_minutes, job_type)
                self.queues[cfg.name].add_job(
                    Job(
                        job_id=global_job_id,
                        machine=cfg.name,
                        job_type=job_type,
                        priority=priority,
                        arrival_hour=arrival_hour,
                        service_hours=service_hours,
                    )
                )
                global_job_id += 1

        results: List[JobResult] = []
        for queue in self.queues.values():
            queue.run_until_empty()
            results.extend(queue.jobs)

        jobs_df = pd.DataFrame(asdict(r) for r in results)

        summary_df = pd.DataFrame(
            queue.summary(horizon_hours=horizon_hours) for queue in self.queues.values()
        ).sort_values(["simulated_utilization_pct", "avg_wait_hours"], ascending=[False, False])

        priority_summary_df = (
            jobs_df.groupby(["machine", "job_type"])
            .agg(
                jobs=("job_id", "count"),
                avg_wait_hours=("wait_hours", "mean"),
                max_wait_hours=("wait_hours", "max"),
                avg_total_time_hours=("total_time_hours", "mean"),
                avg_service_hours=("service_hours", "mean"),
            )
            .reset_index()
            .round(3)
            .sort_values(["machine", "avg_wait_hours"], ascending=[True, False])
        )

        priority_pivot_df = (
            priority_summary_df.pivot(
                index="machine",
                columns="job_type",
                values="avg_wait_hours",
            )
            .round(3)
            .fillna(0)
            .reset_index()
        )

        return jobs_df, summary_df, priority_summary_df, priority_pivot_df


def default_machine_configs() -> List[MachineConfig]:
    return [
        MachineConfig("Konturograf", 44, 31.667, 43.18, 396.0, 8.00),
        MachineConfig("Konturograf 2", 50, 14.550, 17.47, 396.0, 4.19),
        MachineConfig("P40 1", 2607, 925.500, 21.30, 1980.0, 47.66),
        MachineConfig("P65", 711, 714.017, 60.25, 1980.0, 36.52),
        MachineConfig("Wenzel WGT350", 2397, 1028.183, 25.73, 1980.0, 62.93),
        MachineConfig("Wenzel WGT400", 2345, 1024.667, 26.22, 1980.0, 68.06),
        MachineConfig("Wenzel WGT500", 2234, 902.000, 24.24, 1980.0, 56.20),
        MachineConfig("Wenzel WGT500 2", 787, 943.250, 71.92, 1980.0, 56.84),
        MachineConfig("Wenzel WGT600", 2220, 994.883, 26.88, 1980.0, 68.60),
        MachineConfig("Ytmatare", 146, 46.283, 19.01, 1980.0, 39.95),
        MachineConfig("Zeiss 1", 1330, 504.167, 22.75, 1980.0, 64.84),
        MachineConfig("Zeiss 12", 17, 13.717, 48.42, 1980.0, 0.94),
        MachineConfig("Zeiss 14", 272, 475.850, 104.83, 1980.0, 38.88),
        MachineConfig("Zeiss 15", 367, 264.483, 43.24, 1980.0, 24.35),
    ]


def save_outputs(
    jobs_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    priority_summary_df: pd.DataFrame,
    priority_pivot_df: pd.DataFrame,
    output_dir: str = "simulation_output",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    jobs_df.to_csv(output_path / "jobs.csv", index=False)
    summary_df.to_csv(output_path / "summary.csv", index=False)
    priority_summary_df.to_csv(output_path / "priority_summary.csv", index=False)
    priority_pivot_df.to_csv(output_path / "priority_wait_matrix.csv", index=False)


def print_report(
    summary_df: pd.DataFrame,
    priority_summary_df: pd.DataFrame,
    priority_pivot_df: pd.DataFrame,
) -> None:
    print("\n=== OVERGRIPANDE RESULTAT PER MASKIN ===")
    print(summary_df.to_string(index=False))

    print("\n=== VANTETID PER MASKIN OCH JOBBTYP ===")
    print(priority_summary_df.to_string(index=False))

    print("\n=== MATRIS: GENOMSNITTLIG VANTETID (TIMMAR) ===")
    print(priority_pivot_df.to_string(index=False))

    top_wait = summary_df.nlargest(5, "avg_wait_hours")[
        ["machine", "avg_wait_hours", "max_wait_hours", "simulated_utilization_pct"]
    ]
    print("\n=== TOPP 5 MASKINER MED LANGST GENOMSNITTLIG VANTETID ===")
    print(top_wait.to_string(index=False))


def main() -> None:
    configs = default_machine_configs()
    sim = MeasurementLabSimulation(configs, seed=42)
    jobs_df, summary_df, priority_summary_df, priority_pivot_df = sim.run(horizon_days=90.0)
    save_outputs(jobs_df, summary_df, priority_summary_df, priority_pivot_df)
    print_report(summary_df, priority_summary_df, priority_pivot_df)


if __name__ == "__main__":
    main()