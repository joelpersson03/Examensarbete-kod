from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class MachineConfig:
    name: str
    measurements: int
    total_measure_time_hours: float
    avg_measure_time_minutes: float
    available_hours: float
    historical_utilization_pct: float


@dataclass
class JobResult:
    job_id: int
    machine: str
    arrival_hour: float
    service_start_hour: float
    finish_hour: float
    wait_hours: float
    service_hours: float
    total_time_hours: float


class SingleMachineQueue:
    def __init__(self, config: MachineConfig):
        self.config = config
        self.next_free_hour = 0.0
        self.jobs: List[JobResult] = []

    def process_job(self, job_id: int, arrival_hour: float, service_hours: float) -> JobResult:
        service_start = max(arrival_hour, self.next_free_hour)
        finish = service_start + service_hours
        self.next_free_hour = finish
        result = JobResult(
            job_id=job_id,
            machine=self.config.name,
            arrival_hour=arrival_hour,
            service_start_hour=service_start,
            finish_hour=finish,
            wait_hours=service_start - arrival_hour,
            service_hours=service_hours,
            total_time_hours=finish - arrival_hour,
        )
        self.jobs.append(result)
        return result

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
    def __init__(self, machine_configs: List[MachineConfig], seed: int = 42):
        self.machine_configs = machine_configs
        self.random = random.Random(seed)
        self.queues = {cfg.name: SingleMachineQueue(cfg) for cfg in machine_configs}

    def _generate_arrivals(self, horizon_hours: float, count: int) -> List[float]:
        arrivals = [self.random.uniform(0, horizon_hours) for _ in range(count)]
        arrivals.sort()
        return arrivals

    def _sample_service_time_hours(self, avg_minutes: float) -> float:
        mean_hours = max(avg_minutes / 60.0, 0.01)
        return max(self.random.triangular(mean_hours * 0.5, mean_hours * 1.8, mean_hours), 0.01)

    def run(self, horizon_days: float = 90.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        horizon_hours = horizon_days * 22.0
        results: List[JobResult] = []
        global_job_id = 1

        for cfg in self.machine_configs:
            arrivals = self._generate_arrivals(horizon_hours=horizon_hours, count=cfg.measurements)
            for arrival_hour in arrivals:
                service_hours = self._sample_service_time_hours(cfg.avg_measure_time_minutes)
                result = self.queues[cfg.name].process_job(
                    job_id=global_job_id,
                    arrival_hour=arrival_hour,
                    service_hours=service_hours,
                )
                results.append(result)
                global_job_id += 1

        jobs_df = pd.DataFrame(asdict(r) for r in results)
        summary_df = pd.DataFrame(
            queue.summary(horizon_hours=horizon_hours) for queue in self.queues.values()
        ).sort_values("simulated_utilization_pct", ascending=False)
        return jobs_df, summary_df


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
        MachineConfig("Ytmätare", 146, 46.283, 19.01, 1980.0, 39.95),
        MachineConfig("Zeiss 1", 1330, 504.167, 22.75, 1980.0, 64.84),
        MachineConfig("Zeiss 12", 17, 13.717, 48.42, 1980.0, 0.94),
        MachineConfig("Zeiss 14", 272, 475.850, 104.83, 1980.0, 38.88),
        MachineConfig("Zeiss 15", 367, 264.483, 43.24, 1980.0, 24.35),
    ]


def save_outputs(jobs_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str = "simulation_output") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    jobs_df.to_csv(output_path / "jobs.csv", index=False)
    summary_df.to_csv(output_path / "summary.csv", index=False)


def main() -> None:
    configs = default_machine_configs()
    sim = MeasurementLabSimulation(configs, seed=42)
    jobs_df, summary_df = sim.run(horizon_days=90.0)
    save_outputs(jobs_df, summary_df)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
