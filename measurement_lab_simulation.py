from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import heapq
import pandas as pd


INPUT_FILE = "mätresultat.xlsx"

RAW_PRIORITY_TO_CLASS = {
    "Förstabit": "fh",
    "Förstabit UH": "fh",
    "Normal": "normal",
}

PRIORITY_ORDER = {
    ("started", "fh"): 1,
    ("started", "normal"): 2,
    ("new", "fh"): 3,
    ("new", "normal"): 4,
}

MACHINE_COLUMNS = [
    "Mätutrustning 1:",
    "Mätutrustning 2:",
    "Mätutrustning 3:",
    "Mätutrustning 4:",
]


@dataclass
class Article:
    article_id: str
    article_number: str
    raw_priority: str
    article_class: str
    arrival_hour: float
    required_machines: List[str]
    measured_machines: List[str]
    current_status: str
    last_queue_entry_hour: float
    total_wait_hours: float = 0.0
    machine_visits: int = 0

    @property
    def remaining_machines(self) -> List[str]:
        return [m for m in self.required_machines if m not in self.measured_machines]

    def needs_machine(self, machine_name: str) -> bool:
        return machine_name in self.remaining_machines

    def current_priority_rank(self) -> int:
        return PRIORITY_ORDER[(self.current_status, self.article_class)]


@dataclass
class MachineConfig:
    name: str
    avg_service_hours: float


@dataclass
class MachineRunRecord:
    article_id: str
    article_number: str
    raw_priority: str
    article_class: str
    machine: str
    queue_status: str
    arrival_hour: float
    queue_entry_hour: float
    service_start_hour: float
    finish_hour: float
    wait_hours: float
    service_hours: float
    remaining_after_run: int


def to_hours_since_start(ts: pd.Timestamp, start_ts: pd.Timestamp) -> float:
    return (ts - start_ts).total_seconds() / 3600.0


def normalize_machine_name(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def convert_maskintid_to_hours(value) -> Optional[float]:
    if pd.isna(value):
        return None

    if isinstance(value, pd.Timedelta):
        return value.total_seconds() / 3600.0

    if hasattr(value, "hour") and hasattr(value, "minute") and hasattr(value, "second"):
        return value.hour + value.minute / 60.0 + value.second / 3600.0

    try:
        td = pd.to_timedelta(value)
        return td.total_seconds() / 3600.0
    except Exception:
        return None


def parse_input_file(path: str) -> Tuple[List[Article], Dict[str, MachineConfig]]:
    df = pd.read_excel(path)

    df = df[df["Prioritet:"].isin(RAW_PRIORITY_TO_CLASS.keys())].copy()

    df["arrival_ts"] = pd.to_datetime(
        df["Inlämn.datum:"].astype(str) + " " + df["Inlämn.tid:"].astype(str),
        errors="coerce",
    )
    df = df.dropna(subset=["arrival_ts"])

    df["service_hours"] = df["Maskintid"].apply(convert_maskintid_to_hours)

    start_ts = pd.to_datetime(
        df["Startdatum:"].astype(str) + " " + df["Starttid:"].astype(str),
        errors="coerce",
    )
    end_ts = pd.to_datetime(
        df["Slutdatum:"].astype(str) + " " + df["Sluttid:"].astype(str),
        errors="coerce",
    )

    fallback_service = (end_ts - start_ts).dt.total_seconds() / 3600.0
    df["service_hours"] = df["service_hours"].fillna(fallback_service)

    df["article_class"] = df["Prioritet:"].map(RAW_PRIORITY_TO_CLASS)

    df["machine_list"] = df[MACHINE_COLUMNS].apply(
        lambda row: [m for m in (normalize_machine_name(v) for v in row) if m is not None],
        axis=1,
    )

    df = df[df["machine_list"].map(len) > 0].copy()

    start_of_horizon = df["arrival_ts"].min()
    df["arrival_hour"] = df["arrival_ts"].apply(lambda x: to_hours_since_start(x, start_of_horizon))

    machine_service: Dict[str, List[float]] = {}
    for _, row in df.iterrows():
        if pd.notna(row["service_hours"]) and row["service_hours"] > 0:
            first_machine = row["machine_list"][0]
            machine_service.setdefault(first_machine, []).append(float(row["service_hours"]))

    machine_configs: Dict[str, MachineConfig] = {}
    all_machines = sorted({m for machines in df["machine_list"] for m in machines})

    overall_avg = df["service_hours"].dropna()
    default_avg = float(overall_avg.mean()) if not overall_avg.empty else 0.25

    for machine in all_machines:
        values = machine_service.get(machine, [])
        avg_service = float(pd.Series(values).mean()) if values else default_avg
        machine_configs[machine] = MachineConfig(name=machine, avg_service_hours=max(avg_service, 0.01))

    articles: List[Article] = []
    for _, row in df.iterrows():
        articles.append(
            Article(
                article_id=str(row["Löpnr:"]),
                article_number=str(row["Artikelnr:"]),
                raw_priority=str(row["Prioritet:"]),
                article_class=str(row["article_class"]),
                arrival_hour=float(row["arrival_hour"]),
                required_machines=list(dict.fromkeys(row["machine_list"])),
                measured_machines=[],
                current_status="new",
                last_queue_entry_hour=float(row["arrival_hour"]),
            )
        )

    return articles, machine_configs


class FlexibleMeasurementLabSimulation:
    def __init__(self, articles: List[Article], machine_configs: Dict[str, MachineConfig]):
        self.articles: Dict[str, Article] = {a.article_id: a for a in articles}
        self.machine_configs = machine_configs
        self.records: List[MachineRunRecord] = []
        self.machine_busy_until: Dict[str, float] = {name: 0.0 for name in machine_configs}
        self.machine_busy_hours: Dict[str, float] = {name: 0.0 for name in machine_configs}

    def _all_done(self) -> bool:
        return all(article.current_status == "done" for article in self.articles.values())

    def _available_articles_for_machine(self, machine_name: str, current_time: float) -> List[Article]:
        candidates: List[Article] = []
        for article in self.articles.values():
            if article.current_status == "done":
                continue
            if article.arrival_hour > current_time and article.current_status == "new":
                continue
            if article.needs_machine(machine_name):
                candidates.append(article)
        return candidates

    def _select_article_for_machine(self, machine_name: str, current_time: float) -> Optional[Article]:
        candidates = self._available_articles_for_machine(machine_name, current_time)
        if not candidates:
            return None

        candidates.sort(
            key=lambda a: (
                a.current_priority_rank(),
                a.last_queue_entry_hour,
                a.arrival_hour,
                a.article_id,
            )
        )
        return candidates[0]

    def _next_article_arrival_after(self, current_time: float) -> Optional[float]:
        future_times = [
            article.arrival_hour
            for article in self.articles.values()
            if article.current_status == "new" and article.arrival_hour > current_time
        ]
        return min(future_times) if future_times else None

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        current_time = min(article.arrival_hour for article in self.articles.values())

        while not self._all_done():
            progress_made = False

            for machine_name, cfg in self.machine_configs.items():
                if self.machine_busy_until[machine_name] > current_time:
                    continue

                article = self._select_article_for_machine(machine_name, current_time)
                if article is None:
                    continue

                queue_status = article.current_status
                service_start = current_time
                wait_hours = service_start - article.last_queue_entry_hour
                service_hours = cfg.avg_service_hours
                finish_hour = service_start + service_hours

                article.total_wait_hours += max(wait_hours, 0.0)
                article.machine_visits += 1
                article.measured_machines.append(machine_name)

                remaining_after = len(article.remaining_machines)
                if remaining_after == 0:
                    article.current_status = "done"
                else:
                    article.current_status = "started"
                    article.last_queue_entry_hour = finish_hour

                self.machine_busy_until[machine_name] = finish_hour
                self.machine_busy_hours[machine_name] += service_hours

                self.records.append(
                    MachineRunRecord(
                        article_id=article.article_id,
                        article_number=article.article_number,
                        raw_priority=article.raw_priority,
                        article_class=article.article_class,
                        machine=machine_name,
                        queue_status=queue_status,
                        arrival_hour=article.arrival_hour,
                        queue_entry_hour=service_start - wait_hours,
                        service_start_hour=service_start,
                        finish_hour=finish_hour,
                        wait_hours=max(wait_hours, 0.0),
                        service_hours=service_hours,
                        remaining_after_run=remaining_after,
                    )
                )
                progress_made = True

            if progress_made:
                future_finish = [t for t in self.machine_busy_until.values() if t > current_time]
                next_finish = min(future_finish) if future_finish else None
                next_arrival = self._next_article_arrival_after(current_time)
                candidates = [t for t in [next_finish, next_arrival] if t is not None]
                if candidates:
                    current_time = min(candidates)
                continue

            future_finish = [t for t in self.machine_busy_until.values() if t > current_time]
            next_finish = min(future_finish) if future_finish else None
            next_arrival = self._next_article_arrival_after(current_time)
            candidates = [t for t in [next_finish, next_arrival] if t is not None]
            if not candidates:
                break
            current_time = min(candidates)

        runs_df = pd.DataFrame(asdict(r) for r in self.records)

        if runs_df.empty:
            return runs_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        horizon_hours = runs_df["finish_hour"].max()

        machine_summary_df = (
            runs_df.groupby("machine")
            .agg(
                machine_runs=("article_id", "count"),
                unique_articles=("article_id", pd.Series.nunique),
                avg_wait_hours=("wait_hours", "mean"),
                max_wait_hours=("wait_hours", "max"),
                avg_service_hours=("service_hours", "mean"),
                last_finish_hour=("finish_hour", "max"),
            )
            .reset_index()
            .round(3)
        )

        machine_summary_df["simulated_utilization_pct"] = machine_summary_df["machine"].map(
            lambda m: round(100.0 * self.machine_busy_hours[m] / horizon_hours, 2) if horizon_hours > 0 else 0.0
        )
        machine_summary_df = machine_summary_df.sort_values(
            ["simulated_utilization_pct", "avg_wait_hours"], ascending=[False, False]
        )

        article_summary_rows = []
        for article in self.articles.values():
            article_runs = runs_df[runs_df["article_id"] == article.article_id].sort_values("service_start_hour")
            if article_runs.empty:
                continue
            article_summary_rows.append(
                {
                    "article_id": article.article_id,
                    "article_number": article.article_number,
                    "raw_priority": article.raw_priority,
                    "article_class": article.article_class,
                    "required_machine_count": len(article.required_machines),
                    "completed_machine_count": len(article.measured_machines),
                    "arrival_hour": round(article.arrival_hour, 3),
                    "first_start_hour": round(article_runs["service_start_hour"].min(), 3),
                    "finish_hour": round(article_runs["finish_hour"].max(), 3),
                    "total_wait_hours": round(article.total_wait_hours, 3),
                    "total_lead_time_hours": round(article_runs["finish_hour"].max() - article.arrival_hour, 3),
                    "machine_route_used": " -> ".join(article_runs["machine"].tolist()),
                }
            )

        article_summary_df = pd.DataFrame(article_summary_rows).sort_values("finish_hour")

        priority_summary_df = (
            article_summary_df.groupby(["article_class", "raw_priority"])
            .agg(
                articles=("article_id", "count"),
                avg_total_wait_hours=("total_wait_hours", "mean"),
                max_total_wait_hours=("total_wait_hours", "max"),
                avg_lead_time_hours=("total_lead_time_hours", "mean"),
                avg_required_machine_count=("required_machine_count", "mean"),
            )
            .reset_index()
            .round(3)
            .sort_values(["article_class", "avg_total_wait_hours"], ascending=[True, False])
        )

        return runs_df, machine_summary_df, article_summary_df, priority_summary_df


def save_outputs(
    runs_df: pd.DataFrame,
    machine_summary_df: pd.DataFrame,
    article_summary_df: pd.DataFrame,
    priority_summary_df: pd.DataFrame,
    output_dir: str = "simulation_output",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    runs_df.to_csv(output_path / "machine_runs.csv", index=False)
    machine_summary_df.to_csv(output_path / "machine_summary.csv", index=False)
    article_summary_df.to_csv(output_path / "article_summary.csv", index=False)
    priority_summary_df.to_csv(output_path / "priority_summary.csv", index=False)


def print_report(
    machine_summary_df: pd.DataFrame,
    article_summary_df: pd.DataFrame,
    priority_summary_df: pd.DataFrame,
) -> None:
    print("\n=== MASKINSAMMANFATTNING ===")
    print(machine_summary_df.to_string(index=False))

    print("\n=== PRIORITETSSAMMANFATTNING PER ARTIKELKLASS ===")
    print(priority_summary_df.to_string(index=False))

    print("\n=== TOPP 10 ARTIKLAR MED LANGST TOTAL VANTETID ===")
    top_articles = article_summary_df.nlargest(10, "total_wait_hours")[
        [
            "article_id",
            "raw_priority",
            "required_machine_count",
            "total_wait_hours",
            "total_lead_time_hours",
            "machine_route_used",
        ]
    ]
    print(top_articles.to_string(index=False))


def main() -> None:
    articles, machine_configs = parse_input_file(INPUT_FILE)
    sim = FlexibleMeasurementLabSimulation(articles, machine_configs)
    runs_df, machine_summary_df, article_summary_df, priority_summary_df = sim.run()
    save_outputs(runs_df, machine_summary_df, article_summary_df, priority_summary_df)
    print_report(machine_summary_df, article_summary_df, priority_summary_df)


if __name__ == "__main__":
    main()