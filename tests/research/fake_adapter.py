from __future__ import annotations

from research.models import (
    Intent,
    PreflightResult,
    ProbeRequest,
    ProgressUpdate,
    TrialCommand,
    TrialContext,
    TrialReport,
)


class FakeAdapter:
    name = "fake"

    def generate_probe_intents(self, request: ProbeRequest) -> list[Intent]:
        return [
            Intent(
                adapter=self.name,
                model=request.model,
                profile=request.profile,
                phase="probe",
                name="small",
                config={"batch_size": 1},
                objective=request.objective,
                source="probe",
            )
        ]

    def preflight(self, intent: Intent, context: TrialContext) -> PreflightResult:
        return PreflightResult(ok=True, checks={"fake": "ok"}, message="ok")

    def build_trial(self, intent: Intent, context: TrialContext) -> TrialCommand:
        return TrialCommand(argv=["python", "-c", "print('metric=1.0')"], env={})

    def parse_progress(self, event_or_log_line: str) -> ProgressUpdate | None:
        if "metric=" not in event_or_log_line:
            return None
        return ProgressUpdate(
            step=1,
            metrics={"metric": 1.0},
            message=event_or_log_line.strip(),
        )

    def analyze_result(self, context: TrialContext) -> TrialReport:
        return TrialReport(
            status="succeeded",
            metrics={"metric": 1.0},
            failure={},
            summary="fake trial succeeded",
        )
