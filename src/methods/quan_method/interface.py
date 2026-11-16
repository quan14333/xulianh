"""Integration slot only. No algorithm is attributed to Quân without a spec."""
from dataclasses import dataclass

from ..fusion_interface import WaitingForMethodSpec


@dataclass(frozen=True)
class QuanMethodSpec:
    hypothesis: str
    algorithm: str
    configuration: str
    prototype: str
    ablation_protocol: str
    designed_by: str = "Quân"

    def missing_fields(self):
        fields = ("hypothesis", "algorithm", "configuration", "prototype", "ablation_protocol")
        return [name for name in fields if not getattr(self, name).strip()
                or "PLACEHOLDER" in getattr(self, name).upper()]

    def validate(self):
        missing = self.missing_fields()
        if missing:
            raise WaitingForMethodSpec("Missing Quân spec fields: " + ", ".join(missing))


class QuanMethodPending:
    status = "WAITING_FOR_METHOD_SPEC"

    def __init__(self, *args, **kwargs):
        raise WaitingForMethodSpec(
            "No reviewed Quân hypothesis/algorithm/config/prototype/ablation has been supplied. "
            "No fallback to SAGE or another architecture is allowed.")
