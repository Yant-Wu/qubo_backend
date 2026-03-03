"""CQM 採樣器包裝（單一職責：提供 sample_cqm 介面給 dimod SimulatedAnnealingSampler）。"""
from __future__ import annotations

import dimod
from dimod import ConstrainedQuadraticModel, SampleSet
from dwave.samplers import SimulatedAnnealingSampler


class SimulatedAnnealingCQMSampler:
    """
    將 SimulatedAnnealingSampler 包裝成提供 sample_cqm(cqm) 的介面。
    本機執行，不連線 D-Wave 雲端服務。
    """

    def __init__(self) -> None:
        self._sampler = SimulatedAnnealingSampler()

    def sample_cqm(
        self,
        cqm: ConstrainedQuadraticModel,
        num_reads: int = 1000,
        seed: int | None = None,
    ) -> SampleSet:
        """
        將 CQM 轉換為 BQM 後使用模擬退火求解，回傳 SampleSet。
        """
        bqm, _ = dimod.cqm_to_bqm(cqm)
        return self._sampler.sample(bqm, num_reads=num_reads, seed=seed)
