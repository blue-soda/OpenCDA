# -*- coding: utf-8 -*-
"""Dynamic C/V scheduler for dense SGCP probes.

This variant keeps the formal SGCP two-stage structure but updates receiver
evidence after every accepted upload:

* Stage 1 coverage: C(i,g|r,A) = q_i(g) * (1 - Q_r^A(g)).
* Stage 2 view refinement: V(i,g|r,A) = q_i(g) if Q_r^A(g) > 0 else 0.

It is isolated from the paper-facing ``cov_potential_game`` so we can test
whether dense point clouds benefit from accounting for already-collected grid
evidence during scheduling.
"""

from opencda.core.clustering.algorithms.resource_allocation.dynamic_marginal_two_stage_potential_game import (
    DynamicMarginalTwoStagePotentialGame,
)


class DynamicCVPotentialGame(DynamicMarginalTwoStagePotentialGame):
    """Two-stage C/V scheduler with dynamic receiver-side evidence."""

    def __init__(self, cav_world):
        super(DynamicCVPotentialGame, self).__init__(cav_world)
        self.grid_score_mode = 'dynamic_cv'

    def _grid_view_score(self, head_id, member_id, grid_id):
        member_quality = self._member_quality(head_id, member_id, grid_id)
        if member_quality <= 0.0:
            return 0.0
        current_quality = self._current_evidence(head_id, grid_id)
        if current_quality <= 0.0:
            return 0.0
        return member_quality
