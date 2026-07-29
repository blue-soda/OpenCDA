# -*- coding: utf-8 -*-
"""Dynamic C/V scheduler for dense SGCP probes.

This variant keeps the formal SGCP two-stage structure but updates receiver
evidence after every accepted upload:

* Stage 1 coverage: C(i,g|r,A) = min(q_i(g), 1 - Q_r^A(g)).
* Stage 2 view refinement: V(i,g|r,A) = min(q_i(g), 1 - Q_r^A(g))
  if Q_r^A(g) > 0 else 0.

Both stages use the same residual-density gain as the density-saturated upload
path, so the score of a sender-grid action matches the remaining amount that
can actually be transmitted before the receiver grid reaches ``rho_th``.
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
        return min(member_quality, max(0.0, 1.0 - current_quality))
