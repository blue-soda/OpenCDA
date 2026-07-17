# -*- coding: utf-8 -*-
"""Factory for SGCP resource allocation algorithms."""

from opencda.core.clustering.algorithms.resource_allocation.mws import MWS
from opencda.core.clustering.algorithms.resource_allocation.naive_ra import NaiveRA
from opencda.core.clustering.algorithms.resource_allocation.pcs import PCS
from opencda.core.clustering.algorithms.resource_allocation.potential_game import (
    PotentialGame,
)
from opencda.core.clustering.algorithms.resource_allocation.random_ra import (
    RandomRA,
)
from opencda.core.clustering.algorithms.resource_allocation.target_aware_potential_game import (
    TargetAwarePotentialGame,
)
from opencda.core.clustering.algorithms.resource_allocation.object_aware_potential_game import (
    ObjectAwarePotentialGame,
)
from opencda.core.clustering.algorithms.resource_allocation.perception_aware_potential_game import (
    PerceptionAwarePotentialGame,
)


RESOURCE_ALLOCATION_ALGORITHMS = {
    'potential_game': PotentialGame,
    'potentialgame': PotentialGame,
    'pg': PotentialGame,
    'target_aware_potential_game': TargetAwarePotentialGame,
    'target_aware_pg': TargetAwarePotentialGame,
    'tapg': TargetAwarePotentialGame,
    'object_aware_potential_game': ObjectAwarePotentialGame,
    'object_aware_pg': ObjectAwarePotentialGame,
    'oapg': ObjectAwarePotentialGame,
    'perception_aware_potential_game': PerceptionAwarePotentialGame,
    'perception_aware_pg': PerceptionAwarePotentialGame,
    'papg': PerceptionAwarePotentialGame,
    'pcs': PCS,
    'mws': MWS,
    'random': RandomRA,
    'random_ra': RandomRA,
    'naive': NaiveRA,
    'naive_ra': NaiveRA,
}


def build_resource_allocator(name, cav_world):
    algorithm_name = (name or 'potential_game').lower()
    if algorithm_name not in RESOURCE_ALLOCATION_ALGORITHMS:
        raise ValueError(
            'Unknown resource allocation algorithm: %s' % name)
    return RESOURCE_ALLOCATION_ALGORITHMS[algorithm_name](cav_world)
