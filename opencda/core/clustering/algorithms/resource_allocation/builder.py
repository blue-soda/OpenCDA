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


RESOURCE_ALLOCATION_ALGORITHMS = {
    'potential_game': PotentialGame,
    'potentialgame': PotentialGame,
    'pg': PotentialGame,
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
