"""
The spread metric can be calculated using the functions in this
sub-module. It assumes very little about the data apart from that
dark matter particles must be 'conserved'.
"""

from transfer import LOGGER
from holder import SimulationData
from transfer.utils.numba import create_numba_hashtable

from unyt import unyt_array
from scipy.spatial import cKDTree
from numpy import int64, float64, empty, arange
from numba import njit
from math import sqrt

from typing import Tuple, Dict, Optional
import numpy as np

def find_closest_neighbours(
    dark_matter_coordinates: unyt_array,
    dark_matter_ids: unyt_array,
    boxsize: unyt_array,
    gas_coordinates: Optional[unyt_array] = None,
    gas_ids: Optional[unyt_array] = None,
) -> Tuple[Dict[int, int]]:
    """
    Finds the closest neighbours in the initial conditions, and returns a
    hashtable between their IDs.

    Parameters
    ----------

    dark_matter_coordinates: unyt_array
        Dark matter co-ordinates. These will have a tree built from them and
        must be ordered in the same way as ``dark_matter_ids``.

    dark_matter_ids: unyt_array
        Unique IDs for dark matter particles.

    boxsize: unyt_array
        Box-size for the simulation volume so that periodic tree searches can
        be used to find neighbours over boundaries.

    gas_coordinates: unyt_array, optional
        Gas co-ordinates, must be ordered in the same way as ``gas_ids``.

    gas_ids: unyt_array, optional
        Unique IDs for gas particles.


    Returns
    -------

    dark_matter_neighbours: numba.typed.Dict
        Dictionary of dark matter neighbours. Takes the particle and links
        to its neighbour so dark_matter_neighbours[id] gives the ID of the
        particle that was its nearest neighbour.

    gas_neighbours: numba.typed.Dict
        Dictionary of gas neighbours. Takes the praticle and links it to its
        nearest dark matter neighbour, so gas_neighbours[id] gives the ID of
        the particle that was its nearest neighbour.


    Notes
    -----

    The returned hashtables are slower than their pythonic cousins
    in regular use. However, in ``@jit``ified functions, they are
    significantly faster.
    """

    boxsize = boxsize.to(dark_matter_coordinates.units)

    assert dark_matter_coordinates.units == gas_coordinates.units

    LOGGER.info("Building dark matter tree for spread metric")
    tree = cKDTree(dark_matter_coordinates.value, boxsize=boxsize.value)
    LOGGER.info("Finished tree build")

    # For dark matter, the cloest particle will be ourself.
    _, closest_indicies = tree.query(x=dark_matter_coordinates.value, k=2, n_jobs=-1)

    dark_matter_neighbours = create_numba_hashtable(
        dark_matter_ids.value, dark_matter_ids.value[closest_indicies[:, 1]]
    )

    # For gas, we can just use closest neighbour
    if gas_coordinates is not None:
        _, closest_indicies = tree.query(x=gas_coordinates.value, k=1, n_jobs=-1)

        gas_neighbours = create_numba_hashtable(
            gas_ids.value, dark_matter_ids.value[closest_indicies]
        )
    else:
        gas_neighbours = None

    return dark_matter_neighbours, gas_neighbours


@njit(parallel=True, fastmath=True)
def find_neighbour_distances(
    neighbours: Dict[int, int],
    particle_coordinates: float64,
    dark_matter_coordinates: float64,
    particle_ids: int64,
    dark_matter_ids: int64,
    boxsize: float64,
):
    """
    Find the nearest neighbour distances (i.e. the spread metric) for
    a set of particles. This is jitted, so ensure all units are equal.
    Currently only works for cubic boxes.

    Parameters
    ----------

    neighbours: dict
        The relevant hashtable returned from :func:`find_closest_neighbours`.

    particle_coordinates: np.array[float64]
        The co-ordinates array for your particles.

    dark_matter_coordinates: np.array[float64]
        The co-oridnates array for dark matter particles. Distances will be
        found between these particles and your particles.

    particle_ids: np.array[int64]
        The IDs of your particles.

    dark_matter_ids: np.array[int64]
        The IDs of the associated dark matter particles. Must be unique.

    boxsize: float64
        Box-size so that periodic wrapping can be used.


    Returns
    -------

    distances: np.array[float64]
        The final state distances between the particles and their initial
        state closest dark matter neighbours.


    Notes
    -----

    In this verison of the code, we use hashtables instead of relying on
    sorts. This means that your particles do not necessarily need to be
    sorted. If they are storted, it may help speed things up, though.
    """

    distances = empty(particle_ids.size, dtype=float64)

    dark_matter_hashtable = create_numba_hashtable(
        dark_matter_ids, arange(dark_matter_ids.size)
    )

    half_boxsize = 0.5 * boxsize

    for particle in range(particle_ids.size):
        id = np.uint64(particle_ids[particle])
        x = particle_coordinates[particle]

        nearest_dm_id = neighbours[id]
        nearest_dm_position = dark_matter_hashtable[nearest_dm_id]
        x_dm = dark_matter_coordinates[nearest_dm_position]

        # This is done explicitly to let the compiler know
        # that we only 'care' about the first three elements
        # as it believes that x, x_dm are generic ndarrays.
        dx = x[0] - x_dm[0]
        dy = x[1] - x_dm[1]
        dz = x[2] - x_dm[2]

        dx -= (dx > half_boxsize) * boxsize
        dx += (dx <= -half_boxsize) * boxsize
        dy -= (dy > half_boxsize) * boxsize
        dy += (dy <= -half_boxsize) * boxsize
        dz -= (dz > half_boxsize) * boxsize
        dz += (dz <= -half_boxsize) * boxsize

        distances[particle] = sqrt(dx * dx + dy * dy + dz * dz)

    return distances


class SpreadMetricCalculator(object):
    """
    Calculates the spread metric and stores the result in various
    attributes.

    Note that this *only* allows for the use of keyword arguments. You
    can choose to instantiate this object by either providing an instance
    of the :class:`transfer.holder.SimulationData` class, or by providing
    the arrays for dark matter and gas positions, along with their IDs,
    manually. For more information, see the documentation for the parameters
    below. Note that it is assumed that everything is done in co-moving space.

    Parameters
    ----------

    simulation: SimulationData, optional, keyword only
        ``SimulationData`` instance describing the initial and final state. The
        parameters in this can be over-ridden, or provided separately, by
        providing the other keyword arguments in this constructor. However,
        a full ``SimulationData`` instance has all of the information required
        for a full spread metric calculation.

    dark_matter_initial_coordinates: unyt_array, optional, keyword only
        Dark matter initial co-ordinates. Not required if ``simulation``
        is passed, as are all of the rest of the parameters here.

    gas_initial_coordinates: unyt_array, optional, keyword only
        Gas initial co-ordinates, with units. Should be dimensionally
        consistent with the dark_matter_initial_coordinates.

    dark_matter_initial_ids: unyt_array, optional, keyword only
        Initial-state IDs for the dark matter particles.

    gas_initial_ids: unyt_array, optional, keyword only
        Initial-state IDs for the gas particles.

    dark_matter_final_coordinates: unyt_array, optional, keyword only
        Final-state dark matter coordinates. Note that the analsysis assumes
        that no dark matter particles are created or destroyed.

    dark_matter_final_ids: unyt_array, optional, keyword only
        Final-state dark matter IDs. Note that the analsysis assumes
        that no dark matter particles are created or destroyed.

    gas_final_coordinates: unyt_array, optional, keyword only
        Final-state gas coordinates.

    gas_final_ids: unyt_array, optional, keyword only
        Final-state gas IDs.

    star_final_coordinates: unyt_array, optional, keyword only
        Final-state star coordinates (if present).

    star_final_ids: unyt_array, optional, keyword only
        Final-state star IDs (if present).

    boxsize: unyt_array, optional, keyword only
        Boxsize (comoving) for the simulation volume so that periodic
        tree wrapping can be used.


    Attributes
    ----------

    dark_matter_spread: unyt_array[float64]
        Dark matter spread metric for the particles specified in the initialiser.

    gas_spread: unyt_array[float64]
        Gas spread metric for the particles specified in the initialiser.

    star_spread: unyt_array[float64]
        Star spread metric for the particles specified in the initialiser.
    """

    simulation: Optional[SimulationData] = None

    dark_matter_initial_coordinates: Optional[unyt_array] = None
    gas_initial_coordinates: Optional[unyt_array] = None
    dark_matter_initial_ids: Optional[unyt_array] = None
    gas_initial_ids: Optional[unyt_array] = None
    dark_matter_final_coordinates: Optional[unyt_array] = None
    dark_matter_final_ids: Optional[unyt_array] = None
    gas_final_coordinates: Optional[unyt_array] = None
    gas_final_ids: Optional[unyt_array] = None
    star_final_coordinates: Optional[unyt_array] = None
    star_final_ids: Optional[unyt_array] = None
    boxsize: Optional[unyt_array] = None

    dark_matter_neighbours: Optional[Dict[int64, int64]] = None
    gas_neighbours: Optional[Dict[int64, int64]] = None

    dark_matter_spread: Optional[unyt_array] = None
    gas_spread: Optional[unyt_array] = None
    star_spread: Optional[unyt_array] = None

    def __init__(
        self,
        *,  # Keyword arguments only
        # Arguments for using the simulation data
        simulation: Optional[SimulationData] = None,
        # Arguments for manual use of dark matter coordinates
        dark_matter_initial_coordinates: Optional[unyt_array] = None,
        gas_initial_coordinates: Optional[unyt_array] = None,
        dark_matter_initial_ids: Optional[unyt_array] = None,
        gas_initial_ids: Optional[unyt_array] = None,
        dark_matter_final_coordinates: Optional[unyt_array] = None,
        dark_matter_final_ids: Optional[unyt_array] = None,
        gas_final_coordinates: Optional[unyt_array] = None,
        gas_final_ids: Optional[unyt_array] = None,
        star_final_coordinates: Optional[unyt_array] = None,
        star_final_ids: Optional[unyt_array] = None,
        boxsize: Optional[unyt_array] = None,
    ):
        # If present, extract everything from the simulation object.
        self.simulation = simulation

        if simulation is not None:

            initial_snap = simulation.initial_snapshot
            final_snap = simulation.final_snapshot

            self.dark_matter_initial_coordinates = initial_snap.dark_matter.coordinates
            
            self.dark_matter_initial_ids = initial_snap.dark_matter.particle_ids
            #LOGGER.info(f"Initial IDs SM {self.dark_matter_initial_ids}")

            if initial_snap.gas is not None:
                self.gas_initial_coordinates = initial_snap.gas.coordinates
                
                self.gas_initial_ids = initial_snap.gas.particle_ids

            self.dark_matter_final_coordinates = final_snap.dark_matter.coordinates
            self.dark_matter_final_ids = final_snap.dark_matter.particle_ids
            #LOGGER.info(f"Final IDs SM {self.dark_matter_final_ids}")            

            if final_snap.gas is not None:
                self.gas_final_coordinates = final_snap.gas.coordinates
                LOGGER.info(self.gas_final_coordinates[:,0])
                self.gas_final_ids = final_snap.gas.particle_ids

            #if final_snap.stars is not None:
            #    self.star_final_coordinates = final_snap.stars.coordinates
            #    self.star_final_ids = final_snap.stars.particle_ids

            self.boxsize = simulation.final_snapshot.boxsize

        if dark_matter_initial_coordinates is not None:
            self.dark_matter_initial_coordinates = dark_matter_initial_coordinates

        if gas_initial_coordinates is not None:
            self.gas_initial_coordinates = gas_initial_coordinates

        if dark_matter_initial_ids is not None:
            self.dark_matter_initial_ids = dark_matter_initial_ids

        if gas_initial_ids is not None:
            self.gas_initial_ids = gas_initial_ids

        if dark_matter_final_coordinates is not None:
            self.dark_matter_final_coordinates = dark_matter_final_coordinates

        if dark_matter_final_ids is not None:
            self.dark_matter_final_ids = dark_matter_final_ids

        if gas_final_coordinates is not None:
            self.gas_final_coordinates = gas_final_coordinates
            LOGGER.info(self.gas_final_coordinates[:,0])

        if gas_final_ids is not None:
            self.gas_final_ids = gas_final_ids
            LOGGER.info(self.gas_final_ids)

        if star_final_coordinates is not None:
            self.star_final_coordinates = star_final_coordinates

        if star_final_ids is not None:
            self.star_final_ids = star_final_ids

        if boxsize is not None:
            self.boxsize = boxsize

        return

    def find_neighbours(self):
        """
        Finds the initial state neighbours. Sets self.dark_matter_neighbours
        and self.gas_neighbours.
        """

        LOGGER.info("Beginning search for initial neighbours")
        self.dark_matter_neighbours, self.gas_neighbours = find_closest_neighbours(
            dark_matter_coordinates=self.dark_matter_initial_coordinates,
            dark_matter_ids=self.dark_matter_initial_ids,
            boxsize=self.boxsize,
            gas_coordinates=self.gas_initial_coordinates,
            gas_ids=self.gas_initial_ids,
        )
        LOGGER.info("Finished search for initial neighbours")

        return

    def find_neighbour_distances(self):
        """
        Finds the distances to the initial nearest neighbours (i.e.
        this function computes the spread metric for all particles associated
        with this object). If relevant, sets: ``self.dark_matter_spread``,
        ``self.gas_spread``, ``self.star_spread``.
        """

        if self.dark_matter_neighbours is None:
            self.find_neighbours()

        if self.dark_matter_final_coordinates is not None:
            LOGGER.info("Computing dark matter spread metric")
            self.dark_matter_spread = find_neighbour_distances(
                neighbours=self.dark_matter_neighbours,
                particle_coordinates=self.dark_matter_final_coordinates.value,
                dark_matter_coordinates=self.dark_matter_final_coordinates.value,
                particle_ids=self.dark_matter_final_ids.value,
                dark_matter_ids=self.dark_matter_final_ids.value,
                boxsize=self.boxsize.value,
            )
            self.dark_matter_spread = unyt_array(
                self.dark_matter_spread, units=self.dark_matter_final_coordinates.units
            )
            LOGGER.info("Finished computing dark matter spread metric")
        else:
            self.dark_matter_spread = None

        if self.gas_final_coordinates is not None:
            LOGGER.info("Computing gas spread metric")
            LOGGER.info(f"Gas initial ids: {self.gas_initial_ids}")
            LOGGER.info(f"Gas final ids: {self.gas_final_ids}")
            LOGGER.info(f"Sorted gas final ids: {np.sort(self.gas_final_ids)}")
            LOGGER.info(f"Gas final coords: {self.gas_final_coordinates}")
            self.gas_spread = find_neighbour_distances(
                neighbours=self.gas_neighbours,
                particle_coordinates=self.gas_final_coordinates.value,
                dark_matter_coordinates=self.dark_matter_final_coordinates.value,
                particle_ids=self.gas_final_ids.value,
                dark_matter_ids=self.dark_matter_final_ids.value,
                boxsize=self.boxsize.value,
            )
            self.gas_spread = unyt_array(
                self.gas_spread, units=self.gas_final_coordinates.units
            )
            LOGGER.info("Finished computing gas spread metric")
        else:
            self.gas_spread = None

        if self.star_final_coordinates is not None:
            LOGGER.info("Computing star spread metric")
            self.star_spread = find_neighbour_distances(
                neighbours=self.gas_neighbours,
                particle_coordinates=self.star_final_coordinates.value,
                dark_matter_coordinates=self.dark_matter_final_coordinates.value,
                particle_ids=self.star_final_ids.value,
                dark_matter_ids=self.dark_matter_final_ids.value,
                boxsize=self.boxsize.value,
            )
            self.star_spread = unyt_array(
                self.star_spread, units=self.star_final_coordinates.units
            )
            LOGGER.info("Finished computing star spread metric")
        else:
            self.star_spread = None

        return
