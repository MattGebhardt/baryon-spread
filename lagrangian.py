"""
Computes the overlap in Lagrangian Regions taking into
account that there can be particles that swap 'types'.
"""

from transfer import LOGGER

from numpy import int32, int64, isin
from numba import njit

from scipy.spatial import cKDTree
from unyt import unyt_array


def parse_lagrangian_regions(
    initial_particle_ids: int64,
    initial_lagrangian_regions: int32,
    comparison_particle_ids: int64,
) -> int32:
    """
    Parses the Lagrangian Regions from the initial conditions to the output
    particle types. This is required as some particles will have disappeared.

    Call this for each particle type. This is just a linear pass over particles
    so repeating it is no big deal.


    Parameters
    ----------

    initial_particle_ids: np.array[int64]
        Particle IDs in the initial conditions.

    initial_lagrangian_regions: np.array[int32]
        Lagrangian regions in the initial conditions.

    comparison_particle_ids: np.array[int64]
        Particle IDs for comparison in the final state of the simulation.
        This would be, for example, the IDs of the z=0 gas particles.

        For these particles, we will find LR IDs, corresponding to their
        particle ID.


    Returns
    -------

    comparison_lagrangian_regions: np.array[int32]
        Lagrangian Region IDs correspondong to the ``comparison_particle_ids``.


    Notes
    -----

    By comparison to LT 1.0, this version of the code simplifies this function
    a lot by doing separate runs through particles for gas, stars, and dark matter,
    instead of trying to combine it all together. We also no longer truncate
    particle IDs.
    """

    comparison_lagrangian_regions = initial_lagrangian_regions[
        isin(initial_particle_ids, comparison_particle_ids, assume_unique=False)
    ]

    return comparison_lagrangian_regions


def calculate_gas_lagrangian_regions(
    dark_matter_coordinates: unyt_array,
    gas_coordinates: unyt_array,
    dark_matter_lagrangian_regions: int32,
    boxsize: unyt_array,
) -> int32:
    """
    Computes the Lagrangian Regions for gas particles by using a tree of their
    co-ordinates and the (defined) Lagrangian Regions for dark matter.

    Parameters
    ----------

    dark_matter_coordinates: unyt_array[float64]
        The co-oridnates of dark matter particles

    gas_coordinates: unyt_array[float64]
        The co-ordinates of gas particles

    dark_matter_lagrangian_regions: np.array[int32]
        Lagrangian Regions of the dark matter particles.

    boxsize: unyt_array
        The box-size of the simulation so that periodic overlaps
        can be considered in the nearest neighbour calculation.


    Returns
    -------

    gas_lagrangian_regions: np.array[int32]
        Lagrangian Regions of the gas particles, based on the tree search
        of the dark matter particles.


    Notes
    -----

    The Lagrangian Region of a gas particle is defined as being the same
    as the Lagrangian Region of the closest dark matter particle.
    """

    # We should just crash here if this is not the case.
    assert gas_coordinates.units == dark_matter_coordinates.units

    boxsize = boxsize.to(gas_coordinates.units)

    LOGGER.info("Beginning treebuild")
    dark_matter_tree = cKDTree(dark_matter_coordinates.value, boxsize=boxsize)
    LOGGER.info("Finished treebuild")

    LOGGER.info("Beginning tree search")
    _, indicies = dark_matter_tree.query(x=gas_coordinates, k=1, workers=-1)
    LOGGER.info("Finished tree walk")

    gas_lagrangian_regions = dark_matter_lagrangian_regions[indicies]

    return gas_lagrangian_regions
