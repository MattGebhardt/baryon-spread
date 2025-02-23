"""
Handles transformation of data into a format that we can
use with transfer.
"""

from transfer import LOGGER

from typing import Union, Optional
from numpy import array, full, int32
from unyt import unyt_array
from scipy.spatial import cKDTree


class ParticleData(object):
    """
    Individual particle dataset.

    Attributes
    ----------

    coordinates: unyt.unyt_array[float64]
        Co-moving co-ordinates of the particles.

    masses: unyt.unyt_array[float64]
        Masses of the particles.

    particle_ids: unyt.unyt_array[int64]
        IDs of the particles.

    haloes: np.array[int32]
        Halo IDs of the particles (if relevant).

    lagrangian_regions: np.array[int32]
        Lagrangian Region IDs of the particles (if relevant).


    Notes
    -----

    Typically the final state snapshot will include the halo IDs and
    Lagrangian Region IDs, whereas the initial state particles will
    only contain Lagrangian Region IDs.
    """

    coordinates: unyt_array
    masses: unyt_array
    particle_ids: unyt_array
    haloes: Optional[array]
    lagrangian_regions: Optional[array]

    def __init__(self):
        return

    def sort_by_particle_id(self):
        """
        Sorts the internal data by the particle IDs.
        """

        LOGGER.info("Beginning sort for particle IDs")
        mask = self.particle_ids.argsort()
        LOGGER.info("Finished sort on particle IDs")

        LOGGER.info("Beginning masking of sorted arrays")
        self.coordinates = self.coordinates[mask]
        self.masses = self.masses[mask]
        self.particle_ids = self.particle_ids[mask]

        try:
            self.haloes = self.haloes[mask]
        except (NameError, TypeError, AttributeError):
            # Haloes does not exist.
            LOGGER.info("No haloes property found on this instance")
            pass

        try:
            self.lagrangian_regions = self.lagrangian_regions[mask]
        except (NameError, TypeError, AttributeError):
            # Haloes does not exist.
            LOGGER.info("No lagrangian_regions property found on this instance")
            pass

        LOGGER.info("Finished masking of sorted arrays")

        return

    def associate_haloes(
        self, halo_coordinates: unyt_array, halo_radii: unyt_array, boxsize: unyt_array
    ):
        """
        Associates the haloes with this dataset. Performs this task by building
        a tree of the particle co-ordinates, and then searching it.

        Parameters
        ----------

        halo_coordinates: unyt.unyt_array[float64]
            Co-ordinates of the haloes. Should be NX3.

        halo_radii: unyt.unyt_array[float64]
            Halo radii. Should be NX1.

        boxsize: unyt.unyt_array[float64]
            Boxsize of the simulation to enable periodic overlaps.


        Notes
        -----

        This function sets the self.haloes property.
        """

        boxsize = boxsize.to(self.coordinates.units)

        LOGGER.info("Beginning treebuild")
        tree = cKDTree(self.coordinates.value, boxsize=boxsize.value)
        LOGGER.info("Finished treebuild")

        haloes = full(self.masses.size, -1, dtype=int32)

        halo_coordinates.convert_to_units(self.coordinates.units)
        halo_radii.convert_to_units(self.coordinates.units)

        # Search the tree in blocks of haloes as this improves load balancing
        # by allowing the tree to parallelise.
        block_size = 1024
        number_of_haloes = halo_radii.size
        number_of_blocks = 1 + number_of_haloes // block_size

        LOGGER.info("Beginning tree search")

        for block in range(number_of_blocks):
            LOGGER.debug(f"Running tree search on block {block}/{number_of_blocks}")

            starting_index = block * block_size
            ending_index = (block + 1) * (block_size)

            if ending_index > number_of_haloes:
                ending_index = number_of_haloes + 1

            if starting_index >= ending_index:
                break

            particle_indicies = tree.query_ball_point(
                x=halo_coordinates[starting_index:ending_index].value,
                r=halo_radii[starting_index:ending_index].value,
                n_jobs=-1,
            )

            for halo, indicies in enumerate(particle_indicies):
                haloes[indicies] = int32(halo + starting_index)

        self.haloes = haloes

        LOGGER.debug(
            f"Maximal halo ID = {haloes.max()}. Number of haloes: {number_of_haloes}"
        )
        LOGGER.info("Finished tree search")

        return


class SnapshotData(object):
    """
    Represents a snapshot. You should sub-class this if you are going
    to use a specific frontend (e.g. there is a ``swiftsimio`` frontend).

    Attributes
    ----------

    filename: str
        Filename for the snapshot file

    halo_filename: str
        Halo catalogue filename

    dark_matter: ParticleData
        Dark matter particle data, including halo and LR information.

    gas: ParticleData
        Gas particle data, including halo and LR information.

    boxsize: unyt_array[float64]
        The box-size of the simulation so that periodic bounary conditions
        are considered.

    number_of_groups: int32
        Total number of groups (haloes) in the simulation.

    stars: ParticleData, optional
        Star particle data, including halo and LR information.



    Notes
    -----

    This class first loads the particle data, then the halo data. Once
    this is complete, it sorts all data using the
    :meth:`ParticleData.sort_by_particle_id` method.

    """

    filename: str
    halo_filename: Optional[str]
    dark_matter: ParticleData
    gas: ParticleData
    stars: Optional[ParticleData]

    boxsize: unyt_array
    number_of_groups: int32

    def __init__(self, filename: str, halo_filename: Optional[str]):
        self.filename = filename
        self.halo_filename = halo_filename

        LOGGER.info(f"Beginning particle load operation for {self.filename}")
        self.load_particle_data()
        LOGGER.info("Finished particle loading")
        LOGGER.info(f"Beginning halo loading for {self.halo_filename}")
        self.load_halo_data()
        LOGGER.info("Finished halo loading")
        LOGGER.info(f"Beginning master sort for {self.filename}")
        self.sort_all_data()
        LOGGER.info(f"Finished master sort")

        return

    def load_particle_data(self):
        """
        Loads the particle data. Must be implemented by a frontend.
        """

        return

    def load_halo_data(self):
        """
        Loads the halo data. Must be implemented by a frontend.
        """

        return

    def sort_all_data(self):
        """
        Sorts all data that has been read by each individual
        frontend.
        """

        for particle_type in ["dark_matter", "gas", "stars"]:
            try:
                getattr(self, particle_type, None).sort_by_particle_id()
            except (NameError, TypeError):
                # This particle type does not exist.
                pass

        return
