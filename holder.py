"""
Contains the holder object for our simulation datasets, and performs
comparisons between them.
"""

from transfer import LOGGER
from transfer.data import SnapshotData
from transfer.transfer import calculate_transfer_masses, TransferOutput

import lagrangian as lagrangian

from unyt import unyt_quantity
from numpy import float64, int32


class SimulationData(object):
    """
    Simulation dataset holder. This takes two instances of the
    :class:`transfer.data.SnapshotData` class, one for the initial
    conditions and one for the final conditions. This class then
    contains all the necessary logic to match the Lagrangian Regions
    between each.

    Attributes
    ----------

    initial_snapshot: SnapshotData
        The initial conditions in a SnapshotData class or subclass.

    final_snapshot: SnapshotData
        The final conditions in a SnapshotData class or subclass.

    transfer_units: unyt.unyt_quantity
        Units for the below TransferOutput masses.

    dark_matter_transfer: TransferOutput
        Results of the Lagrangian Transfer calculation on dark matter.

    gas_transfer: TransferOutput
        Results of the Lagrangian Transfer calculation on gas.

    star_transfer: TransferOutput
        Results of the Lagrangian Transfer calculation on stars.

    Notes
    -----

    This is less complex than the version presented in V1.0 as
    it expects a user to deal with the snapshots themselves, and
    only return an object that conforms to the SnapshotData class.
    """

    initial_snapshot: SnapshotData
    final_snapshot: SnapshotData
    transfer_units: unyt_quantity
    dark_matter_transfer: TransferOutput
    gas_transfer: TransferOutput
    star_transfer: TransferOutput
    dark_matter_haloes: int32
    dark_matter_lagrangian_regions: int32
    gas_haloes: int32
    gas_lagrangian_regions: int32
    star_haloes: int32
    star_lagrangian_regions: int32

    def __init__(self, initial_snapshot: SnapshotData, final_snapshot: SnapshotData):
        self.initial_snapshot = initial_snapshot
        self.final_snapshot = final_snapshot

        self.transfer_units = initial_snapshot.dark_matter.masses.units

        self.associate_lagrangian_to_dark_matter()
        self.associate_initial_lagrangian_to_gas()
        self.parse_lagrangian_regions()
        self.calculate_transfer()

        return

    def associate_lagrangian_to_dark_matter(self):
        """
        Associates the haloes in the final snapshot with the dark matter
        lagrangian regions (and hence haloes, as these are defined as being
        the same thing) in the initial conditions.

        Notes
        -----

        This method assumes that all data has already been sorted by particle ID.
        """

        LOGGER.info("Setting lagrangian regions for dark matter")

        initial_dm = self.initial_snapshot.dark_matter
        final_dm = self.final_snapshot.dark_matter

        final_dm.lagrangian_regions = final_dm.haloes

        initial_dm.haloes = final_dm.haloes
        initial_dm.lagrangian_regions = final_dm.haloes

        LOGGER.debug(
            (
                "There are: "
                f"{initial_dm.particle_ids.size} initial DM particles, "
                f"{final_dm.particle_ids.size} final DM particles. "
                f"The final DM particle ID is {final_dm.particle_ids[-1]} "
                f"and the initial DM particle ID is {initial_dm.particle_ids[-1]}."
            )
        )

        return

    def associate_initial_lagrangian_to_gas(self):
        """
        Associates the Lagrangian Region IDs to the gas particles in the
        initial conditions state using a tree search.
        """

        LOGGER.info(
            "Setting lagrangian regions from dark matter to gas in initial conditions"
        )

        gas_lagrangian_regions = lagrangian.calculate_gas_lagrangian_regions(
            dark_matter_coordinates=self.initial_snapshot.dark_matter.coordinates,
            gas_coordinates=self.initial_snapshot.gas.coordinates,
            dark_matter_lagrangian_regions=self.initial_snapshot.dark_matter.lagrangian_regions,
            boxsize=self.initial_snapshot.boxsize,
        )

        self.initial_snapshot.gas.lagrangian_regions = gas_lagrangian_regions

        return

    def parse_lagrangian_regions(self):
        """
        Performs the cross-matching between snapshots to set the Lagrangian
        Regions of gas and stellar particles (if present).
        """

        initial_gas_ids = self.initial_snapshot.gas.particle_ids
        initial_gas_lrs = self.initial_snapshot.gas.lagrangian_regions

        for particle_type in ["gas", "stars"]:
            particle_data = getattr(self.final_snapshot, particle_type, None)

            if particle_data is not None:
                LOGGER.info(f"Running LR parsing on {particle_type}")

                lagrangian_regions = lagrangian.parse_lagrangian_regions(
                    initial_particle_ids=initial_gas_ids,
                    initial_lagrangian_regions=initial_gas_lrs,
                    comparison_particle_ids=particle_data.particle_ids,
                )

                LOGGER.info("Finished running cross matching for this particle type")

                getattr(
                    self.final_snapshot, particle_type
                ).lagrangian_regions = lagrangian_regions
            else:
                LOGGER.info(f"No {particle_type} data available in final snapshot")

        return

    def get_initial_particle_mass(self, particle_type: str) -> float64:
        """
        Gets the initial particle mass corresponding to the particle type
        and returns it as a float in internal units.
        """

        if particle_type == "dark_matter":
            return float64(self.initial_snapshot.dark_matter.masses[0].value)
        else:
            # Return gas particle mass (e.g. stars)
            if self.initial_snapshot.gas is not None:
                return float64(self.initial_snapshot.gas.masses[0].value)
            else:
                return float64(0.0)

    def calculate_transfer(self):
        """
        Calculates the amounts of lagrangian transfer and stores it in
        self.{particle_type}_transfer for all present particle types.
        """

        number_of_groups = self.final_snapshot.number_of_groups

        for particle_type in ["dark_matter", "gas", "stars"]:
            particle_data = getattr(self.final_snapshot, particle_type, None)
            
            initial_particle_mass = self.get_initial_particle_mass(particle_type)

            if particle_data is not None:
                LOGGER.info(f"Beginning transfer calculation for {particle_type}")
                LOGGER.debug(
                    (
                        "This particle type has: "
                        f"{len(particle_data.haloes)} haloes, "
                        f"{len(particle_data.lagrangian_regions)} LRs, "
                        f"{len(particle_data.masses)} masses."
                    )
                )
                setattr(
                    self,
                    f"{particle_type}_transfer",
                    calculate_transfer_masses(
                        haloes=particle_data.haloes,
                        lagrangian_regions=particle_data.lagrangian_regions,
                        particle_masses=particle_data.masses.value,
                        initial_particle_mass=initial_particle_mass,
                        number_of_groups=number_of_groups,
                    ),
                )
                setattr(
                    self,
                    f"{particle_type}_haloes",
                    particle_data.haloes,
                )
                setattr(
                    self,
                    f"{particle_type}_lagrangian_regions",
                    particle_data.lagrangian_regions,
                )

        return
    
#    def output_halo_particle_data(self):
        
