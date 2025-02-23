"""
The SIMBA and AHF frontend for :mod:``transfer``.

Requires the ``h5py`` python package.
"""

##########################################################################################################
# I (Matt) have edited this code to make it work with initial conditions that require readgadget to load #
#                                       All edits signed as - MG                                         #
##########################################################################################################

from transfer import LOGGER

from transfer.data import ParticleData, SnapshotData

from typing import Optional
import unyt
from unyt import unyt_array, unyt_quantity
import numpy as np
from numpy import full, concatenate, isin, genfromtxt, array, uint64, unique
import readgadget # - MG 6/24/21

import h5py

# Units without the h-correction.
unit_mass = unyt_quantity(1e10, "Solar_Mass")
unit_length = unyt_quantity(1.0, "kpc")
unit_velocity = unyt_quantity(1.0,"km/s")


class SIMBAParticleData(ParticleData):
    """
    SIMBA particle data frontend. Implements the :class:`ParticleData`
    class with real functionality for reading EAGLE snapshots.
    """

    def __init__(self, filename: str, particle_type: int, truncate_ids: Optional[int] = None):
        """
        Parameters
        ----------

        filename: str
            The SIMBA snapshot filename to extract the particle data from.

        particle_type: int
            The particle type to load (0, 1, 4, etc.)

        truncate_ids: int, optional
            Truncate IDs above this by using the % operator; i.e. discard
            higher bits.
        """
        super().__init__()

        self.filename = filename
        self.particle_type = particle_type
        self.truncate_ids = truncate_ids

        LOGGER.info(f"Loading particle data from particle type {particle_type}")
        self.coordinates = self.load_coordinates()
        self.masses = self.load_masses()
        self.particle_ids = self.load_particle_ids()
        LOGGER.info(f"Finished loading data from particle type {particle_type}")
        LOGGER.info(f"Loaded {self.particle_ids.size} particles")

        self.perform_particle_id_postprocessing()

        return

    def load_coordinates(self):
        """
        Loads the coordinates from the file, returning an unyt array.
        """
        coords = "Coordinates"

        header = readgadget.header(self.filename)
        h = header.hubble
        units = unyt_quantity(1.0 / h, units=unit_length).to("Mpc")
        
        raw, h = self.load_data(coords)
        
        print(raw)

        coordinates = unyt_array(raw, units=units)
                    
        print(coordinates)
        #Here, I added an additional boxsize loading operation because, apparently, coordinates are loaded before
        #the boxsize. This appeared to be a simpler solution than creating a setter function to change the positions
        #later on outside of the class. - MG 6/24/21
        if self.filename[-4:] == 'hdf5':

            with h5py.File(self.filename, "r") as handle:
                hubble_param = handle["Header"].attrs["HubbleParam"]
                corrected_unit_length = unit_length / hubble_param
                boxsize = handle["Header"].attrs["BoxSize"]
                units = unyt_quantity(1.0 / hubble_param, units=unit_length).to("Mpc")

            self.hubble_param = hubble_param
            self.boxsize = unyt_quantity(boxsize, units=units)
            
        else:
            header = readgadget.header(self.filename)
            h = header.hubble
            units = unyt_quantity(1.0 / h, units=unit_length).to("Mpc")
            self.hubble_param = h
            self.boxsize  = unyt.unyt_array(header.boxsize, units=units)
            
        #Checks to make sure there are no particles outside the box - MG 6/24/21
        ind_check0 = np.where(coordinates[:,0].value >= self.boxsize.value)[0]
        if ind_check0.any() == True:
            coordinates[ind_check0, 0] = self.boxsize.value - 0.1
            LOGGER.info(f"Fixed index {ind_check0} of the X coordinate of particle type {self.particle_type}")
        ind_check1 = np.where(coordinates[:,1].value >= self.boxsize.value)[0]
        if ind_check1.any() == True:
            coordinates[ind_check1, 1] = self.boxsize.value - 0.1
            LOGGER.info(f"Fixed index {ind_check0} of the Y coordinate of particle type {self.particle_type}")
        ind_check2 = np.where(coordinates[:,2].value >= self.boxsize.value)[0]
        if ind_check2.any() == True:
            coordinates[ind_check2, 2] = self.boxsize.value - 0.1
            LOGGER.info(f"Fixed index {ind_check0} of the Z coordinate of particle type {self.particle_type}")    
        
        return coordinates

    def load_masses(self):
        """
        Loads the masses from the file, returning an unyt array.
        """
            
        header = readgadget.header(self.filename)
        h = header.hubble
        units = unyt_quantity(1.0 / h, units=unit_mass).to("Solar_Mass")
        
        raw, h = self.load_data("Masses")

        masses = unyt_array(raw, units=units)
            

        return masses

    def load_particle_ids(self):
        """
        Loads the Particle IDs from the file, returning an unyt array.
        """
        
        

        raw, _ = self.load_data("ParticleIDs")
            
        IDs = unyt_array(raw.astype(uint64), units=None, dtype=uint64)
        
        print(IDs)
        
        #if self.filename[-4:] == 'hdf5':
            #LOGGER.info(f"Final {self.particle_type} IDs SPD {IDs}")
            
        #else:
             #LOGGER.info(f"Initial {self.particle_type} IDs SPD {IDs}")
        return IDs

    def load_data(self, array_name: str):
        """
        Loads an array and returns it.

        Parameters
        ----------

        array_name: str
            Name of the array (without particle type) to read, e.g. Coordinates


        Returns
        -------

        output: np.array
            Output read from the HDF5 file

        h: float
            Hubble parameter.

        """
        
        #Added functionality to read initial conditions, which require readgadget - MG 6/21/19
        
        if self.filename[-4:] == 'hdf5':
            
            full_path = f"/PartType{self.particle_type}/{array_name}"

            LOGGER.info(f"Loading data from {full_path}.")

            with h5py.File(f"{self.filename}", "r") as handle:
                h = handle["Header"].attrs["HubbleParam"]
                output = handle[full_path][:]
                
        else:
            header = readgadget.header(self.filename)
            h = header.hubble
            if array_name == "Coordinates":
                array_name = "POS "
            if array_name == "ParticleIDs":
                array_name = "ID  "
            if array_name == "Masses":
                array_name = "MASS"               
                
            #we should have no masses for stars in initial conditions - MG 6/24/21    
            if array_name == "MASS" and self.particle_type == 4:
                output = unyt.unyt_array([])
                
            else:
                output = readgadget.read_block(self.filename, array_name, [self.particle_type], verbose=True)
            

        return output, h

    def perform_particle_id_postprocessing(self):
        """
        Performs postprocessing on ParticleIDs to ensure that they link
        correctly (required in cases where the particle IDs are offset when
        new generations of particles are spawned).
        """

        if self.truncate_ids is None:
            LOGGER.info("Beginning particle ID postprocessing (empty).")
        else:
            LOGGER.info("Beginning particle ID postprocessing.")
            LOGGER.info(f"Truncating particle IDs above {self.truncate_ids}")

            self.particle_ids %= self.truncate_ids
            
            # TODO: Remove this requiremnet. At the moment, isin() breaks when
            #       you have repeated values.

            self.particle_ids, indicies = unique(self.particle_ids, return_index=True)
            self.coordinates = self.coordinates[indicies]
            self.masses = self.masses[indicies]

        return


class SIMBASnapshotData(SnapshotData):
    """
    SIMBA particle data frontend. Implements the :class:`SnapshotData`
    class with real functionality for reading SIMBA snapshots and
    SIMBA AHF catalogues.
    """

    def __init__(
            self, filename: str, halo_filename: Optional[str] = None, truncate_ids: Optional[dict] = None,
    ):
        """
        Parameters
        ----------

        filename: str
            Filename for the snapshot file.

        halo_filename: str, optional
            Filename for the halo file.

        truncate_ids: Dict[int,int], optional
            Dictionary from particle_type (i.e. 0, 1, 4): truncation 
            amount.

            Truncate IDs above this by using the % operator; i.e. discard
            higher bits.
        """

        self.load_boxsize(filename=filename)
        self.truncate_ids = truncate_ids

        super().__init__(filename=filename, halo_filename=halo_filename)

        return

    def load_boxsize(self, filename: str):
        """
        Loads the boxsize and hubble param from the particle file.
        """
        #Added readgadget functionality - MG 6/24/21
        
        if filename[-4:] == 'hdf5':

            with h5py.File(filename, "r") as handle:
                hubble_param = handle["Header"].attrs["HubbleParam"]
                corrected_unit_length = unit_length / hubble_param
                boxsize = handle["Header"].attrs["BoxSize"]
                units = unyt_quantity(1.0 / hubble_param, units=unit_length).to("Mpc")

            self.hubble_param = hubble_param
            self.boxsize = unyt_quantity(boxsize, units=units)
            
        else:
            header = readgadget.header(filename)
            h = header.hubble
            units = unyt_quantity(1.0 / h, units=unit_length).to("Mpc")
            self.hubble_param = h
            self.boxsize  = unyt.unyt_array(header.boxsize, units=units)

        return

    def load_particle_data(self):
        """
        Loads the particle data from a snapshot using ``h5py``.
        """

        for particle_type, particle_name in zip(
            [1, 0, 4], ["dark_matter", "gas", "stars"]
        ):
            truncate_ids = self.truncate_ids[particle_type] if self.truncate_ids is not None else None

            try:
                setattr(
                    self,
                    particle_name,
                    SIMBAParticleData(
                        filename=self.filename, particle_type=particle_type, truncate_ids=truncate_ids
                    ),
                )
            except KeyError:
                # No particles of this type (e.g. stars in ICs)
                LOGGER.info(
                    (
                        f"No particles of type {particle_type} ({particle_name}) "
                        "in this file. Skipping."
                    )
                )

                setattr(self, particle_name, None)

        return

    def load_halo_data(self):
        """
        Loads haloes from AHF and, using the center of mass of the halo
        and R_vir, uses trees through :meth:`ParticleDataset.associate_haloes`
        to set the halo values.

        Loads only central haloes (with hostHalo = -1)
        """
        
        if self.halo_filename is None:
                return
        
        if self.halo_filename[-4:] == 'hdf5':
            
            LOGGER.info(f"Loading halo catalogue data from {self.halo_filename}")
            new_haloes = h5py.File(self.halo_filename, 'r')
            pos = new_haloes['Group/GroupPos'][:,:]
            xmbp = pos[:,0]
            ymbp = pos[:,1]
            zmbp = pos[:,2]
            center_of_potential = array([xmbp, ymbp, zmbp]).T

            r_vir = new_haloes['Group/Group_R_Crit200'][:]
            
            
        else:
            
            

            LOGGER.info(f"Loading halo catalogue data from {self.halo_filename}")

            raw_data = genfromtxt(self.halo_filename, usecols=[1, 5, 6, 7, 11]).T

            hostHalo = raw_data[0].astype(int)
            mask = hostHalo == -1

            #Currently, the search for only central haloes has been stopped due to that column of the data being absent - MG 6/24/21

            #xmbp = raw_data[1][mask]
            #ymbp = raw_data[2][mask]
            #zmbp = raw_data[3][mask]
            xmbp = raw_data[1]
            ymbp = raw_data[2]
            zmbp = raw_data[3]


            center_of_potential = array([xmbp, ymbp, zmbp]).T


            #r_vir = raw_data[4][mask]
            r_vir = raw_data[4]



        units = unyt_quantity(1.0 / self.hubble_param, units=unit_length).to("Mpc")

        halo_coordinates = unyt_array(center_of_potential, units=units)
        halo_radii = unyt_array(r_vir, units=units)

        #LOGGER.info(halo_coordinates)
        #LOGGER.info(halo_radii)

        self.number_of_groups = halo_radii.size

        LOGGER.info("Finished loading halo catalogue data")

        for particle_type in ["dark_matter", "gas", "stars"]:
            particle_data = getattr(self, particle_type, None)

            if particle_data is not None:
                LOGGER.info(f"Associating haloes for {particle_type}")
                particle_data.associate_haloes(
                    halo_coordinates=halo_coordinates,
                    halo_radii=halo_radii,
                    boxsize=self.boxsize,
                )

        self.halo_coordinates = halo_coordinates
        self.halo_radii = halo_radii
        
        return
    
    #I have commented out this function to stop sorting of particles by ID in final snapshot - MG 6/24/21
    
    #sike

    def sort_all_data(self):
        """
        Sorts all data currently present in the snapshot by particle ID on
        a particle type by particle type basis.
        """

        for particle_type in ["dark_matter", "gas", "stars"]:
            particle_data = getattr(self, particle_type, None)

            if particle_data is not None:
                particle_data.sort_by_particle_id()

        return
