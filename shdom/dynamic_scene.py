"""
Dynamic_cloud related objects used for time dependant cloud changing.

"""

import warnings
import glob
from collections import OrderedDict
from joblib import Parallel, delayed
import scipy.ndimage as sci
import numpy as np
import time, os, copy, shutil
from scipy.optimize import minimize
import shdom
import dill as pickle
import tensorboardX as tb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import filters
import scipy.io as sio
from scipy.stats import norm



def save_dynamic_forward_model(directory, dynamic_medium, dynamic_solver, measurements):
    """
    Save the forward model parameters for reconstruction.

    Parameters
    ----------
    directory: str
        Directory path where the forward modeling parameters are saved.
        If the folder doesnt exist it will be created.
    medium: shdom.Medium object
        The atmospheric medium. This ground-truth medium will be used for comparisons.
    solver: shdom.RteSolver object
        The solver and the parameters used. This includes the scene parameters (such as solar and surface parameters)
        and the numerical parameters.
    measurements: shdom.Measurements
        Contains the camera used and the measurements acquired.

    Notes
    -----
    The ground-truth medium is later used for evaulation of the recovery.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)
    measurements.save(os.path.join(directory, 'measurements'))
    if dynamic_medium is not None:
        dynamic_medium.save(os.path.join(directory, 'ground_truth_dynamic_medium'))
    dynamic_solver.save_params(os.path.join(directory, 'solver_parameters'))


def load_dynamic_forward_model(directory):
    """
    Save the forward model parameters for reconstruction.

    Parameters
    ----------
    directory: str
        Directory path where the forward modeling parameters are saved.

    Returns
    -------
    medium: shdom.Medium object
        The ground-truth atmospheric medium.
    solver: shdom.RteSolver object
        The solver and the parameters used. This includes the scene parameters (such as solar and surface parameters)
        and the numerical parameters.
    measurements: shdom.Measurements
        Contains the sensor used to image the mediu and the radiance measurements.

    Notes
    -----
    The ground-truth medium is used for evaulation of the recovery.
    """
    # Load the ground truth medium for error analysis and ground-truth known phase and albedo
    medium_path = os.path.join(directory, 'ground_truth_dynamic_medium')
    if os.path.exists(medium_path):
        medium = DynamicMedium()
        medium.load(path=medium_path)
    else:
        medium = None

    # Load shdom.Measurements object (sensor geometry and radiances)
    measurements = DynamicMeasurements()
    measurements_path = os.path.join(directory, 'measurements')
    assert os.path.exists(measurements_path), 'No measurements file in directory: {}'.format(directory)
    measurements.load(path=measurements_path)

    # Load RteSolver according to numerical and scene parameters
    solver_path = os.path.join(directory, 'solver_parameters')
    solver = DynamicRteSolver()
    if os.path.exists(solver_path):
        solver.load_params(path=os.path.join(directory, 'solver_parameters'))
    # solver.set_dynamic_medium(medium)
    return medium, solver, measurements


class TemporaryScatterer(object):
    # TODO
    def __init__(self, scatterer, time=0.0):
        assert isinstance(scatterer,shdom.Scatterer) or isinstance(scatterer, shdom.MultispectralScatterer) #check if time is a number
        super().__init__()
        self._time = float(time)
        self._scatterer = scatterer
        if isinstance(scatterer,shdom.OpticalScatterer):
            # super().__init__(wavelength=scatterer.wavelength, extinction=scatterer.extinction, albedo=scatterer.albedo, phase=scatterer.phase)
            self._type = 'OpticalScatterer'
        elif isinstance(scatterer,shdom.MicrophysicalScatterer):
            # super().__init__(lwc=scatterer.lwc, reff=scatterer.reff, veff=scatterer.veff)
            # if scatterer.num_wavelengths > 0:
            #     self.add_mie(scatterer.mie[scatterer.wavelength])
            self._type = 'MicrophysicalScatterer'
        elif isinstance(scatterer,shdom.MultispectralScatterer):
            # super().__init__(lwc=scatterer.lwc, reff=scatterer.reff, veff=scatterer.veff)
            # if scatterer.num_wavelengths > 0:
            #     self.add_mie(scatterer.mie[scatterer.wavelength])
            self._type = 'MultispectralScatterer'
        else:
            assert False, 'Unknown Scatterer type'

    def get_scatterer(self):
        return self._scatterer
        # if self._type == 'OpticalScatterer':
        #     return shdom.OpticalScatterer(wavelength=self._wavelength, extinction=self._extinction, albedo=self._albedo, phase=self._phase)
        # elif self._type == 'MicrophysicalScatterer':
        #     scatterer = shdom.MicrophysicalScatterer(lwc=self._lwc, reff=self._reff, veff=self._veff)
        #     if self.num_wavelengths > 0:
        #         scatterer.add_mie(self._mie[self._wavelength])
        #     return scatterer
        # else:
        #     assert 'Unknown Scatterer type'
        #     return None
    @property
    def scatterer(self):
        return self._scatterer

    @property
    def time(self):
        return self._time

    @property
    def type(self):
        return self._type


class DynamicScatterer(object):
    # TODO
    def __init__(self):
        self._num_scatterers = 0
        self._wavelength = []
        self._temporary_scatterer_list = []
        self._time_list = []
        self._type = None

    def get_velocity(self):
        assert self._temporary_scatterer_list is not None and self._num_scatterers >= 1,\
            'Dynamic Scatterer should have more than 1 scatterer'
        scatterer_location = []
        for temporary_scatterer in self._temporary_scatterer_list:
            scatterer = temporary_scatterer.get_scatterer()
            scatterer_location.append([scatterer.grid.x[0],scatterer.grid.y[0],scatterer.grid.z[0]])
        scatterer_location = np.asarray(scatterer_location)
        time_list = np.asarray(self._time_list).reshape((-1,1))
        scatterer_velocity_list = (scatterer_location[1:,:] - scatterer_location[:-1,:]) / (time_list[1:] - time_list[:-1])
        return scatterer_velocity_list

    def add_temporary_scatterer(self, temporary_scatterer_list):
        if not isinstance(temporary_scatterer_list,list):
            temporary_scatterer_list =[temporary_scatterer_list]
        assert all(isinstance(temporary_scatterer, TemporaryScatterer)
                   for temporary_scatterer in temporary_scatterer_list)
        'Type of Temporary scatterers should be TemporaryScatterer'
        for temporary_scatterer in temporary_scatterer_list:
            first_scatterer = True if self._num_scatterers == 0 else False
            if first_scatterer:
                self._wavelength = temporary_scatterer.scatterer.wavelength
                self._type = temporary_scatterer.type
            else:
                assert np.allclose(self.wavelength,
                                   temporary_scatterer.scatterer.wavelength), ' Dynamic Scatterer wavelength {} differs from temporary scatterer wavelength {}'.format(
                    self.wavelength, temporary_scatterer.scatterer.wavelength)
                assert self.type == temporary_scatterer.type, ' Dynamic Scatterer type {} differs from temporary scatterer type {}'.format(
                    self.type, temporary_scatterer.type)
            self._num_scatterers += 1
            self._temporary_scatterer_list.append(temporary_scatterer)
            self._time_list.append(temporary_scatterer.time)
            assert len(self._time_list) == len(set(self._time_list)), \
                ' Dynamic Scatterer is already defined for time = {}'.format(temporary_scatterer.time)

    def get_dynamic_optical_scatterer(self, wavelength):
        scatterer_list = []

        for temporary_scatterer in self._temporary_scatterer_list:

            if self.type == 'MicrophysicalScatterer':
                scatterer_list.append(TemporaryScatterer(temporary_scatterer.scatterer.get_optical_scatterer(wavelength),temporary_scatterer.time))
            elif self.type== 'OpticalScatterer' or self._type == 'MultispectralScatterer':
                scatterer_list.append(
                    TemporaryScatterer(temporary_scatterer.scatterer, temporary_scatterer.time))
            else:
                assert 'Unknown Scatterer type'

        dynamic_optical_scatterer = DynamicScatterer()
        dynamic_optical_scatterer.add_temporary_scatterer(scatterer_list)

        return dynamic_optical_scatterer

    def generate_dynamic_scatterer(self, scatterer, time_list, scatterer_velocity_list):
        time_list = np.asarray(time_list).reshape((-1, 1))
        scatterer_velocity_list = np.asarray(scatterer_velocity_list).reshape((3, -1))
        assert scatterer_velocity_list.shape[0] == 3 and \
               (scatterer_velocity_list.shape[1] == time_list.shape[0] or scatterer_velocity_list.shape[1] == 1),\
            'time_list, scatterer_velocity_list have wrong dimensions'
        scatterer_shifts = 1e-3 * time_list * scatterer_velocity_list.T #km
        assert isinstance(scatterer, shdom.Scatterer), 'scatterer is not a Scatterer object'
        self._num_scatterers = 0
        self._wavelength = scatterer.wavelength
        self._temporary_scatterer_list = []
        self._time_list = []
        self._type = None

        for scatterer_shift, time in zip(scatterer_shifts, time_list):
            shifted_scatterer = self.shift_scatterer(scatterer, scatterer_shift)
            temporary_scatterer = TemporaryScatterer(shifted_scatterer, time)

            self._temporary_scatterer_list.append(temporary_scatterer)
            self._num_scatterers += 1
            self._time_list.append(float(time))

    def shift_scatterer(self, scatterer, scatterer_shift):
        if isinstance(scatterer, shdom.MicrophysicalScatterer):
            shifted_scatterer = shdom.MicrophysicalScatterer()
            assert scatterer.grid.type == '3D', 'Scatterer grid type has to be 3D'
            grid_lwc = shdom.Grid(x=scatterer.grid.x + scatterer_shift[0], y=scatterer.grid.y + scatterer_shift[1],
                                  z=scatterer.grid.z + scatterer_shift[2])
            if scatterer.reff.type == '3D':
                grid_reff = grid_lwc
            else:
                grid_reff = scatterer.reff.grid
            if scatterer.veff.type == '3D':
                grid_veff = grid_lwc
            else:
                grid_veff = scatterer.veff.grid
            shifted_scatterer.set_microphysics(
                lwc=shdom.GridData(grid_lwc, scatterer.lwc.data).squeeze_dims(),
                reff=shdom.GridData(grid_reff, scatterer.reff.data).squeeze_dims(),
                veff=shdom.GridData(grid_veff, scatterer.veff.data).squeeze_dims()
            )
            if scatterer.num_wavelengths > 1:
                shifted_scatterer.add_mie(scatterer.mie)
            else:
                shifted_scatterer.add_mie(scatterer.mie[scatterer.wavelength])
            self._type = 'MicrophysicalScatterer'
        elif isinstance(scatterer, shdom.OpticalScatterer):
            grid_extinction = shdom.Grid(x=scatterer.grid.x + scatterer_shift[0], y=scatterer.grid.y +
                                                                                    scatterer_shift[1],
                                         z=scatterer.grid.z + scatterer_shift[2])
            if scatterer.albedo.type == '3D':
                grid_albedo = grid_extinction
            else:
                grid_albedo = scatterer.albedo.grid
            if scatterer.phase.type == '3D':
                grid_phase = grid_extinction
            else:
                grid_phase = scatterer.phase.grid
            shifted_scatterer = shdom.OpticalScatterer(wavelength=scatterer.wavelength,
                                                       extinction=shdom.GridData(grid_extinction,
                                                                                 scatterer.extinction.data).squeeze_dims(),
                                                       albedo=shdom.GridData(grid_albedo,
                                                                             scatterer.albedo.data).squeeze_dims(),
                                                       phase=shdom.GridPhase(scatterer.phase.legendre_table, grid_phase)
                                                       )
        else:
            assert 'Scatterer type is not supported'
        return shifted_scatterer

    def get_mask(self, threshold):
        """
        Get a mask based on the optical extinction.

        Parameters
        ----------
        threshold: float
            A threshold which above this value it is considered a populated voxel.

        Returns
        -------
        mask: shdom.GridData object
            A boolean mask with True for dense voxels and False for optically thin regions.
        """
        first_mask = True
        for temporal_scatterer in self.temporary_scatterer_list:
            scatterer = temporal_scatterer.scatterer
            if self._type == 'MicrophysicalScatterer':
                mask = scatterer.lwc.data > threshold
            elif self._type == 'OpticalScatterer':
                mask = scatterer.extinction.data > threshold
            elif self._type == 'MultispectralScatterer':
                mask = None
                for _, optical_scatterer in scatterer.scatterer.items():
                    if mask is None:
                        mask = optical_scatterer.extinction.data > threshold
                    else:
                        mask = (optical_scatterer.extinction.data > threshold) | mask
            else:
                assert 'Scatterer type is not supported'
            if first_mask:
                joint_mask = mask
                first_mask = False
            else:
                joint_mask = joint_mask | mask

        mask_list = [shdom.GridData(scatterer.grid, joint_mask)] * self.num_scatterers
        return mask_list

    def get_albedo(self):
        if self._type == 'MicrophysicalScatterer':
            dynamic_scatterer = self.get_dynamic_optical_scatterer(self._wavelength)
        elif self._type == 'OpticalScatterer':
            dynamic_scatterer = self
        else:
            assert 'Scatterer type is not supported'
        albedo_list = []
        for temporal_scatterer in dynamic_scatterer.temporary_scatterer_list:
            scatterer = temporal_scatterer.scatterer
            albedo_list.append(scatterer.albedo)
        return albedo_list

    def get_phase(self):
        if self._type == 'MicrophysicalScatterer':
            dynamic_scatterer = self.get_dynamic_optical_scatterer(self._wavelength)
        elif self._type == 'OpticalScatterer':
            dynamic_scatterer = self
        else:
            assert 'Scatterer type is not supported'
        phase_list = []
        for temporal_scatterer in dynamic_scatterer.temporary_scatterer_list:
            scatterer = temporal_scatterer.get_scatterer()
            phase_list.append(scatterer.phase)
        return phase_list

    def get_extinction(self,dynamic_grid=None):
        if self._type == 'MicrophysicalScatterer':
            dynamic_scatterer = self.get_dynamic_optical_scatterer(self._wavelength)
        elif self._type == 'OpticalScatterer':
            dynamic_scatterer = self
        else:
            assert 'Scatterer type is not supported'
        extinction_list = []
        for i, temporal_scatterer in enumerate(dynamic_scatterer.temporary_scatterer_list):
            scatterer = temporal_scatterer.get_scatterer()
            data = scatterer.extinction.data
            grid = scatterer.extinction.grid
            extinction = shdom.GridData(grid, data)
            if dynamic_grid is not None:
                extinction = extinction.resample(dynamic_grid[i])
            extinction_list.append(extinction)
        return extinction_list

    def get_grid(self):
        grid_list = []
        for i, temporal_scatterer in enumerate(self._temporary_scatterer_list):
            scatterer = temporal_scatterer.get_scatterer()
            grid_list.append(scatterer.grid)
        return grid_list

    def get_lwc(self,dynamic_grid=None):
        if not self._type == 'MicrophysicalScatterer':
            assert 'Scatterer type has no LWC attribute'
        lwc_list = []
        grid_list = []
        for i, temporal_scatterer in enumerate(self._temporary_scatterer_list):
            scatterer = temporal_scatterer.get_scatterer()
            data = scatterer.lwc.data
            grid = scatterer.lwc.grid
            lwc = shdom.GridData(grid, data)
            if dynamic_grid is not None:
                lwc = lwc.resample(dynamic_grid[i])
            lwc_list.append(lwc)
            grid_list.append(lwc.grid)
        return lwc_list

    def get_reff(self,dynamic_grid=None):
        if not self._type == 'MicrophysicalScatterer':
            assert 'Scatterer type has no reff attribute'
        reff_list = []
        grid_list = []
        for i, temporal_scatterer in enumerate(self._temporary_scatterer_list):
            scatterer = temporal_scatterer.get_scatterer()
            data = scatterer.reff.data
            grid = scatterer.reff.grid
            reff = shdom.GridData(grid, data)
            if dynamic_grid is not None:
                reff = reff.resample(dynamic_grid[i])
            reff_list.append(reff)
            grid_list.append(reff.grid)

        return reff_list

    def get_veff(self,dynamic_grid=None):
        if not self._type == 'MicrophysicalScatterer':
            assert 'Scatterer type has no veff attribute'
        veff_list = []
        grid_list = []
        for i, temporal_scatterer in enumerate(self._temporary_scatterer_list):
            scatterer = temporal_scatterer.get_scatterer()
            data = scatterer.veff.data
            grid = scatterer.veff.grid
            veff = shdom.GridData(grid, data)
            if dynamic_grid is not None:
                veff = veff.resample(dynamic_grid[i])
            veff_list.append(veff)
            grid_list.append(veff.grid)

        return veff_list

    def __getitem__(self, val):
        return self._temporary_scatterer_list[val]

    def pop_temporary_scatterer(self, index):
        """
        Pop a Medium from the Dynamic Medium.

        Parameters
        ----------
        index: int

        """
        assert isinstance(index,int) and index >=0 and index < self.num_scatterers
        self._num_scatterers -= 1
        temporary_scatterer = self.temporary_scatterer_list[index]
        del self._temporary_scatterer_list[index]
        return temporary_scatterer

    @property
    def type(self):
        return self._type

    @property
    def num_scatterers(self):
        return self._num_scatterers

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def time_list(self):
        return self._time_list

    @property
    def temporary_scatterer_list(self):
        if self.num_scatterers == 0:
            return None
        else:
            return self._temporary_scatterer_list


class DynamicMedium(object):
    # TODO
    def __init__(self, dynamic_scatterer=None, air=None):
        self._num_mediums = 0
        self._wavelength = []
        self._medium_list = []
        self._time_list = []
        self._dynamic_scatterer = None
        if dynamic_scatterer is not None and air is not None:
            self.set_medium_list(dynamic_scatterer,air)

    def set_medium_list(self, dynamic_scatterer, air):
        assert isinstance(dynamic_scatterer,DynamicScatterer) #and (isinstance(air,shdom.Scatterer) or isinstance(air,shdom.MultispectralScatterer))
        self._num_mediums = 0
        self._medium_list = []
        self._time_list = []
        self._dynamic_scatterer = dynamic_scatterer
        temporary_scatterer_list = dynamic_scatterer.temporary_scatterer_list
        for temporary_scatterer, time in zip(temporary_scatterer_list, dynamic_scatterer.time_list):
            scatterer = temporary_scatterer.get_scatterer()
            first_scatterer = True if self._num_mediums == 0 else False
            if first_scatterer:
                self._wavelength = scatterer.wavelength
            else:
                assert np.allclose(self.wavelength,
                                   scatterer.wavelength), 'medium wavelength {} differs from dynamic_scatterers wavelength {}'.format(
                    self.wavelength, scatterer.wavelength)
            atmospheric_grid = scatterer.grid + air.grid
            atmosphere = shdom.Medium(atmospheric_grid)
            atmosphere.add_scatterer(scatterer, name='cloud_at_time_{}'.format(time))
            atmosphere.add_scatterer(air, name='air')
            self._medium_list.append(atmosphere)
            self._num_mediums += 1
            self._time_list.append(time)

    def get_dynamic_scatterer(self):
        return self._dynamic_scatterer

    def add_medium(self, medium):
        """
        Add a Medium to the Dynamic Medium.

        Parameters
        ----------
        medium: shdom.Medium

        """
        first_medium = True if self.num_mediums == 0 else False

        if first_medium:
            self._wavelength = medium.wavelength
        else:
            assert np.allclose(self.wavelength,
                               medium.wavelength), ' medium wavelength {} differs from scatterer wavelength {}'.format(
                self.wavelength, medium.wavelength)
        self._num_mediums += 1
        self._medium_list.append(medium)

    def pop_medium(self, index):
        """
        Pop a Medium from the Dynamic Medium.

        Parameters
        ----------
        index: int

        """
        assert isinstance(index,int) and index >=0 and index < self.num_mediums
        self._num_mediums -= 1
        medium = DynamicMedium()
        medium.add_medium(self._medium_list[index])
        del self._medium_list[index]

        return medium

    def __getitem__(self, val):
        return self._medium_list[val]

    def save(self, path):
        """
        Save DynamicMedium parameters to file.

        Parameters
        ----------
        path: str,
            Full path to file.
        """
        file = open(path, 'wb')
        file.write(pickle.dumps(self.__dict__, -1))
        file.close()

    def load(self, path):
        """
        Load RteSolverArray parameters from file.

        Parameters
        ----------
        path: str,
            Full path to file.
        """
        file = open(path, 'rb')
        data = file.read()
        file.close()
        self.__dict__ = pickle.loads(data)

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def num_mediums(self):
        return self._num_mediums

    @property
    def dynamic_scatterer(self):
        return self._dynamic_scatterer

    @property
    def time_list(self):
        return self._time_list

    @property
    def medium_list(self):
        if self.num_mediums == 0:
            return None
        else:
            return self._medium_list

    @medium_list.setter
    def medium_list(self, val):
        assert isinstance(val, list), 'medium_list is not a list'
        self._medium_list = val


class DynamicRteSolver(object):
    def __init__(self, scene_params=None, numerical_params=None):
        if isinstance(scene_params,list):
            self._scene_params = scene_params
        else:
            self._scene_params = [[scene_params]]
        if isinstance(scene_params,list):
            self._numerical_params = numerical_params
        else:
            self._numerical_params = [[numerical_params]]
        self._solver_array_list = []
        self._dynamic_medium = None
        self._wavelength = []
        self._maxiters = 0

    def set_dynamic_medium(self, dynamic_medium, num_stokes_list=None):
        assert isinstance(dynamic_medium, DynamicMedium) or isinstance(dynamic_medium, DynamicMediumEstimator), ' dynamic_medium type is wrong'
        self._dynamic_medium = dynamic_medium
        self._solver_array_list = []
        if isinstance(dynamic_medium.wavelength,list):
            self._wavelength = dynamic_medium.wavelength
        else:
            self._wavelength = [dynamic_medium.wavelength]

        medium_list = dynamic_medium.medium_list
        assert len(self._scene_params)==len(self._numerical_params)==1 or\
               len(self._scene_params)==len(self._numerical_params)==len(medium_list)

        scene_params = self._scene_params
        numerical_params = self._numerical_params
        if len(self._scene_params) == 1:
            scene_params *= len(medium_list)
        if len(numerical_params) == 1:
            numerical_params *= len(medium_list)
        if num_stokes_list is None:
            num_stokes_list = [1]*len(self._wavelength)
        for medium, scene_params_wls, numerical_params_wls in zip(medium_list, scene_params, numerical_params):
            medium_solver_list = []
            assert isinstance(scene_params_wls, list) and isinstance(numerical_params_wls, list)
            assert len(scene_params_wls)==len(numerical_params_wls)==len(self._wavelength)==len(num_stokes_list)
            for scene_params_wl, numerical_params_wl, wl,n_stokes in zip(scene_params_wls, numerical_params_wls,
                                                                self._wavelength,num_stokes_list):
                assert wl == scene_params_wl.wavelength
                rte_solver = shdom.RteSolver(scene_params_wl,numerical_params_wl,num_stokes=n_stokes)
                medium_solver_list.append(rte_solver)
            solver_array = shdom.RteSolverArray(medium_solver_list)
            solver_array.set_medium(medium)
            self._solver_array_list.append(solver_array)

    def replace_dynamic_medium(self, dynamic_medium):
        assert isinstance(dynamic_medium, DynamicMedium) or isinstance(dynamic_medium, DynamicMediumEstimator), ' dynamic_medium type is wrong'
        self._dynamic_medium = dynamic_medium

        if isinstance(dynamic_medium.wavelength,list):
            wavelength = dynamic_medium.wavelength
        else:
            wavelength = [dynamic_medium.wavelength]
        assert np.allclose(wavelength, self.wavelength)
        assert dynamic_medium.num_mediums == self.num_solver_arrays
        for solver_array, medium in zip(self.solver_array_list, dynamic_medium.medium_list):
            solver_array.set_medium(medium)

    def get_param_dict(self):
        """
        Retrieve a dictionary with the solver array parameters

        Returns
        -------
        param_dict: dict
            The parameters dictionary contains: type and a list of scene_params, numerical_params matching all the solvers
        """
        param_dict = {}
        for key, param in self.__dict__.items():
            if key != '_solver_array_list':
                param_dict[key] = param
        return param_dict

    def set_param_dict(self, param_dict):
        """
        Set the solver array parameters from a parameter dictionary

        Parameters
        ----------
        param_dict: dict
            The parameters dictionary contains: type and a list of scene_params, numerical_params matching all the solvers
        """
        self._scene_params = param_dict['_scene_params']
        self._numerical_params = param_dict['_numerical_params']
        self.set_dynamic_medium(param_dict['_dynamic_medium'])

    def save_params(self, path):
        """
        Save RteSolverArray parameters to file.

        Parameters
        ----------
        path: str,
            Full path to file.
        """
        file = open(path, 'wb')
        file.write(pickle.dumps(self.get_param_dict(), -1))
        file.close()

    def load_params(self, path):
        """
        Load RteSolverArray parameters from file.

        Parameters
        ----------
        path: str,
            Full path to file.
        """
        file = open(path, 'rb')
        data = file.read()
        file.close()
        params = pickle.loads(data)
        self.set_param_dict(params)

    def __getitem__(self, val):
        return self.solver_array_list[val]

    def init_solution(self):
        Parallel(n_jobs=self.num_solver_arrays, backend="threading")(
                delayed(solver_array.init_solution, check_pickle=False)()
                for solver_array in self.solver_array_list)

    def solve(self, maxiter, init_solution=False, verbose=False):
        """
        Parallel solving of all solvers.

        Main solver routine. This routine is comprised of two parts:
          1. Initialization (init_solution method), optional
          2. Parallel solution iterations (solution_iterations method followed by update_solution_arguments method)

        Parameters
        ----------
        maxiter: integer
            Maximum number of iterations for the iterative solution.
        init_solution: boolean, default=False
            If False the solution is initialized according to the existing radiance and source function saved within the RteSolver object (previously computed)
            If True or no prior solution (I,J fields) exists then an initialization is preformed (part 1.).
        verbose: boolean
            True will output solution iteration information into stdout.
        """
        # for solver in self.solver_array_list:
        #     if init_solution or solver.num_iterations == 0:
        #         solver.init_solution()

        Parallel(n_jobs=self.num_solver_arrays, backend="threading")(
                delayed(solver_array.solve, check_pickle=False)(
                    maxiter, init_solution=init_solution, verbose=verbose) for solver_array in self.solver_array_list)
        #
        # for solver, arguments in zip(self.solver_list, output_arguments):
        #     solver.update_solution_arguments(arguments)

        self._maxiters = max([solver_array.num_iterations for solver_array in self.solver_array_list])


    def pop_solver(self, index):
        """
        Pop a RTE solver from the Dynamic Rte Solver.

        Parameters
        ----------
        index: int

        """
        assert isinstance(index,int) and index >=0 and index < self.num_solver_arrays
        dynamic_medium = self.dynamic_medium.pop_medium(index)
        rte_solver = DynamicRteSolver(self._scene_params, self._numerical_params)
        rte_solver.set_dynamic_medium(dynamic_medium)
        return rte_solver
    @property
    def dynamic_medium(self):
        return self._dynamic_medium

    @property
    def solver_array_list(self):
        return self._solver_array_list

    @property
    def num_solver_arrays(self):
        return len(self.solver_array_list)

    @property
    def num_iterations(self):
        return self._maxiters

    @property
    def wavelength(self):
        if len(self._wavelength) == 1:
            return self._wavelength[0]
        else:
            return self._wavelength



    # def add_dynamic_solver(self, rte_solver):
    #     """
    #     Add an rte_solver or solvers to the RteSolverArray
    #
    #     Parameters
    #     ----------
    #     rte_solver: RteSolver object or list of RteSolvers or RteSolverArray
    #         Add RteSolver or solvers to the RteSolverArray
    #     """
    #
    #     if self.type is None:
    #         self._type = rte_solver.type
    #     else:
    #         assert self.type == rte_solver.type, \
    #             '[add_solver] Assert: RteSolverArray is of type {} and new solver is of type {}'.format(self.type,
    #                                                                                                     rte_solver.type)
    #
    #     if isinstance(rte_solver, shdom.RteSolver):
    #         self._solver_list.append(rte_solver)
    #         self._name.append(rte_solver.name)
    #         self._num_solvers += 1
    #     else:
    #         for solver in rte_solver:
    #             self._solver_list.append(solver)
    #             self._name.append(solver.name)
    #             self._num_solvers += 1


class DynamicProjection(shdom.Projection):
    """
    A MultiViewProjection object encapsulate several projection geometries for multi-view imaging of a domain.

    Parameters
    ----------
    projection_list: list, optional
        A list of Sensor objects
    """
    def __init__(self, projection_list=None):
        super().__init__()
        self._num_viewed_medium = 0
        self._multiview_projection_list = []
        self._names = []
        if projection_list:
            for projection in projection_list:
                self.add_projection(projection)

    def add_projection(self, projection, name=None):
        """
        Add a projection to the projection list

        Parameters
        ----------
        projection: Projection object
            A Projection object to add to the MultiViewProjection
        name: str, optional
            An ID for the projection.
        """
        # Set a default name for the projection
        if name is None:
            name = 'Viewed_Medium_Projections{}'.format(self.num_viewed_medium)

        attributes = ['x', 'y', 'z', 'mu', 'phi']

        if self._num_viewed_medium == 0:
            for attr in attributes:
                self.__setattr__('_' + attr, projection.__getattribute__(attr))
            self._npix = [projection.npix]
            self._resolution = [projection.resolution]
            self._names = [name]
        else:
            for attr in attributes:
                self.__setattr__('_' + attr, np.concatenate((self.__getattribute__(attr),
                                                             projection.__getattribute__(attr))))
            self._npix.append(projection.npix)
            self._names.append(name)
            self._resolution.append(projection.resolution)
        if not isinstance(projection,shdom.MultiViewProjection):
            projection = shdom.MultiViewProjection([projection])
        self._multiview_projection_list.append(projection)
        self._num_viewed_medium += 1

    def get_flatten_projections(self):
        projection_list = []
        for multiview_projection in self.multiview_projection_list:
            projection_list += multiview_projection.projection_list
        return projection_list

    @property
    def multiview_projection_list(self):
        return self._multiview_projection_list

    @property
    def num_viewed_medium(self):
        return self._num_viewed_medium


class DynamicCamera(object):
    def __init__(self, sensor=shdom.Sensor(), dynamic_projection=DynamicProjection()):
        """
        An DynamicCamera object ecapsulates both sensor and projection for Dynamic camera.

        Parameters
        ----------
        sensor: shdom.Sensor
            A sensor object
        projection: shdom.DynamicProjection
            A Dynamic Projection object
        """
        self.set_sensor(sensor)
        self.set_dynamic_projection(dynamic_projection)

    def set_dynamic_projection(self, dynamic_projection):
        """
        Add a projection.

        Parameters
        ----------
        projection: shdom.Projection
            A projection geomtry
        """
        self._dynamic_projection = dynamic_projection

    def set_sensor(self, sensor):
        """
        Add a sensor.

        Parameters
        ----------
        sensor: shdom.Sensor
            A sensor object

        Notes
        -----
        This method also updates the docstring of the render method according to the specific sensor
        """
        self._sensor = sensor

        # Update function docstring
        if sensor.render.__doc__ is not None:
            self.render.__func__.__doc__ += sensor.render.__doc__

    def render(self, dynamic_solver, n_jobs=1, verbose=0):
        """
        Render an image according to the render function defined by the sensor.

        Notes
        -----
        This is a dummy docstring that is overwritten when the set_sensor method is used.
        """
        assert isinstance(dynamic_solver, DynamicRteSolver)
        images=[]

        # rte_solver_array = dynamic_solver.get_rte_solver_array()


        for solver_array, multiview_projection in zip(dynamic_solver.solver_array_list, self._dynamic_projection.multiview_projection_list):
            # for projection in self.projection.projection_list:
            images += (self.sensor.render(solver_array, multiview_projection, n_jobs, verbose))
        return images

    # def projection_split(self, n_parts):
    #     avg = len(self.projection.projection_list) / float(n_parts)
    #     out = []
    #     last = 0.0
    #
    #     while last < len(self.projection.projection_list):
    #         out.append(shdom.MultiViewProjection(self.projection.projection_list[int(last):int(last + avg)]))
    #         last += avg
    #
    #     return out

    @property
    def dynamic_projection(self):
        return self._dynamic_projection

    @property
    def sensor(self):
        return self._sensor

    @property
    def num_viewed_medium(self):
        return self.dynamic_projection.num_viewed_medium


class DynamicMeasurements(shdom.Measurements):
    def __init__(self, camera=None, images=None, pixels=None, wavelength=None, uncertainties=None, time_list=None, masks=None):
        super().__init__(camera=camera, images=images, pixels=pixels, wavelength=wavelength, uncertainties=uncertainties)
        assert (images is None) == (time_list is None),'images and time_list have to be None or not'
        # if images is not None and  time_list is not None:
            # assert len(images) == len(time_list*camera.projection), 'images and time_list have to be with the same length'
        self._time_list = time_list
        self._mask_list = masks

        if camera is not None:
            self._num_viewed_mediums = camera.num_viewed_medium
        else:
            self._num_viewed_mediums = 0

    def split(self, split_indices):
        """
        Split the measurements and projection.

        Returns
        -------
        measurements: list
            A list of measurements each with n_parts

        Notes
        -----
        An even split doesnt always exist, in which case some parts will have slightly more pixels.
        """
        avg = len(split_indices) / float(self._num_viewed_mediums)
        out = []
        last = 0.0

        while last < len(split_indices):
            out.append((split_indices[int(last + avg)-1]))
            last += avg
        pixels = np.array_split(self.pixels, out[:-1], axis=1)
        if self.uncertainties is None:
            uncertainties = [None]* len(pixels)
        else:
            uncertainties = np.array_split(self.uncertainties, out[:-1], axis=-2)
        projections = self.camera.dynamic_projection.multiview_projection_list

        if self._mask_list is None:

            masks = [None]*len(pixels)
        else:
            pix_mask = np.array([],dtype=bool)
            for mask in self._mask_list:
                pix_mask = np.hstack((pix_mask,mask.ravel(order='F')))
            masks = np.array_split(np.array(pix_mask), out[:-1])

        measurements = [shdom.Measurements(
            camera=shdom.Camera(self.camera.sensor, projection), wavelength=self.wavelength,
            pixels=pixel, uncertainties=uncertainty) for projection, pixel,uncertainty in zip(projections, pixels, uncertainties)
        ]
        return measurements, masks

    def downsample_viewed_mediums(self, new_num_viewed_mediums):
        assert self.num_viewed_mediums >= new_num_viewed_mediums
        if new_num_viewed_mediums == self.num_viewed_mediums:
            return DynamicMeasurements(self.camera, self.images, self.pixels, self.wavelength, self.uncertainties, self.time_list, self._mask_list)

        dynamic_projection = shdom.DynamicProjection(self.projection_split(new_num_viewed_mediums))
        dynamic_camera = shdom.DynamicCamera(self.camera.sensor, dynamic_projection)
        time_list = np.mean(np.split(np.array(self.time_list), new_num_viewed_mediums), 1)
        return DynamicMeasurements(dynamic_camera, images=self.images, wavelength=self.wavelength,
                                   uncertainties= self.uncertainties, time_list=time_list, masks=self._mask_list)

    def projection_split(self, n_parts):
        projection_list = []
        for multiview_projection in self.camera.dynamic_projection.multiview_projection_list:
            projection_list += (multiview_projection.projection_list)
        avg = len(projection_list) / float(n_parts)
        out = []
        last = 0.0

        while last < len(projection_list):
            out.append(shdom.MultiViewProjection(projection_list[int(last):int(last + avg)]))
            last += avg
        return out

    def get_cross_validation_measurements(self, cv_index):
        num_projections = len(self.camera.dynamic_projection.multiview_projection_list)
        assert isinstance(cv_index,int) and cv_index >= 0 and cv_index < num_projections
        projections_list = self.camera.dynamic_projection.multiview_projection_list
        projections = [x for i, x in enumerate(projections_list) if i != cv_index]
        camera = DynamicCamera(self.camera.sensor, DynamicProjection(projections))
        images = [x for i, x in enumerate(self.images) if i != cv_index]
        time_list = [x for i, x in enumerate(self.time_list) if i != cv_index]
        measurements = DynamicMeasurements(camera=camera, images=images, wavelength=self.wavelength,
                                           time_list=time_list, masks=self._mask_list)
        camera = DynamicCamera(self.camera.sensor, DynamicProjection([projections_list[cv_index]]))
        image = [self.images[cv_index]]
        time = [self.time_list[cv_index]]
        cv_measurement = DynamicMeasurements(camera=camera, images=image, wavelength=self.wavelength,
                                           time_list=time, masks=self._mask_list)
        return cv_measurement, measurements


    @property
    def time_list(self):
        return self._time_list

    @property
    def num_viewed_mediums(self):
        return self._num_viewed_mediums


class DynamicHybridMeasurements(shdom.HybridMeasurements):
    def __init__(self, camera=None, images=None, pixels=None, wavelength=None, uncertainties=None, time_list=None):
        super().__init__(camera=camera, images=images, pixels=pixels, wavelength=wavelength,
                         uncertainties=uncertainties)
        assert (images is None) == (time_list is None), 'images and time_list have to be None or not'
        # if images is not None and  time_list is not None:
        # assert len(images) == len(time_list*camera.projection), 'images and time_list have to be with the same length'
        self._time_list = time_list
        # if camera is not None:
        #     self._num_viewed_mediums = camera.num_viewed_medium
        # else:
        self._num_viewed_mediums = 0

    def split(self, split_indices):
        """
        Split the measurements and projection.

        Returns
        -------
        measurements: list
            A list of measurements each with n_parts

        Notes
        -----
        An even split doesnt always exist, in which case some parts will have slightly more pixels.
        """
        # assert self._num_viewed_mediums > 0
        measurements =[]
        for pix, cam in zip(self.pixels, self.camera):
            pixels = np.array_split(pix, split_indices,axis=1)
            projections = cam.dynamic_projection.multiview_projection_list
            measurements.append([shdom.Measurements(
                camera=shdom.Camera(self.camera.sensor, projection),wavelength=self.wavelength,
                pixels=pixel) for projection, pixel in zip(projections, pixels)
            ])
        return measurements


    @property
    def time_list(self):
        return self._time_list

    @property
    def num_viewed_mediums(self):
        return self._num_viewed_mediums


class Homogeneous(shdom.CloudGenerator):
    """
    Define a homogeneous Medium.

    Parameters
    ----------
    args: arguments from argparse.ArgumentParser()
        Arguments required for this generator.
    """
    def __init__(self, args):
        super(Homogeneous, self).__init__(args)

    @classmethod
    def update_parser(self, parser):
        """
        Update the argument parser with parameters relevant to this generator.

        Parameters
        ----------
        parser: argparse.ArgumentParser()
            The main parser to update.

        Returns
        -------
        parser: argparse.ArgumentParser()
            The updated parser.
        """
        parser.add_argument('--nx',
                            default=20,
                            type=int,
                            help='(default value: %(default)s) Number of grid cell in x (North) direction')
        parser.add_argument('--ny',
                            default=20,
                            type=int,
                            help='(default value: %(default)s) Number of grid cell in y (East) direction')
        parser.add_argument('--nz',
                            default=20,
                            type=int,
                            help='(default value: %(default)s) Number of grid cell in z (Up) direction')
        parser.add_argument('--domain_size',
                            default=2.0,
                            type=float,
                            help='(default value: %(default)s) Cubic domain size [km]')
        parser.add_argument('--extinction',
                            default=1.0,
                            type=np.float32,
                            help='(default value: %(default)s) Extinction [km^-1]')
        parser.add_argument('--lwc',
                            default=None,
                            type=np.float32,
                            help='(default value: %(default)s) Liquid water content of the center voxel [g/m^3]. If specified, extinction argument is ignored.')
        parser.add_argument('--reff',
                            default=10.0,
                            type=np.float32,
                            help='(default value: %(default)s) Effective radius [micron]')
        parser.add_argument('--veff',
                            default=0.1,
                            type=np.float32,
                            help='(default value: %(default)s) Effective variance')
        parser.add_argument('--time_list',
                            default=[0]*9,
                            type=float,
                            help='(default value: %(default)s) Effective variance')
        parser.add_argument('--cloud_velocity',
                            default=[0,0,0],
                            type=np.float32,
                            help='Estimated cloud velocity.')
        return parser

    def get_grid(self):
        """
        Retrieve the scatterer grid.

        Returns
        -------
        grid: shdom.Grid
            A Grid object for this scatterer
        """

        bb = shdom.BoundingBox(0, 0, 0, self.args.domain_size, self.args.domain_size, self.args.domain_size)

        return shdom.Grid(bounding_box=bb,nx=self.args.nx,ny=self.args.ny,nz=self.args.nz)

    # def get_grid(self):
    #     """
    #     Retrieve the scatterer grid.
    #
    #     Returns
    #     -------
    #     grid: shdom.Grid
    #         A Grid object for this scatterer
    #     """
    #     time_list = np.asarray(self.args.time_list).reshape((-1, 1))
    #     scatterer_velocity_list = np.asarray(self.args.cloud_velocity).reshape((3, -1))
    #     assert scatterer_velocity_list.shape[0] == 3 and \
    #            (scatterer_velocity_list.shape[1] == time_list.shape[0] or scatterer_velocity_list.shape[1] == 1), \
    #         'time_list, scatterer_velocity_list have wrong dimensions'
    #     scatterer_shifts = 1e-3 * time_list * scatterer_velocity_list.T  # km
    #     bb = shdom.BoundingBox(0.0, 0.0, 0.0, self.args.domain_size, self.args.domain_size, self.args.domain_size)
    #     grid_list = []
    #     for scatterer_shift in scatterer_shifts:
    #         grid_list.append(shdom.Grid(bounding_box=bb,
    #             x=np.linspace(scatterer_shift[1], self.args.nx+ scatterer_shift[0], self.args.nx),
    #                    y=np.linspace(scatterer_shift[1], self.args.ny + scatterer_shift[1], self.args.ny),
    #                    z=np.linspace(0.1 + scatterer_shift[2], self.args.nz + scatterer_shift[2], self.args.nz)))
    #         # grid_list.append(shdom.Grid(bounding_box=bb, nx=self.args.nx + scatterer_shift[0], ny=self.args.ny+ scatterer_shift[1], nz=self.args.nz+ scatterer_shift[2]))
    #     return grid_list

    def get_extinction(self, wavelength=None, grid_list=None):
        """
        Retrieve the optical extinction at a single wavelength on a grid.

        Parameters
        ----------
        wavelength: float
            Wavelength in microns. A Mie table at this wavelength should be added prior (see add_mie method).
        grid: shdom.Grid, optional
            A shdom.Grid object. If None is specified than a grid is created from Arguments given to the generator (get_grid method)

        Returns
        -------
        extinction: shdom.GridData
            A GridData object containing the optical extinction on a grid

        Notes
        -----
        If the LWC is specified then the extinction is derived using (lwc,reff,veff). If not the extinction needs to be directly specified.
        The input wavelength is rounded to three decimals.
        """
        if grid_list is None:
            NotImplemented()
            grid_list = self.get_grid()
        extinction =[]
        if self.args.lwc is None:
            for grid in grid_list:
                if grid.type == 'Homogeneous':
                    ext_data = self.args.extinction
                elif grid.type == '1D':
                    ext_data = np.full(shape=(grid.nz), fill_value=self.args.extinction, dtype=np.float32)
                elif grid.type == '3D':
                    ext_data = np.full(shape=(grid.nx, grid.ny, grid.nz), fill_value=self.args.extinction, dtype=np.float32)
                extinction.append(shdom.GridData(grid, ext_data))
        else:
            assert wavelength is not None, 'No wavelength provided'
            lwc_list = self.get_lwc(grid_list)
            reff_list = self.get_reff(grid_list)
            veff_list = self.get_veff(grid_list)
            for lwc, reff, veff in zip(lwc_list,reff_list,veff_list):
                extinction.append(self.mie[shdom.float_round(wavelength)].get_extinction(lwc, reff, veff))
        return extinction

    def get_albedo(self, wavelength, grid_list=None):
        """
        Retrieve the single scattering albedo at a single wavelength on a grid.

        Parameters
        ----------
        wavelength: float
            Wavelength in microns. A Mie table at this wavelength should be added prior (see add_mie method).
        grid: shdom.Grid, optional
            A shdom.Grid object. If None is specified than a grid is created from Arguments given to the generator (get_grid method)

        Returns
        -------
        albedo: shdom.GridData
            A GridData object containing the single scattering albedo [0,1] on a grid

        Notes
        -----
        The input wavelength is rounded to three decimals.
        """
        if grid_list is None:
            NotImplemented()
            grid_list = self.get_grid()
        albedo =[]
        # if self.args.lwc is None:
        #     for grid in grid_list:
        #         if grid.type == 'Homogeneous':
        #             alb_data = self.args.albedo
        #         elif grid.type == '1D':
        #             alb_data = np.full(shape=(grid.nz), fill_value=self.args.albedo, dtype=np.float32)
        #         elif grid.type == '3D':
        #             alb_data = np.full(shape=(grid.nx, grid.ny, grid.nz), fill_value=self.args.albedo, dtype=np.float32)
        #         albedo.append(shdom.GridData(grid, alb_data))
        # else:
        assert wavelength is not None, 'No wavelength provided'
        reff_list = self.get_reff(grid_list)
        veff_list = self.get_veff(grid_list)
        for reff, veff in zip(reff_list,veff_list):
            albedo.append(self.mie[shdom.float_round(wavelength)].get_albedo(reff, veff))
        return albedo

    def get_phase(self, wavelength, grid_list=None):
        """
        Retrieve the single scattering albedo at a single wavelength on a grid.

        Parameters
        ----------
        wavelength: float
            Wavelength in microns. A Mie table at this wavelength should be added prior (see add_mie method).
        grid: shdom.Grid, optional
            A shdom.Grid object. If None is specified than a grid is created from Arguments given to the generator (get_grid method)

        Returns
        -------
        albedo: shdom.GridData
            A GridData object containing the single scattering albedo [0,1] on a grid

        Notes
        -----
        The input wavelength is rounded to three decimals.
        """
        if grid_list is None:
            NotImplemented()
            grid_list = self.get_grid()
        phase =[]
        # for grid in grid_list:
        #     if grid.type == 'Homogeneous':
        #         phase_data = self.args.phase
        #     elif grid.type == '1D':
        #         phase_data = np.full(shape=(grid.nz), fill_value=self.args.phase, dtype=np.float32)
        #     elif grid.type == '3D':
        #         phase_data = np.full(shape=(grid.nx, grid.ny, grid.nz), fill_value=self.args.phase, dtype=np.float32)
        #     phase.append(shdom.GridData(grid, phase_data))
        # else:
        assert wavelength is not None, 'No wavelength provided'
        reff_list = self.get_reff(grid_list)
        veff_list = self.get_veff(grid_list)
        for reff, veff in zip(reff_list,veff_list):
            phase.append(self.mie[shdom.float_round(wavelength)].get_phase(reff, veff))
        return phase

    def get_lwc(self, grid_list=None):
        """
        Retrieve the liquid water content.

        Parameters
        ----------
        grid: shdom.Grid, optional
            A shdom.Grid object. If None is specified than a grid is created from Arguments given to the generator (get_grid method)

        Returns
        -------
        lwc: shdom.GridData
            A GridData object containing liquid water content (g/m^3) on a 3D grid.
        """
        if grid_list is None:
            NotImplemented()
            grid_list = self.get_grid()

        lwc = self.args.lwc
        lwc_list =[]

        if lwc is not None:
            for grid in grid_list:
                if grid.type == 'Homogeneous':
                    lwc_data = self.args.lwc
                elif grid.type == '1D':
                    lwc_data = np.full(shape=(grid.nz), fill_value=self.args.lwc, dtype=np.float32)
                elif grid.type == '3D':
                    lwc_data = np.full(shape=(grid.nx, grid.ny, grid.nz), fill_value=self.args.lwc, dtype=np.float32)
                lwc_list.append(shdom.GridData(grid, lwc_data))
        return lwc_list

    def get_reff(self, grid_list=None):
        """
        Retrieve the effective radius on a grid.

        Parameters
        ----------
        grid: shdom.Grid, optional
            A shdom.Grid object. If None is specified than a grid is created from Arguments given to the generator (get_grid method)

        Returns
        -------
        reff: shdom.GridData
            A GridData object containing the effective radius [microns] on a grid
        """
        if grid_list is None:
            NotImplemented()
            grid_list = self.get_grid()

        reff = self.args.reff
        reff_list = []

        if reff is not None:
            for grid in grid_list:
                if grid.type == 'Homogeneous':
                    reff_data = self.args.reff
                elif grid.type == '1D':
                    reff_data = np.full(shape=(grid.nz), fill_value=self.args.reff, dtype=np.float32)
                elif grid.type == '3D':
                    reff_data = np.full(shape=(grid.nx, grid.ny, grid.nz), fill_value=self.args.reff, dtype=np.float32)
                reff_list.append(shdom.GridData(grid, reff_data))
        return reff_list

    def get_veff(self, grid_list=None):
        """
        Retrieve the effective radius on a grid.

        Parameters
        ----------
        grid: shdom.Grid, optional
            A shdom.Grid object. If None is specified than a grid is created from Arguments given to the generator (get_grid method)

        Returns
        -------
        reff: shdom.GridData
            A GridData object containing the effective radius [microns] on a grid
        """
        if grid_list is None:
            NotImplemented()
            grid_list = self.get_grid()

        veff = self.args.veff
        veff_list = []

        if veff is not None:
            for grid in grid_list:
                if grid.type == 'Homogeneous':
                    veff_data = self.args.veff
                elif grid.type == '1D':
                    veff_data = np.full(shape=(grid.nz), fill_value=self.args.veff, dtype=np.float32)
                elif grid.type == '3D':
                    veff_data = np.full(shape=(grid.nx, grid.ny, grid.nz), fill_value=self.args.veff, dtype=np.float32)
                veff_list.append(shdom.GridData(grid, veff_data))
        return veff_list


class Static(shdom.CloudGenerator):
    """
    Define a Dynamic Medium from Static Medium.

    Parameters
    ----------
    args: arguments from argparse.ArgumentParser()
        Arguments required for this generator.
    """
    def __init__(self, args):
        super(Static, self).__init__(args)
        self._data = sio.loadmat((glob.glob(((self.args.load_path) + '/' + '*.mat')))[0])


    @classmethod
    def update_parser(self, parser):
        """
        Update the argument parser with parameters relevant to this generator.

        Parameters
        ----------
        parser: argparse.ArgumentParser()
            The main parser to update.

        Returns
        -------
        parser: argparse.ArgumentParser()
            The updated parser.
        """
        parser.add_argument('--load_path',
                            help='load initialization for medium.')
        parser.add_argument('--nx',
                            default=20,
                            type=int,
                            help='(default value: %(default)s) Number of grid cell in x (North) direction')
        parser.add_argument('--ny',
                            default=20,
                            type=int,
                            help='(default value: %(default)s) Number of grid cell in y (East) direction')
        parser.add_argument('--nz',
                            default=20,
                            type=int,
                            help='(default value: %(default)s) Number of grid cell in z (Up) direction')
        parser.add_argument('--reff',
                            default=10.0,
                            type=np.float32,
                            help='(default value: %(default)s) Effective radius [micron]')
        parser.add_argument('--veff',
                            default=0.01,
                            type=np.float32,
                            help='(default value: %(default)s) Effective variance')

        return parser

    def get_grid(self):
        """
        Retrieve the scatterer grid.

        Returns
        -------
        grid: shdom.Grid
            A Grid object for this scatterer
        """

        # nx, ny, nz, nt = (self._data['estimated_extinction']).shape
        x = np.array(self._data['x']).ravel()
        y = np.array(self._data['y']).ravel()
        z = np.array(self._data['z']).ravel()
        # bb = shdom.BoundingBox(0, 0, 0, (nx-1) * dx, (ny-1) * dy, nz * dz)

        return shdom.Grid(x=x,y=y,z=z)

    def get_extinction(self, wavelength=None, grid_list=None):
        """
        Retrieve the optical extinction at a single wavelength on a grid.

        Parameters
        ----------
        wavelength: float
            Wavelength in microns. A Mie table at this wavelength should be added prior (see add_mie method).
        grid: shdom.Grid, optional
            A shdom.Grid object. If None is specified than a grid is created from Arguments given to the generator (get_grid method)

        Returns
        -------
        extinction: shdom.GridData
            A GridData object containing the optical extinction on a grid

        Notes
        -----
        If the LWC is specified then the extinction is derived using (lwc,reff,veff). If not the extinction needs to be directly specified.
        The input wavelength is rounded to three decimals.
        """
        if grid_list is None:
            NotImplemented()
            grid_list = self.get_grid()
        extinction =[]
        if True: # does not support microphysics
            for grid in grid_list:
                data = np.squeeze(self._data['estimated_extinction'])
                data[data<0.001]=0.001
                ext = shdom.GridData(self.get_grid(), np.squeeze(self._data['estimated_extinction']))
                extinction.append(ext.resample(grid))
        else:
            NotImplemented()
        return extinction

    def get_albedo(self, wavelength, grid_list=None):
        """
        Retrieve the single scattering albedo at a single wavelength on a grid.

        Parameters
        ----------
        wavelength: float
            Wavelength in microns. A Mie table at this wavelength should be added prior (see add_mie method).
        grid: shdom.Grid, optional
            A shdom.Grid object. If None is specified than a grid is created from Arguments given to the generator (get_grid method)

        Returns
        -------
        albedo: shdom.GridData
            A GridData object containing the single scattering albedo [0,1] on a grid

        Notes
        -----
        The input wavelength is rounded to three decimals.
        """
        if grid_list is None:
            NotImplemented()
            grid_list = self.get_grid()
        albedo =[]
        # if self.args.lwc is None:
        #     for grid in grid_list:
        #         if grid.type == 'Homogeneous':
        #             alb_data = self.args.albedo
        #         elif grid.type == '1D':
        #             alb_data = np.full(shape=(grid.nz), fill_value=self.args.albedo, dtype=np.float32)
        #         elif grid.type == '3D':
        #             alb_data = np.full(shape=(grid.nx, grid.ny, grid.nz), fill_value=self.args.albedo, dtype=np.float32)
        #         albedo.append(shdom.GridData(grid, alb_data))
        # else:
        assert wavelength is not None, 'No wavelength provided'
        reff_list = self.get_reff(grid_list)
        veff_list = self.get_veff(grid_list)
        for reff, veff in zip(reff_list,veff_list):
            albedo.append(self.mie[shdom.float_round(wavelength)].get_albedo(reff, veff))
        return albedo

    def get_phase(self, wavelength, grid_list=None):
        """
        Retrieve the single scattering albedo at a single wavelength on a grid.

        Parameters
        ----------
        wavelength: float
            Wavelength in microns. A Mie table at this wavelength should be added prior (see add_mie method).
        grid: shdom.Grid, optional
            A shdom.Grid object. If None is specified than a grid is created from Arguments given to the generator (get_grid method)

        Returns
        -------
        albedo: shdom.GridData
            A GridData object containing the single scattering albedo [0,1] on a grid

        Notes
        -----
        The input wavelength is rounded to three decimals.
        """
        if grid_list is None:
            NotImplemented()
            grid_list = self.get_grid()
        phase =[]
        # for grid in grid_list:
        #     if grid.type == 'Homogeneous':
        #         phase_data = self.args.phase
        #     elif grid.type == '1D':
        #         phase_data = np.full(shape=(grid.nz), fill_value=self.args.phase, dtype=np.float32)
        #     elif grid.type == '3D':
        #         phase_data = np.full(shape=(grid.nx, grid.ny, grid.nz), fill_value=self.args.phase, dtype=np.float32)
        #     phase.append(shdom.GridData(grid, phase_data))
        # else:
        assert wavelength is not None, 'No wavelength provided'
        reff_list = self.get_reff(grid_list)
        veff_list = self.get_veff(grid_list)
        for reff, veff in zip(reff_list,veff_list):
            phase.append(self.mie[shdom.float_round(wavelength)].get_phase(reff, veff))
        return phase

    def get_lwc(self, grid_list=None):
        """
        Retrieve the liquid water content.

        Parameters
        ----------
        grid: shdom.Grid, optional
            A shdom.Grid object. If None is specified than a grid is created from Arguments given to the generator (get_grid method)

        Returns
        -------
        lwc: shdom.GridData
            A GridData object containing liquid water content (g/m^3) on a 3D grid.
        """
        if grid_list is None:
            NotImplemented()
            grid_list = self.get_grid()

        lwc = self.args.lwc
        lwc_list =[]

        if lwc is not None:
            for grid in grid_list:
                if grid.type == 'Homogeneous':
                    lwc_data = self.args.lwc
                elif grid.type == '1D':
                    lwc_data = np.full(shape=(grid.nz), fill_value=self.args.lwc, dtype=np.float32)
                elif grid.type == '3D':
                    lwc_data = np.full(shape=(grid.nx, grid.ny, grid.nz), fill_value=self.args.lwc, dtype=np.float32)
                lwc_list.append(shdom.GridData(grid, lwc_data))
        return lwc_list

    def get_reff(self, grid_list=None):
        """
        Retrieve the effective radius on a grid.

        Parameters
        ----------
        grid: shdom.Grid, optional
            A shdom.Grid object. If None is specified than a grid is created from Arguments given to the generator (get_grid method)

        Returns
        -------
        reff: shdom.GridData
            A GridData object containing the effective radius [microns] on a grid
        """
        if grid_list is None:
            NotImplemented()
            grid_list = self.get_grid()

        reff = self.args.reff
        reff_list = []

        if reff is not None:
            for grid in grid_list:
                if grid.type == 'Homogeneous':
                    reff_data = self.args.reff
                elif grid.type == '1D':
                    reff_data = np.full(shape=(grid.nz), fill_value=self.args.reff, dtype=np.float32)
                elif grid.type == '3D':
                    reff_data = np.full(shape=(grid.nx, grid.ny, grid.nz), fill_value=self.args.reff, dtype=np.float32)
                reff_list.append(shdom.GridData(grid, reff_data))
        return reff_list

    def get_veff(self, grid_list=None):
        """
        Retrieve the effective radius on a grid.

        Parameters
        ----------
        grid: shdom.Grid, optional
            A shdom.Grid object. If None is specified than a grid is created from Arguments given to the generator (get_grid method)

        Returns
        -------
        reff: shdom.GridData
            A GridData object containing the effective radius [microns] on a grid
        """
        if grid_list is None:
            NotImplemented()
            grid_list = self.get_grid()

        veff = self.args.veff
        veff_list = []

        if veff is not None:
            for grid in grid_list:
                if grid.type == 'Homogeneous':
                    veff_data = self.args.veff
                elif grid.type == '1D':
                    veff_data = np.full(shape=(grid.nz), fill_value=self.args.veff, dtype=np.float32)
                elif grid.type == '3D':
                    veff_data = np.full(shape=(grid.nx, grid.ny, grid.nz), fill_value=self.args.veff, dtype=np.float32)
                veff_list.append(shdom.GridData(grid, veff_data))
        return veff_list


class DynamicGridDataEstimator(object):

    def __init__(self, dynamic_data, min_bound=None, max_bound=None, precondition_scale_factor=1.0):
        self._dynamic_data = []
        for data in dynamic_data:
            init_data = shdom.GridData(data.grid, data.data)
            self._dynamic_data.append(shdom.GridDataEstimator(init_data,min_bound, max_bound,precondition_scale_factor))

    def get_dynamic_data(self):
        return self._dynamic_data

    @property
    def dynamic_data(self):
        return self._dynamic_data


class TemporaryScattererEstimator(shdom.ScattererEstimator,TemporaryScatterer):

    def __init__(self, scatterer, time=0.0):
        TemporaryScatterer.__init__(self,scatterer,time)
        shdom.ScattererEstimator.__init__(self)


class DynamicScattererEstimator(object):
    def __init__(self, wavelength, time_list, **kwargs):
        self._num_scatterers = 0
        self._wavelength = wavelength
        self._time_list = []
        self._type = None
        dynamic_lwc = None
        dynamic_extinction = None
        for key, value in kwargs.items():
            if key == "extinction":
                assert isinstance(value,DynamicGridDataEstimator),\
                    'extinction type has to be DynamicGridDataEstimator'
                assert dynamic_lwc is None
                dynamic_extinction = value
                self._type = 'OpticalScattererEstimator'
            elif key == "albedo":
                dynamic_albedo = value
            elif key == "phase":
                dynamic_phase = value
            elif key == "lwc":
                assert dynamic_extinction is None
                dynamic_lwc = value
                self._type = 'MicrophysicalScatterer'
            elif key == "reff":
                dynamic_reff = value
            elif key == "veff":
                dynamic_veff = value

        if self._type == 'OpticalScattererEstimator':
            assert len(dynamic_extinction.dynamic_data)==len(dynamic_albedo)==len(dynamic_phase)==len(time_list),\
            'All attributes should have the same length'
            self._temporary_scatterer_estimator_list = []
            for extinction, albedo, phase, time in \
                    zip(dynamic_extinction.get_dynamic_data(), dynamic_albedo, dynamic_phase, time_list):
                scatterer_estimator = shdom.OpticalScattererEstimator(wavelength, extinction, albedo, phase)
                self._temporary_scatterer_estimator_list.append(TemporaryScattererEstimator(scatterer_estimator,time))
                self._time_list.append(time)
                self._num_scatterers += 1
        elif self._type == 'MicrophysicalScatterer':
            # Mie scattering for water droplets
            mie = shdom.MiePolydisperse()
            # mie_table_path = 'mie_tables/polydisperse/Water_{}nm.scat'.format(shdom.int_round(wavelength))
            mie_list = []
            if not isinstance(wavelength, list) and not isinstance(wavelength, np.ndarray):
                wavelength = [wavelength]
            for wl in wavelength:
                mie_table_path = 'mie_tables/polydisperse/Water_{}nm.scat'.format(shdom.int_round(wl))
                mie = shdom.MiePolydisperse()
                mie.read_table(file_path=mie_table_path)
                mie_list.append(mie)
            # assert len(dynamic_extinction.dynamic_grid_data) == len(dynamic_albedo) == len(dynamic_phase) == len(
            #     time_list), \
            #     'All attributes should have the same length'
            self._temporary_scatterer_estimator_list = []
            for lwc, reff, veff, time in \
                    zip(dynamic_lwc, dynamic_reff, dynamic_veff, time_list):
                scatterer_estimator = shdom.MicrophysicalScattererEstimator(mie_list, lwc, reff, veff)
                self._temporary_scatterer_estimator_list.append(
                    TemporaryScattererEstimator(scatterer_estimator, time))
                self._time_list.append(time)
                self._num_scatterers += 1
            else:
                assert 'Not supported'


    def get_velocity(self):
        assert self._temporary_scatterer_estimator_list is not None and self._num_scatterers > 1, \
            'Dynamic Scatterer should have more than 1 scatterer'
        scatterer_location = []
        for temporary_scatterer in self._temporary_scatterer_estimator_list:
            scatterer = temporary_scatterer.get_scatterer()
            scatterer_location.append([scatterer.grid.x[0], scatterer.grid.y[0], scatterer.grid.z[0]])
        scatterer_location = np.asarray(scatterer_location)
        time_list = np.asarray(self._time_list).reshape((-1, 1))
        scatterer_velocity_list = (scatterer_location[1:, :] - scatterer_location[:-1, :]) / (
                    time_list[1:] - time_list[:-1])
        return scatterer_velocity_list



    def set_mask(self, mask_list):
        for scatterer_estimator, mask in zip(self._temporary_scatterer_estimator_list, mask_list):
            scatterer_estimator.scatterer.set_mask(mask)

    def get_dynamic_optical_scatterer(self):
        return self._temporary_scatterer_estimator_list

    @property
    def temporary_scatterer_estimator_list(self):
        return self._temporary_scatterer_estimator_list

    @property
    def type(self):
        return self._type

    @property
    def num_scatterers(self):
        return self._num_scatterers

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def time_list(self):
        return self._time_list


class DynamicMediumEstimator(object):

    def __init__(self, dynamic_scatterer_estimator=None, air=None, scatterer_velocity=[0,0,0],
                 loss_type='l2', regularization_type='weighted_normal', exact_single_scatter=True, stokes_weights=None, images_weight=None, sigma=20,regularization_const=1):
        self._num_mediums = 0
        self._wavelength = []
        self._dynamic_medium = []
        self._time_list = []
        self._regularization_const = regularization_const
        self._medium_list = []
        self._images_mask = None
        self._images_weight = images_weight
        self._dynamic_scatterer_estimator = dynamic_scatterer_estimator
        self._scatterer_velocity = scatterer_velocity
        self._regularization_type = regularization_type
        self._sigma = sigma
        if dynamic_scatterer_estimator is not None and air is not None:
            self.set_dynamic_medium_estimator(dynamic_scatterer_estimator,air,loss_type, exact_single_scatter, stokes_weights)

    def set_dynamic_medium_estimator(self, dynamic_scatterer_estimator, air, loss_type='l2', exact_single_scatter=True, stokes_weights=None):
        assert isinstance(dynamic_scatterer_estimator,DynamicScattererEstimator) \
                    and (isinstance(air,shdom.Scatterer) or isinstance(air,shdom.MultispectralScatterer))
        self._num_mediums = 0
        self._medium_list = []
        self._time_list = []
        # temporary_scatterer_list = dynamic_scatterer_estimator.temporary_scatterer_list
        for temporary_scatterer, time in zip(dynamic_scatterer_estimator.temporary_scatterer_estimator_list, dynamic_scatterer_estimator.time_list):
            scatterer = temporary_scatterer.get_scatterer()
            first_scatterer = True if self._num_mediums == 0 else False
            if first_scatterer:
                self._wavelength = scatterer.wavelength
            else:
                assert np.allclose(self.wavelength,
                                   scatterer.wavelength), ' medium wavelength {} differs from dynamic_scatterers wavelength {}'.format(
                    self.wavelength, scatterer.wavelength)
            medium_grid = scatterer.grid + air.grid
            medium = shdom.MediumEstimator(grid=medium_grid, loss_type=loss_type, exact_single_scatter=exact_single_scatter, stokes_weights=stokes_weights)
            medium.add_scatterer(scatterer, name='cloud')
            medium.add_scatterer(air, name='air')
            self._medium_list.append(medium)
            self._num_mediums += 1
            self._time_list.append(time)

    def compute_gradient(self,dynamic_solver, measurements, n_jobs=1):
        print('computing gradient')
        regularization_type = self._regularization_type
        regularization_const = self._regularization_const
        sigma = self._sigma
        data_gradient = []
        data_loss = 0.0
        images = []
        loss =[]
        images_weight = self._images_weight
        resolutions = measurements.camera.dynamic_projection.resolution
        pix = np.array(measurements.camera.dynamic_projection.npix).ravel()
        total_pix = np.sum(pix)
        num_images = len(measurements.images)
        avg_npix = np.mean(resolutions)**2
        vmax = [image.max() * 1.25 for image in measurements.images]
        vmax = max(vmax)
        # split_indices = np.array(np.sum(measurements.camera.dynamic_projection.npix[:-1],axis=1)).ravel()
        split_indices = np.cumsum(pix[:])
        measurements, masks = measurements.split(split_indices)
        if images_weight is not None:
            assert np.isclose(np.sum(images_weight), num_images)
            images_weight = np.repeat(images_weight, pix)
            measurements_weight = np.array_split(images_weight, split_indices)
        else:
            measurements_weight = [None]*len(measurements)
        # if len(self.wavelength)>2:
        wavelength = self.wavelength
        if not isinstance(self.wavelength,list):
            wavelength = [self.wavelength]
        num_channels = len(wavelength)
        multichannel = num_channels > 1
        norm_const = total_pix * vmax
        # a= np.array(
        #     [332, 623, 1001, 1364, 1800, 2473, 3223, 4172, 5063, 5702, 6209, 5965, 5538, 5208, 4583, 4054, 3499, 3136,
        #      2782, 2376, 2063])
        # a= a/np.sum(a)
        # i=0
        grad = []
        for medium_estimator, rte_solver, measurement, measurement_weight, mask in zip(self.medium_list, dynamic_solver.solver_array_list, measurements, measurements_weight, masks):
            grad_output = medium_estimator.compute_gradient(shdom.RteSolverArray(rte_solver),measurement,n_jobs,measurement_weight, mask)
            # data_gradient.extend(grad_output[0] / measurement.images.size/ len(measurements)) #unit less grad
            # data_loss += (grad_output[1] / measurement.images.size) #unit less loss
            norm_uncertainties = 1
            image = grad_output[2]
            num_images = len(image)
            if measurement.uncertainties is not None:
                norm_uncertainties = np.mean(measurement.uncertainties)
            # a = np.mean(np.abs(grad_output[0]))

            data_gradient.extend(grad_output[0] / num_images / norm_uncertainties )
            grad.append(grad_output[0] / num_images / norm_uncertainties)
            data_loss += (grad_output[1] /num_images / norm_uncertainties)
            images += image

        # loss.append(data_loss ** 0.5 / norm_const) #unit less loss
        loss.append(data_loss)
        # spatial_regularization_loss, spatial_regularization_grad=self.compute_spatial_gradient_regularization()
        spatial_regularization_loss= 0
        spatial_regularization_grad=0
        if regularization_const != 0 and len(self.medium_list) > 1:
            regularization_loss, regularization_grad = self.compute_gradient_regularization(regularization_const, avg_npix, data_grad=grad, regularization_type=regularization_type, sigma=sigma)
            # regularization_loss, regularization_grad = self.compute_gradient_weighted(regularization_const, loss1, grad)
            if regularization_type =='weighted_normal':
                state_gradient = regularization_grad + spatial_regularization_grad
                # loss.append(0)
                loss.append(spatial_regularization_loss)
            else:
                loss.append(regularization_loss + spatial_regularization_loss)
                state_gradient = np.asarray(data_gradient) + regularization_grad + spatial_regularization_grad
        else:
            loss.append(0+spatial_regularization_loss)
            state_gradient = np.asarray(data_gradient) + spatial_regularization_grad

        return state_gradient, loss, images


    def compute_gradient_regularization(self,regularization_const, avg_npix, data_grad=None, regularization_type='weighted_normal', sigma = None):
        loss = 0
        ind_param = 0
        data_grad = np.asarray(data_grad)
        grad = np.zeros(data_grad.shape)
        for param_name, param in self.dynamic_scatterer_estimator.temporary_scatterer_estimator_list[0].scatterer.estimators.items():

            if regularization_type == 'l2':
                typical_grad = {
                    'extinction': 1 / 100,
                    'lwc': 0.01 / 100,
                    'reff': 10 / 100,
                    'veff': 0.01 / 100
                }
                # typical_avg = {
                #     'extinction': 1,
                #     'lwc':  1,
                #     'reff': 1,
                #     'veff': 1
                # }
                param_typical_avg = typical_grad[param_name]
                # param_typical_avg = param.precondition_scale_factor ** 0.5

                estimated_parameter_stack = []
                # grid_size = param.grid.nx * param.grid.ny * param.grid.nz
                # grid_size = 1

                for scatterer_estimator in self._medium_list:
                    estimated_parameter_stack.append(scatterer_estimator.get_state())

                grad = np.empty(shape=(0), dtype=np.float64)

                dynamic_estimated_parameter = np.stack(estimated_parameter_stack, axis=1)
                curr_grad = np.zeros_like(dynamic_estimated_parameter)
                num_voxels = dynamic_estimated_parameter.shape[0]
                M = dynamic_estimated_parameter.shape[1]
                I_typical = 0.1

                time = np.array(self.time_list)
                assert curr_grad.shape[1]>1,'cant calculate gradient for 1 image'
                norm_const = (1 / (M-1)) * (I_typical/param_typical_avg)**2 * (avg_npix/num_voxels)

                if 0: #small derevatives
                    curr_grad[:,:-1] += 2*(dynamic_estimated_parameter[:,:-1] - dynamic_estimated_parameter[:,1:]) / (time[:-1] - time[1:])
                    curr_grad[:, 1:] += 2*(dynamic_estimated_parameter[:,1:] - dynamic_estimated_parameter[:,:-1]) / (time[1:] - time[:-1])
                    curr_loss = np.sum(((dynamic_estimated_parameter[:,:-1] - dynamic_estimated_parameter[:,1:]) /
                                        (time[:-1] - time[1:]))**2) * norm_const
                # curr_grad[:,:-1] += 2*(dynamic_estimated_parameter[:,:-1] - dynamic_estimated_parameter[:,1:]) / (time[:-1] - time[1:]) /(dynamic_estimated_parameter[:,:-1] + dynamic_estimated_parameter[:,1:])
                # curr_grad[:, 1:] += 2*(dynamic_estimated_parameter[:,1:] - dynamic_estimated_parameter[:,:-1]) / (time[1:] - time[:-1]) /(dynamic_estimated_parameter[:,1:] + dynamic_estimated_parameter[:,:-1])

                if 1:#small relative change l2
                    t0 = (dynamic_estimated_parameter[:,:-1] + dynamic_estimated_parameter[:,1:])
                    # t1 = np.sum(t0**2,axis=0)
                    t2 = (dynamic_estimated_parameter[:,:-1] - dynamic_estimated_parameter[:,1:])
                    # curr_grad[:,:-1] += 2/t1[np.newaxis,:]*t2-2/t1**2*(np.sum(t2**2,axis=0))[np.newaxis,:]*t0
                    curr_grad[:, :-1] += 2 * t2 / t0 / t0 - 2*t2 * t2 / t0 / (t0 * t0)
                    t0 = (dynamic_estimated_parameter[:, 1:] + dynamic_estimated_parameter[:,:-1])
                    # t1 = np.sum(t0 ** 2, axis=0)
                    t2 = (dynamic_estimated_parameter[:, 1:] - dynamic_estimated_parameter[:,:-1])
                    #
                    # curr_grad[:, 1:] += 2/t1[np.newaxis,:]*t2-2/t1**2*(np.sum(t2**2,axis=0))[np.newaxis,:]*t0
                    curr_grad[:, 1:] += 2 * t2 / t0 / t0 - 2*t2 * t2 / t0 / (t0 * t0)
                    curr_loss = np.sum(((dynamic_estimated_parameter[:, :-1] - dynamic_estimated_parameter[:, 1:]) /
                                        (dynamic_estimated_parameter[:, :-1] + dynamic_estimated_parameter[:,
                                                                               1:])) ** 2) * norm_const
                if 0:#small relative change l1
                    t0 = (dynamic_estimated_parameter[:,:-1] + dynamic_estimated_parameter[:,1:])
                    t0[t0==0] = 1e-15
                    t1 = (dynamic_estimated_parameter[:,:-1] - dynamic_estimated_parameter[:,1:])
                    t2 = np.sign(t1/t0)
                    curr_grad[:, :-1] += t2/t0 - t1*t2/(t0**2)
                    t0 = (dynamic_estimated_parameter[:, 1:] + dynamic_estimated_parameter[:,:-1])
                    t0[t0 == 0] = 1e-15
                    t1 = (dynamic_estimated_parameter[:, 1:] - dynamic_estimated_parameter[:,:-1])
                    t2 = np.sign(t1/t0)
                    #
                    # curr_grad[:, 1:] += 2/t1[np.newaxis,:]*t2-2/t1**2*(np.sum(t2**2,axis=0))[np.newaxis,:]*t0
                    curr_grad[:, 1:] += t2/t0 - t1*t2/(t0**2)
                    curr_loss = np.linalg.norm(((dynamic_estimated_parameter[:, :-1] - dynamic_estimated_parameter[:, 1:]) /
                                        (dynamic_estimated_parameter[:, :-1] + dynamic_estimated_parameter[:,
                                                                               1:])),ord=1) * norm_const

                curr_grad = np.reshape(curr_grad,(-1,), order='F') * norm_const
                grad = np.concatenate((grad,curr_grad))


                # curr_loss = np.sum((dynamic_estimated_parameter[:,:-1] - dynamic_estimated_parameter[:,1:])**2 /
                #                     (dynamic_estimated_parameter[:,:-1] + dynamic_estimated_parameter[:,1:])**2 )* norm_const

                loss += curr_loss

            if regularization_type == 'weighted_normal':
                time = np.array(self.time_list)
                curr_grad = data_grad[:,ind_param:ind_param+param.num_parameters]
                for i, curr_time in enumerate(time):
                    dt = time - curr_time
                    w = norm.pdf(dt / sigma)#sigma[param_name]
                    w = w / np.sum(w) #new!!!
                    grad[i,ind_param:ind_param+param.num_parameters] = np.sum(w[:, np.newaxis] * curr_grad, axis=0).ravel(order='F')
                    # reg_grad.append(curr_grad)
                ind_param += param.num_parameters
                # grad = np.asarray(reg_grad)
            else:
                NotImplemented()
        grad = grad.ravel()
        return regularization_const * loss, regularization_const * grad

    def compute_spatial_gradient_regularization(self,regularization_const=1,mu1=1,mu2=1,c1=1,c2=1):
        loss = 0
        from scipy import ndimage
        grad = np.empty(shape=(0), dtype=np.float64)


        for scattarer in self.dynamic_scatterer_estimator.temporary_scatterer_estimator_list:
            for param_name, param in scattarer.scatterer.estimators.items():
                if param_name=='lwc':
                    kz = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                   [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                                   [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

                    kz = kz / np.linalg.norm(kz.flatten(), ord=1)
                    const = 0
                    data = param.data
                    laplacian3d = kz
                    kz_data = ndimage.convolve(data, laplacian3d, mode='nearest')[param.mask.data]
                    loss += const * np.linalg.norm(kz_data,ord=2) / kz_data.size
                    curr_grad = 2 * const * kz_data / kz_data.size
                    grad = np.concatenate((grad, curr_grad))
                if param_name=='reff':
                    # kz = np.array([1,-2,1])
                    from scipy import ndimage
                    ## smooth prior
                    const = 0.1
                    data = param.data
                    # laplace_data = ndimage.laplace(data)
                    l_data = data[param.mask.data]
                    laplace_data = l_data[1:] - l_data[:-1]
                    laplace_data[0] = laplace_data[1]
                    laplace_data[-1] = laplace_data[-2]
                    loss += const * np.linalg.norm(laplace_data,ord=2) / laplace_data.size
                    Dzbf = np.zeros(l_data.shape)
                    Dzbf[:-1] = -2*laplace_data
                    Dzbf[1:] += 2*laplace_data
                    curr_grad = const *Dzbf / laplace_data.size
                    # curr_grad = 2 * const * ndimage.laplace(laplace_data)[param.mask.data] / laplace_data.size
                    ## monotonically increasing in z prior
                    const = 0
                    c1=1
                    c2=1
                    # [param.mask.data.ravel('F')]
                    flat_data = data.ravel('F')
                    indic_b = np.tanh(flat_data / c1)
                    indic_b[param.mask.data.ravel('F')==False] = 0
                    if len(data.shape) == 3:
                        _, __, Dzb = np.gradient(data)
                    else:
                        # Dzb = np.gradient(data)
                        data=data[param.mask.data]
                        Dz1= data[1:] - data[:-1]
                    Dzbf = np.zeros(data.shape)
                    Dzbf[:-1] = (np.cosh(Dz1 / c2) ** (-2))/ c2
                    Dzbf[1:] += -(np.cosh(Dz1 / c2) ** (-2)) / c2
                    Dzbf[0] = Dzbf[1]
                    Dzbf[-1] = Dzbf[-2]
                    # Dzbf[0] = np.cosh((data[1] - data[0]) / c2) ** (-2) / c2
                    mz_grad = 2 * const *Dzbf
                    monoZ_term = -np.sum(np.tanh(Dz1/ c2))
                        # Dzb[3]=0
                    # Dzbf = Dzb.ravel('F')
                    # monoZ_term = -np.dot(np.transpose(indic_b), np.tanh(Dzbf / c2))
                    # dindic_b = np.cosh(flat_data / c1) ** (-2) / c1
                    # dindic_b[param.mask.data.ravel('F')==False] = 0
                    # mg1f = np.dot(np.diag(np.cosh(Dzbf / c2) ** (-2)), indic_b) / c2
                    # mg1f = np.reshape(mg1f, data.shape, 'F')
                    # if len(mg1f.shape) == 3:
                    #     _, __, mg1 = np.gradient(mg1f)
                    # else:
                    #     mg1 = np.gradient(mg1f)
                    #
                    #     mg1[1:] = mg1f[1:] - mg1f[:-1]
                    #     mg1[3] = 0
                    # # _, __, mg1 = np.gradient()
                    # mgf = -2 * const * (mg1.ravel('F') + np.dot(np.diag(dindic_b), np.tanh(Dzbf / c2)))
                    # mz_grad = np.reshape(mgf, data.shape, 'F')
                    curr_grad += mz_grad#[param.mask.data]
                    loss += const * monoZ_term
                    grad = np.concatenate((grad, curr_grad))

        return regularization_const * loss, regularization_const * grad

    def compute_direct_derivative(self, dynamic_solver):
        for solver, medium_estimator in zip(dynamic_solver,self._medium_list):
            medium_estimator.compute_direct_derivative(solver)

    def get_bounds(self):
        bounds = []
        for scatterer_estimator in self._medium_list:
            bounds.extend(scatterer_estimator.get_bounds())
        return bounds

    def get_state(self):
        state = []
        for scatterer_estimator in self._medium_list:
            state.extend(scatterer_estimator.get_state())
        return state

    def set_state(self, state):
        """
        Set the estimator state by setting all the internal estimators states.

        Parameters
        ----------
        state: np.array(dtype=np.float64)
            The combined state of all the internal estimators
        """
        num_parameters =[]
        for medium_estimator in self.medium_list:
            num_parameters.extend(medium_estimator.num_parameters)
        states = np.split(state, np.cumsum(num_parameters[:-1]))
        # if self._regularization_const>0:
        #     time = np.array(self._time_list)
        #     from scipy.stats import norm
        #     for ind, (curr_time, state) in enumerate(zip(time, states)):
        #         dt = time - curr_time
        #         w = norm.pdf(dt / self._sigma)
        #         w= w/np.sum(w)
        #         # w = np.exp(-0.5*(dt / sigma)^2)
        #         states[ind] = np.sum(w[:, np.newaxis] * np.array(states), axis=0).ravel(order='F')
        for medium_estimator, state in zip(self.medium_list, states):
            for (name, estimator) in medium_estimator.estimators.items():
                estimator.set_state(state)
                medium_estimator.scatterers[name] = estimator

    def get_num_parameters(self):
        num_parameters = []
        for scatterer_estimator in self._medium_list:
            num_parameters.append(scatterer_estimator.num_parameters)
        return num_parameters

    def get_scatterer(self, scatterer_name=None):
        return self._dynamic_scatterer_estimator

    @property
    def scatterer_velocity (self):
        return self._scatterer_velocity

    @property
    def medium_list(self):
        return self._medium_list

    @property
    def num_mediums(self):
        return self._num_mediums

    @property
    def time_list(self):
        return self._time_list

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def dynamic_scatterer_estimator(self):
        return self._dynamic_scatterer_estimator


class DynamicLocalOptimizer(object):
    """
   #TODO
    """

    def __init__(self, method, options={}, n_jobs=1, init_solution=True, regularization_const=0, pixles_mask=None):
        self._medium = None
        self._rte_solver = None
        self._measurements = None
        self._writer = None
        self._images = None
        self._iteration = 0
        self._loss = None
        self._n_jobs = n_jobs
        self._init_solution = init_solution
        self._num_parameters = []
        self._regularization_const = regularization_const
        self._cv_rte_solver = None
        self._cv_measurement = None
        self._cv_loss = None
        self.init = True
        self._pixles_mask=pixles_mask
        if method not in ['L-BFGS-B', 'TNC']:
            raise NotImplementedError('Optimization method [{}] not implemented'.format(method))
        self._method = method
        self._options = options

    def set_measurements(self, measurements):
        """
        Set the measurements (data-fit constraints)

        Parameters
        ----------
        measurements: shdom.Measurements
            A measurements object storing the acquired images and sensor geometry
        """
        self._measurements = measurements

    def set_medium_estimator(self, medium_estimator):
        """
        Set the DynamicMediumEstimator for the optimizer.

        Parameters
        ----------
        medium_estimator: shdom.DynamicMediumEstimator
            The DynamicMediumEstimator
        """
        self._medium = medium_estimator

    def set_dynamic_solver(self, dynamic_solver):
        """
        Set the DynamicRteSolver for the SHDOM iterations.

        Parameters
        ----------
        dynamic_solver: shdom.DynamicRteSolver
            The RteSolver
        """
        assert isinstance(dynamic_solver, shdom.DynamicRteSolver), 'dynamic_solver is not DynamicRteSolver'
        self._rte_solver = dynamic_solver

    def set_writer(self, writer):
        """
        Set a log writer to upload summaries into tensorboard.

        Parameters
        ----------
        writer: shdom.SummaryWriter
            Wrapper for the tensorboardX summary writer.
        """
        self._writer = writer
        if writer is not None:
            self._writer.attach_optimizer(self)

    def objective_fun(self, state):
        """
        The objective function (cost) and gradient at the current state.

        Parameters
        ----------
        state: np.array(shape=(self.num_parameters, dtype=np.float64)
            The current state vector

        Returns
        -------
        loss: np.float64
            The total loss accumulated over all pixels
        gradient: np.array(shape=(self.num_parameters), dtype=np.float64)
            The gradient of the objective function with respect to the state parameters

        Notes
        -----
        This function also saves the current synthetic images for visualization purpose
        """
        if 0:
            param_optimaizer = DynamicParametersOptimizer(self.method,n_jobs=self.n_jobs)
            param_optimaizer.set_measurements(self.measurements)
            param_optimaizer.set_dynamic_solver(self.rte_solver)
            param_optimaizer.set_medium_estimator(self.medium)
            param_optimaizer.set_state(state)
            param = param_optimaizer.minimize()
            self.set_state(param * state)
        elif 0 and self.init:
            self.init =False
            param_optimaizer = DynamicParametersOptimizer(self.method,n_jobs=self.n_jobs)
            param_optimaizer.set_measurements(self.measurements)
            param_optimaizer.set_dynamic_solver(self.rte_solver)
            param_optimaizer.set_medium_estimator(self.medium)
            param_optimaizer.set_state(state)
            reff, lwc = param_optimaizer.minimize()
            for medium in self.medium.medium_list:
                for name, estimator in medium.estimators.items():
                    estimator._reff = reff
                    # estimator._lwc = lwc
                    for namee, e in estimator.estimators.items():
                        # if namee =='lwc':
                        #     e._data = lwc.data
                        if namee =='reff':
                            e._data = reff.data
            original_state = np.array(self.get_state())
            self.set_state(original_state)
        elif 0 and  self.iteration>1 and  self.iteration%10>0 :
            param_optimaizer = DynamicParametersOptimizer1(self.method, n_jobs=self.n_jobs)
            param_optimaizer.set_measurements(self.measurements)
            param_optimaizer.set_dynamic_solver(self.rte_solver)
            param_optimaizer.set_medium_estimator(self.medium)
            param_optimaizer.set_state(state)
            reff = param_optimaizer.minimize()
            for medium in self.medium.medium_list:
                for name, estimator in medium.estimators.items():
                    estimator._reff = reff
            original_state = np.array(self.get_state())
            self.set_state(original_state)
            # plt.plot(reff.data)
            # plt.show()
            # self.set_state(param * state)
        else:
            self.set_state(state)
        gradient, loss, images = self._medium.compute_gradient(
            dynamic_solver=self.rte_solver,
            measurements=self.measurements,
            n_jobs=self.n_jobs
            # regularization_const=self._regularization_const
        )
        self._loss = loss
        self._images = images
        return sum(loss), gradient

    def callback(self, state):
        """
        The callback function invokes the callbacks defined by the writer (if any).
        Additionally it keeps track of the iteration number.

        Parameters
        ----------
        state: np.array(shape=(self.num_parameters, dtype=np.float64)
            The current state vector
        """
        self._iteration += 1

        # Writer callback functions
        if self.writer is not None:
            for callbackfn, kwargs in zip(self.writer.callback_fns, self.writer.kwargs):
                time_passed = time.time() - kwargs['ckpt_time']
                if time_passed > kwargs['ckpt_period']:
                    kwargs['ckpt_time'] = time.time()
                    callbackfn(kwargs)

    def minimize(self):
        """
        Local minimization with respect to the parameters defined.
        """
        if self.iteration == 0:
            self.init_optimizer()

        result = minimize(fun=self.objective_fun,
                          x0=self.get_state(),
                          method=self.method,
                          jac=True,
                          bounds=self.get_bounds(),
                          options=self.options,
                          callback=self.callback)
        return result

    def init_optimizer(self):
        """
        Initialize the optimizer.
        This means:
          1. Setting the RteSolver medium
          2. Initializing a solution
          3. Computing the direct solar flux derivatives
          4. Counting the number of unknown parameters
        """
        # TODO replace assert
        # assert self.rte_solver.num_solvers == self.measurements.num_channels == self.medium.num_wavelengths, \
        #     'RteSolver has {} solvers, Measurements have {} channels and Medium has {} wavelengths'.format(
        #         self.rte_solver.num_solvers, self.measurements.num_channels, self.medium.num_wavelengths)

        self.rte_solver.set_dynamic_medium(self.medium)
        self.rte_solver.init_solution()
        self.medium.compute_direct_derivative(self.rte_solver)
        self._num_parameters = []
        for medium in self.medium.medium_list:
            self._num_parameters.append(medium.num_parameters)

    def get_bounds(self):
        """
        Retrieve the bounds for every parameter from the MediumEstimator (used by scipy.minimize)

        Returns
        -------
        bounds: list of tuples
            The lower and upper bound of each parameter
        """
        return self.medium.get_bounds()

    def get_state(self):
        """
        Retrieve MediumEstimator state

        Returns
        -------
        state: np.array(dtype=np.float64)
            The state of the medium estimator
        """
        return self.medium.get_state()

    def set_state(self, state):
        """
        Set the state of the optimization. This means:
          1. Setting the MediumEstimator state
          2. Updating the RteSolver medium
          3. Computing the direct solar flux
          4. Computing the current RTE solution with the previous solution as an initialization

        Returns
        -------
        state: np.array(dtype=np.float64)
            The state of the medium estimator
        """
        self.medium.set_state(state)
        self.rte_solver.replace_dynamic_medium(self.medium)
        if self._init_solution is False:
            self.rte_solver.make_direct()
        self.rte_solver.solve(maxiter=100, init_solution=self._init_solution, verbose=False)

    def save_state(self, path):
        """
        Save Optimizer state to file.

        Parameters
        ----------
        path: str,
            Full path to file.
        """
        file = open(path, 'wb')
        file.write(pickle.dumps(self.get_state(), -1))
        file.close()

    def load_state(self, path):
        """
        Load Optimizer from file.

        Parameters
        ----------
        path: str,
            Full path to file.
        """
        file = open(path, 'rb')
        data = file.read()
        file.close()
        state = pickle.loads(data)
        self.set_state(state)

    def load_results(self, path):

        file = open(path, 'rb')
        data = file.read()
        file.close()
        state = pickle.loads(data)
        self.medium.set_state(state)

    def set_cross_validation_param(self, cv_rte_solver, cv_measurement, cv_index):
        self._cv_rte_solver = cv_rte_solver
        self._cv_measurement = cv_measurement
        assert np.any(np.diff(np.array(self.measurements.time_list)))>=0,'Time list should be sorted'
        self._cv_index = cv_index
        if not isinstance(cv_measurement.time_list, list):
            self._cv_time = [cv_measurement.time_list]
        else:
            self._cv_time = cv_measurement.time_list
        if not isinstance(self.measurements.time_list, list):
            time_list = self.measurements.time_list.tolist()
        else:
            time_list = self.measurements.time_list
        times = np.array(time_list+self._cv_time)
        self._cv_indices = np.argsort(times)

    def get_cross_validation_medium(self, scatterer_name):
        index = np.where(self._cv_indices == self._cv_indices.max())[0]
        if index == 0:
            medium = self._medium.medium_list[np.where(1 == self._cv_indices)[0].item()]
        elif index == len(self._cv_indices) - 1:
            medium = self._medium.medium_list[np.where((index - 1) == self._cv_indices)[0].item()]
        else:
            previous_medium = self._medium.medium_list[np.where((index - 1) == self._cv_indices)[0].item()]
            forward_medium = self._medium.medium_list[np.where((index + 1) == self._cv_indices)[0].item()]
            previous_time = np.array(self.measurements.time_list[np.where((index - 1) == self._cv_indices)[0].item()])
            forward_time = np.array(self.measurements.time_list[np.where((index + 1) == self._cv_indices)[0].item()])
            weighted_average_extinction_data = previous_medium.get_scatterer(scatterer_name).extinction.data + \
                                (np.array(self._cv_time) - previous_time) * (forward_medium.get_scatterer(scatterer_name).extinction.data
                                - previous_medium.get_scatterer(scatterer_name).extinction.data) / (forward_time - previous_time)
            medium = shdom.Medium(previous_medium.grid + forward_medium.grid)
            scatterer = previous_medium.get_scatterer(scatterer_name)
            weighted_average_extinction = shdom.GridData(scatterer.extinction.grid, weighted_average_extinction_data)
            cloud = shdom.OpticalScatterer(scatterer.wavelength,weighted_average_extinction,scatterer.albedo, scatterer.phase)
            medium.add_scatterer(cloud, name=scatterer_name)
            medium.add_scatterer(previous_medium.get_scatterer('air'), name='air')
        dynamic_medium = DynamicMedium()
        dynamic_medium.add_medium(medium)
        return dynamic_medium

    def get_cross_validation_images(self):
        # index = np.argwhere(self._cv_indices == len(self._cv_indices)-1)[0]
        dynamic_medium = self.get_cross_validation_medium('cloud')
        self._cv_rte_solver.set_dynamic_medium(dynamic_medium)
        self._cv_rte_solver.solve(maxiter=100)
        cv_image = self._cv_measurement.camera.render(self._cv_rte_solver, n_jobs=self.n_jobs)
        self._cv_loss = np.sum((np.array(cv_image) - np.array(self._cv_measurement.images)).ravel() ** 2)
        return cv_image


    @property
    def regularization_const(self):
        return self._regularization_const

    @property
    def method(self):
        return self._method

    @property
    def options(self):
        return self._options

    @property
    def rte_solver(self):
        return self._rte_solver

    @property
    def medium(self):
        return self._medium

    @property
    def measurements(self):
        return self._measurements

    @property
    def num_parameters(self):
        return self._num_parameters

    @property
    def writer(self):
        return self._writer

    @property
    def iteration(self):
        return self._iteration

    @property
    def n_jobs(self):
        return self._n_jobs

    @property
    def loss(self):
        return self._loss

    @property
    def images(self):
        return self._images

    @property
    def cv_loss(self):
        return self._cv_loss

class DynamicParametersOptimizer(DynamicLocalOptimizer):
    """
   #TODO
    """

    def __init__(self, method, options={}, n_jobs=1, init_solution=True, regularization_const=0, pixles_mask=None):
        super().__init__(method, options, n_jobs, init_solution)
        self._medium = None
        self._rte_solver = None
        self._measurements = None
        self._writer = None
        self._images = None
        self._iteration = 0
        self._loss = None
        self._n_jobs = n_jobs
        self._init_solution = init_solution
        self._num_parameters = []
        self._regularization_const = regularization_const
        self._cv_rte_solver = None
        self._cv_measurement = None
        self._cv_loss = None
        self._pixles_mask=pixles_mask
        if method not in ['L-BFGS-B', 'TNC']:
            raise NotImplementedError('Optimization method [{}] not implemented'.format(method))
        self._method = method
        self._options = options


    def objective_fun(self, scalar):
        """
        The objective function (cost) and gradient at the current state.

        Parameters
        ----------
        state: np.array(shape=(self.num_parameters, dtype=np.float64)
            The current state vector

        Returns
        -------
        loss: np.float64
            The total loss accumulated over all pixels
        gradient: np.array(shape=(self.num_parameters), dtype=np.float64)
            The gradient of the objective function with respect to the state parameters

        Notes
        -----
        This function also saves the current synthetic images for visualization purpose
        """
        for medium in self.medium.medium_list:
            for name, estimator in medium.estimators.items():
                old_reff = estimator.reff
                old_lwc = estimator.lwc

                reff = self.get_monotonous_reff(old_reff, slope=scalar)
                lwc = self.get_monotonous_lwc(old_lwc, slope=self.lwc_slope)
                estimator._lwc = lwc
                estimator._reff = reff

        self.original_state = np.array(self.get_state())
        self.set_state(self.original_state)
        images = self.measurements.camera.render(self.rte_solver,n_jobs=self._n_jobs)
        loss = np.sum((np.array(images)-np.array(self.measurements.images))**2)
        return loss

    def get_monotonous_reff(self, old_reff, slope, z0=None, reff0=None):
        mask = old_reff.data>0
        grid = old_reff.grid
        if z0 is None:
            z0=0
        z0 = grid.z[mask][z0]
        if reff0 is None:
            reff0 = self.reff0
        Z = grid.z - z0
        Z[Z < 0] = 0
        reff_data = (slope * Z ** (1. / 3.)) + reff0

        # reff_data[Z == 0] = 0
        reff_data[mask==0] = 0
        return shdom.GridData(grid, reff_data)

    def get_monotonous_lwc(self, old_lwc, slope, z0=None):
        mask = old_lwc.data > 0
        mask_z = np.sum(mask,(0,1))>0
        grid = old_lwc.grid
        if z0 is None:
            z0 = grid.z[mask_z][0]

        Z = grid.z - z0
        Z[Z < 0] = 0
        lwc_profile = (slope * Z ) + 0.01
        lwc_data = np.tile(lwc_profile[np.newaxis, np.newaxis, :], (grid.nx, grid.ny, 1))

        lwc_data[mask==0] = 0
        return shdom.GridData(grid, lwc_data)

    def minimize(self):
        """
        Local minimization with respect to the parameters defined.
        """
        # if self.iteration == 0:
            # self.init_optimizer()


        from scipy.optimize import minimize_scalar
        self.original_state = np.array(self.get_state())

        # # res = minimize_scalar(self.objective_fun, method='bounded', options={'maxiter':8}, bracket=(0.5,2), bounds=(0.5, 2))
        loss = np.inf
        for z in range(1, 4):
            self.z0=z
            for l_slope in range(3,6):
                l_slope /= 10
                for r0 in range(1,5):
                    self.reff0=r0
                    self.lwc_slope=l_slope
                    print('check {} {} {} {}'.format(loss, r0, l_slope, z))
                    res = minimize_scalar(self.objective_fun, method='bounded', options={'maxiter':2}, bracket=(6,8), bounds=(6, 8))
                    if res.fun < loss:
                        loss = res.fun
                        reff_slope = res.x
                        reff0 = r0
                        z0 = z
                        lwc_slope = l_slope
                        print('{} {} {} {} {}'.format(loss, reff_slope, reff0, lwc_slope, z))
        # reff_slope = 6.23
        # reff0=5
        # lwc_slope = 0.5
        # reff_slope = 6.67
        # reff0=3
        # lwc_slope = 0.3
        # res = self.grid_search(bounds_slope =(1, 8), bounds_reff0=(1,5))
        for medium in self.medium.medium_list:
            for name, estimator in medium.estimators.items():
                old_reff = estimator.reff
                old_lwc = estimator.lwc
                reff = self.get_monotonous_reff(old_reff, slope=reff_slope, reff0=reff0, z0=z0)
                lwc = self.get_monotonous_lwc(old_lwc, slope=lwc_slope)
                # lwc._data = lwc.data / 10
        print('parameters are {} {} {}'.format(reff_slope, reff0, lwc_slope))
        # shdom.cloud_plot(lwc.data)
        # plt.plot(reff.data)
        # plt.show()
        return reff, lwc

    # def grid_search(self, bounds_slope, bounds_reff0):
    #     for

class DynamicParametersOptimizer1(DynamicLocalOptimizer):
    """
   #TODO
    """

    def __init__(self, method, options={}, n_jobs=1, init_solution=True, regularization_const=0, pixles_mask=None):
        super().__init__(method, options, n_jobs, init_solution)
        self._medium = None
        self._rte_solver = None
        self._measurements = None
        self._writer = None
        self._images = None
        self._iteration = 0
        self._loss = None
        self._n_jobs = n_jobs
        self._init_solution = init_solution
        self._num_parameters = []
        self._regularization_const = regularization_const
        self._cv_rte_solver = None
        self._cv_measurement = None
        self._cv_loss = None
        self._pixles_mask=pixles_mask
        if method not in ['L-BFGS-B', 'TNC']:
            raise NotImplementedError('Optimization method [{}] not implemented'.format(method))
        self._method = method
        self._options = options


    def objective_fun(self, scalar):
        """
        The objective function (cost) and gradient at the current state.

        Parameters
        ----------
        state: np.array(shape=(self.num_parameters, dtype=np.float64)
            The current state vector

        Returns
        -------
        loss: np.float64
            The total loss accumulated over all pixels
        gradient: np.array(shape=(self.num_parameters), dtype=np.float64)
            The gradient of the objective function with respect to the state parameters

        Notes
        -----
        This function also saves the current synthetic images for visualization purpose
        """
        for medium in self.medium.medium_list:
            for name, estimator in medium.estimators.items():
                old_reff = estimator.reff
                # old_lwc = estimator.lwc

                reff = self.get_monotonous_reff(old_reff, slope=scalar)
                # lwc = self.get_monotonous_lwc(old_lwc, slope=self.lwc_slope)
                # estimator._lwc = lwc
                estimator._reff = reff

        self.original_state = np.array(self.get_state())
        self.set_state(self.original_state)
        images = self.measurements.camera.render(self.rte_solver,n_jobs=self._n_jobs)
        loss = np.sum((np.array(images)-np.array(self.measurements.images))**2)
        return loss

    def get_monotonous_reff(self, old_reff, slope, z0=None, reff0=3):
        mask = old_reff.data>0
        grid = old_reff.grid
        if z0 is None:
            z0 = grid.z[mask][0]
        if reff0 is None:
            reff0 = self.reff0
        Z = grid.z - z0
        Z[Z < 0] = 0
        reff_data = (slope * Z ** (1. / 3.)) + reff0

        # reff_data[Z == 0] = 0
        reff_data[mask==0] = 0
        return shdom.GridData(grid, reff_data)

    def get_monotonous_lwc(self, old_lwc, slope, z0=None):
        mask = old_lwc.data > 0
        mask_z = np.sum(mask,(0,1))>0
        grid = old_lwc.grid
        if z0 is None:
            z0 = grid.z[mask_z][0]

        Z = grid.z - z0
        Z[Z < 0] = 0
        lwc_profile = (slope * Z ) + 0.01
        lwc_data = np.tile(lwc_profile[np.newaxis, np.newaxis, :], (grid.nx, grid.ny, 1))

        lwc_data[mask==0] = 0
        return shdom.GridData(grid, lwc_data)

    def minimize(self):
        """
        Local minimization with respect to the parameters defined.
        """
        # if self.iteration == 0:
            # self.init_optimizer()


        from scipy.optimize import minimize_scalar
        self.original_state = np.array(self.get_state())

        # # res = minimize_scalar(self.objective_fun, method='bounded', options={'maxiter':8}, bracket=(0.5,2), bounds=(0.5, 2))


        res = minimize_scalar(self.objective_fun, method='bounded', options={'maxiter':6}, bracket=(3,8), bounds=(3, 8))

        # slope = 6.23
        # reff0=5
        # res = self.grid_search(bounds_slope =(1, 8), bounds_reff0=(1,5))
        for medium in self.medium.medium_list:
            for name, estimator in medium.estimators.items():
                old_reff = estimator.reff
                reff = self.get_monotonous_reff(old_reff, slope=res.x, reff0=3)
        print('parameters are {}'.format(res.x))
        return reff

    # def grid_search(self, bounds_slope, bounds_reff0):
    #     for

class ParametersOptimizer(DynamicLocalOptimizer):
    """
   #TODO
    """

    def __init__(self, method, options={}, n_jobs=1, init_solution=True, regularization_const=0, pixles_mask=None):
        super().__init__(method, options, n_jobs, init_solution)
        self._medium = None
        self._rte_solver = None
        self._measurements = None
        self._writer = None
        self._images = None
        self._iteration = 0
        self._loss = None
        self._n_jobs = n_jobs
        self._init_solution = init_solution
        self._num_parameters = []
        self._regularization_const = regularization_const
        self._cv_rte_solver = None
        self._cv_measurement = None
        self._cv_loss = None
        self._pixles_mask=pixles_mask
        if method not in ['L-BFGS-B', 'TNC']:
            raise NotImplementedError('Optimization method [{}] not implemented'.format(method))
        self._method = method
        self._options = options


    def objective_fun(self, scalar):
        """
        The objective function (cost) and gradient at the current state.

        Parameters
        ----------
        state: np.array(shape=(self.num_parameters, dtype=np.float64)
            The current state vector

        Returns
        -------
        loss: np.float64
            The total loss accumulated over all pixels
        gradient: np.array(shape=(self.num_parameters), dtype=np.float64)
            The gradient of the objective function with respect to the state parameters

        Notes
        -----
        This function also saves the current synthetic images for visualization purpose
        """
        z0_list = []
        print('     check alpha_r {}'.format(scalar))

        for medium in self.medium.medium_list:
            for name, estimator in medium.estimators.items():
                old_reff = estimator.reff
                mask = estimator.lwc.data > 0
                mask_z = np.sum(mask, (0, 1)) > 0
                z0 =  estimator.lwc.grid.z[mask_z][0]
                z0_list.append(z0)
                reff = self.get_monotonous_reff(old_reff, slope=scalar, z0=z0)
                estimator._reff = reff

        self.original_state = np.array(self.get_state())
        self.set_state(self.original_state)
        loss = np.inf
        camera = shdom.Camera(sensor = self.measurements.camera.sensor,
                              projection=self.measurements.camera._dynamic_projection.multiview_projection_list[0])
        for i, solver in enumerate(self.rte_solver):
            images = camera.render(solver,n_jobs=self._n_jobs)
            cur_loss = np.sum((np.array(images)-np.array(self.measurements.images))**2)
            print('          check z0 {}'.format(z0_list[i]))
            if cur_loss<loss:
                loss = cur_loss
                self.z0 = z0_list[i]


        return loss

    def get_monotonous_reff(self, old_reff, slope, z0=None, reff0=None):
        mask = old_reff.data>0
        grid = old_reff.grid
        if z0 is None:
            z0=0
            z0 = grid.z[mask][z0]
        if reff0 is None:
            reff0 = self.reff0
        Z = grid.z - z0
        Z[Z < 0] = 0
        reff_data = (slope * Z ** (1. / 3.)) + reff0

        # reff_data[Z == 0] = 0
        reff_data[mask==0] = 0
        return shdom.GridData(grid, reff_data)

    def get_monotonous_lwc(self, old_lwc, slope, z0=None):
        mask = old_lwc.data > 0
        mask_z = np.sum(mask,(0,1))>0
        grid = old_lwc.grid
        if z0 is None:
            z0 = grid.z[mask_z][0]

        Z = grid.z - z0
        Z[Z < 0] = 0
        lwc_profile = (slope * Z ) + 0.01
        lwc_data = np.tile(lwc_profile[np.newaxis, np.newaxis, :], (grid.nx, grid.ny, 1))

        lwc_data[mask==0] = 0
        return shdom.GridData(grid, lwc_data)

    def minimize(self):
        """
        Local minimization with respect to the parameters defined.
        """
        # if self.iteration == 0:
            # self.init_optimizer()


        from scipy.optimize import minimize_scalar
        self.original_state = np.array(self.get_state())

        # # res = minimize_scalar(self.objective_fun, method='bounded', options={'maxiter':8}, bracket=(0.5,2), bounds=(0.5, 2))
        loss = np.inf
        for r0 in range(1,5):
            self.reff0=r0
            print('check r0 {}'.format(r0))
            res = minimize_scalar(self.objective_fun, method='bounded', options={'maxiter':4}, bracket=(6,8), bounds=(6, 8))

            if res.fun < loss:
                loss = res.fun
                reff_slope = res.x
                reff0 = r0
                z0 = self.z0
                print('loss = {},  reff_slope={}, reff0={}, z0={}'.format(loss, reff_slope, reff0, z0))
        print('final: loss = {},  reff_slope={}, reff0={}, z0={}'.format(loss, reff_slope, reff0, z0))

        # shdom.cloud_plot(lwc.data)
        # plt.plot(reff.data)
        # plt.show()
        return reff_slope, reff0, z0
    def set_state(self, state):
        """
        Set the state of the optimization. This means:
          1. Setting the MediumEstimator state
          2. Updating the RteSolver medium
          3. Computing the direct solar flux
          4. Computing the current RTE solution with the previous solution as an initialization

        Returns
        -------
        state: np.array(dtype=np.float64)
            The state of the medium estimator
        """
        self.medium.set_state(state)
        self.rte_solver.replace_dynamic_medium(self.medium)
        if self._init_solution is False:
            self.rte_solver.make_direct()
        self.rte_solver.solve(maxiter=20, init_solution=self._init_solution, verbose=False)
    # def grid_search(self, bounds_slope, bounds_reff0):
    #     for


class DynamicSummaryWriter(object):
    """
    A wrapper for tensorboardX summarywriter with some basic summary writing implementation.
    This wrapper enables logging of images, error measures and loss with pre-determined temporal intervals into tensorboard.

    To view the summary of this run (and comparisons to all subdirectories):
        tensorboard --logdir LOGDIR

    Parameters
    ----------
    log_dir: str
        The directory where the log will be saved
    """

    def __init__(self, log_dir=None):
        self._dir = log_dir
        self._tf_writer = tb.SummaryWriter(log_dir) if log_dir is not None else None
        self._ground_truth_parameters = None
        self._callback_fns = []
        self._kwargs = []
        self._optimizer = None

    def add_callback_fn(self, callback_fn, kwargs=None):
        """
        Add a callback function to the callback function list

        Parameters
        ----------
        callback_fn: bound method
            A callback function to push into the list
        kwargs: dict, optional
            A dictionary with optional keyword arguments for the callback function
        """
        self._callback_fns.append(callback_fn)
        self._kwargs.append(kwargs)

    def attach_optimizer(self, optimizer):
        """
        Attach the optimizer

        Parameters
        ----------
        optimizer: shdom.Optimizer
            The optimizer that the writer will report for
        """
        self._optimizer = optimizer

    def monitor_loss(self, ckpt_period=-1):
        """
        Monitor the loss.

        Parameters
        ----------
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': 'loss',
        }
        self.add_callback_fn(self.loss_cbfn, kwargs)

    def monitor_time_smoothness(self, ckpt_period=-1):
        """
        Monitor the time smoothness.

        Parameters
        ----------
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': 'time_smoothness'
        }
        self.add_callback_fn(self.time_smoothness_cbfn, kwargs)

    def save_checkpoints(self, ckpt_period=-1):
        """
        Save a checkpoint of the Optimizer

        Parameters
        ----------
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time()
        }
        self.add_callback_fn(self.save_ckpt_cbfn, kwargs)

    def monitor_state(self, ckpt_period=-1):
        """
        Monitor the state of the optimization.

        Parameters
        ----------
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        self.states = []
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time()
        }
        self.add_callback_fn(self.state_cbfn, kwargs)

    def monitor_shdom_iterations(self, ckpt_period=-1):
        """Monitor the number of SHDOM forward iterations.

        Parameters
        ----------
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': 'shdom iterations'
        }
        self.add_callback_fn(self.shdom_iterations_cbfn, kwargs)

    def monitor_scatterer_error(self, estimator_name, ground_truth, ckpt_period=-1):
        """
        Monitor relative and overall mass error (epsilon, delta) as defined at:
          Amit Aides et al, "Multi sky-view 3D aerosol distribution recovery".

        Parameters
        ----------
        estimator_name: str
            The name of the scatterer to monitor
        ground_truth: shdom.Scatterer
            The ground truth medium.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': ['{}/delta/{} at time{}', '{}/epsilon/{} at time{}']
        }
        self.add_callback_fn(self.scatterer_error_cbfn, kwargs)
        if hasattr(self, '_ground_truth'):
            self._ground_truth[estimator_name] = ground_truth
        else:
            self._ground_truth = OrderedDict({estimator_name: ground_truth})

    def monitor_scatter_plot(self, estimator_name, ground_truth, dilute_percent=0.4, ckpt_period=-1, parameters='all'):
        """
        Monitor scatter plot of the parameters

        Parameters
        ----------
        estimator_name: str
            The name of the scatterer to monitor
        ground_truth: shdom.Scatterer
            The ground truth medium.
        dilute_precent: float [0,1]
            Precentage of (random) points that will be shown on the scatter plot.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        parameters: str,
           The parameters for which to monitor scatter plots. 'all' monitors all estimated parameters.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': '{}/scatter_plot/{}{}',
            'percent': dilute_percent,
            'parameters': parameters
        }
        self.add_callback_fn(self.scatter_plot_cbfn, kwargs)
        if hasattr(self, '_ground_truth'):
            self._ground_truth[estimator_name] = ground_truth
        else:
            self._ground_truth = OrderedDict({estimator_name: ground_truth})

    def monitor_horizontal_mean(self, estimator_name, ground_truth, ground_truth_mask=None, ckpt_period=-1):
        """
        Monitor horizontally averaged quantities and compare to ground truth over iterations.

        Parameters
        ----------
        estimator_name: str
            The name of the scatterer to monitor
        ground_truth: shdom.Scatterer
            The ground truth medium.
        ground_truth_mask: shdom.GridData
            The ground-truth mask of the estimator
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': '{}/horizontal_mean/{}{}',
            'mask': ground_truth_mask
        }

        self.add_callback_fn(self.horizontal_mean_cbfn, kwargs)
        if hasattr(self, '_ground_truth'):
            self._ground_truth[estimator_name] = ground_truth
        else:
            self._ground_truth = OrderedDict({estimator_name: ground_truth})

    def monitor_domain_mean(self, estimator_name, ground_truth, ckpt_period=-1):
        """
        Monitor domain mean and compare to ground truth over iterations.

        Parameters
        ----------
        estimator_name: str
            The name of the scatterer to monitor
        ground_truth: shdom.Scatterer
            The ground truth medium.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': '{}/mean/{}'
        }
        self.add_callback_fn(self.domain_mean_cbfn, kwargs)
        if hasattr(self, '_ground_truth'):
            self._ground_truth[estimator_name] = ground_truth
        else:
            self._ground_truth = OrderedDict({estimator_name: ground_truth})

    def monitor_images(self, measurements, ckpt_period=-1):
        """
        Monitor the synthetic images and compare to the acquired images

        Parameters
        ----------
        measurements: shdom.Measurements
            The acquired images will be logged once onto tensorboard for comparison with the current state.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        acquired_images = measurements.images
        sensor_type = measurements.camera.sensor.type
        num_images = len(acquired_images)

        if sensor_type == 'RadianceSensor':
            vmax = [image.max() * 1.25 for image in acquired_images]
            vmax = max(vmax)
        elif sensor_type == 'StokesSensor':
            vmax = [image.reshape(image.shape[0], -1).max(axis=-1) * 1.25 for image in acquired_images]

        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': ['Retrieval/view{}'.format(view) for view in range(num_images)],
            'vmax': vmax
        }
        self.add_callback_fn(self.estimated_images_cbfn, kwargs)

        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': ['Diff/view{}'.format(view) for view in range(num_images)],
            'acquired_images': acquired_images,
            'vmax': vmax
        }
        self.add_callback_fn(self.diff_images_cbfn, kwargs)

        kwargs = {
            'vmax': vmax
        }
        acq_titles = ['Acquired/view{}'.format(view) for view in range(num_images)]
        self.write_image_list(0, acquired_images, acq_titles, vmax=kwargs['vmax'])

        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': 'loss/normalized loss image {}',
            'acquired_images': acquired_images,
            'vmax': vmax
        }
        self.add_callback_fn(self.loss_norm_cbfn, kwargs)

    def save_ckpt_cbfn(self, kwargs=None):
        """
        Callback function that saves checkpoints .

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        timestr = time.strftime("%H%M%S")
        path = os.path.join(self.tf_writer.logdir, timestr + '.ckpt')
        self.optimizer.save_state(path)

    def loss_cbfn(self, kwargs):
        """
        Callback function that is called (every optimizer iteration) for loss monitoring.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        if isinstance(self.optimizer.loss,list):
            self.tf_writer.add_scalars(kwargs['title'], {
                kwargs['title']: sum(self.optimizer.loss),
                'Data term loss': self.optimizer.loss[0],
                'Regularization term loss': self.optimizer.loss[1],
            }
                , self.optimizer.iteration)
        else:
            self.tf_writer.add_scalar(kwargs['Data term loss'], self.optimizer.loss, self.optimizer.iteration)

    # def loss_cbfn(self, kwargs):
    #     """
    #     Callback function that is called (every optimizer iteration) for loss monitoring.
    #
    #     Parameters
    #     ----------
    #     kwargs: dict,
    #         keyword arguments
    #     """
    #     loss = np.sum([np.sum((im1 - im2)**2) for im1, im2 in zip(kwargs['acquired_images'], self.optimizer.images)])
    #     self.tf_writer.add_scalar(kwargs['Data term loss'], diff, self.optimizer.iteration)

    def loss_norm_cbfn(self, kwargs):
        """
        Callback function that is called (every optimizer iteration) for loss monitoring.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """

        losses_norm = [np.sum((im1 - im2).ravel()**2) ** 0.5 / im1.max() / im1.size for im1, im2 in zip(kwargs['acquired_images'], self.optimizer.images)]
        for i, loss in enumerate(losses_norm):

            self.tf_writer.add_scalars('loss/normalized loss', {kwargs['title'].format(i): loss}, self.optimizer.iteration)

    def time_smoothness_cbfn(self, kwargs):
        """
        Callback function that is called (every optimizer iteration) for loss monitoring.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        extinctions=[]
        for dynamic_scatterer_name, gt_dynamic_scatterer in self._ground_truth.items():
            est_scatterer = self.optimizer.medium.get_scatterer(dynamic_scatterer_name)
            for estimator_temporary_scatterer in est_scatterer.temporary_scatterer_estimator_list:
                extinctions.append(estimator_temporary_scatterer.scatterer.extinction.data)
        err=0
        for ind, (extinction_i, extinction_j) in enumerate(zip(extinctions[:-1], extinctions[1:])):
            # err += np.linalg.norm((extinction_i - extinction_j).reshape(-1, 1), ord=2)
            err = np.linalg.norm((extinction_i - extinction_j).reshape(-1, 1), ord=2)
        # self.tf_writer.add_scalar(kwargs['title'], err, self.optimizer.iteration)
            self.tf_writer.add_scalars(
                main_tag=kwargs['title'],
                tag_scalar_dict={'{}'.format(ind): err},
                global_step=self.optimizer.iteration
            )

    def state_cbfn(self, kwargs=None):
        """
        Callback function that is called for state monitoring.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        state = np.empty(shape=(0), dtype=np.float64)
        for estimator in self.optimizer.medium.estimators.values():
            for param in estimator.estimators.values():
                state = np.concatenate((state, param.get_state() / param.precondition_scale_factor))
        self.states.append(state)

    def diff_images_cbfn(self, kwargs):
        """
        Callback function the is called every optimizer iteration image monitoring is set.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        diff = [np.abs(im1 - im2) for im1, im2 in zip(kwargs['acquired_images'], self.optimizer.images)]
        self.write_image_list(self.optimizer.iteration, diff, kwargs['title'], kwargs['vmax'])

    def estimated_images_cbfn(self, kwargs):
        """
        Callback function the is called every optimizer iteration image monitoring is set.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        self.write_image_list(self.optimizer.iteration, self.optimizer.images, kwargs['title'], kwargs['vmax'])

    def shdom_iterations_cbfn(self, kwargs):
        """
        Callback function that is called (every optimizer iteration) for shdom iteration monitoring.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        self.tf_writer.add_scalar(kwargs['title'], self.optimizer.rte_solver.num_iterations, self.optimizer.iteration)

    def scatterer_error_cbfn(self, kwargs):
        """
        Callback function for monitoring parameter error measures.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        for dynamic_scatterer_name, gt_dynamic_scatterer in self._ground_truth.items():
            est_scatterer = self.optimizer.medium.get_scatterer(dynamic_scatterer_name)
            eps_list =[]
            delta_list=[]
            lwc_list = []
            est_temporary_scatterer_estimator_list = []
            est_temporary_scatterer_estimator_list += est_scatterer.temporary_scatterer_estimator_list
            if len(est_temporary_scatterer_estimator_list) < len(gt_dynamic_scatterer.temporary_scatterer_list):
                est_temporary_scatterer_estimator_list *= len(gt_dynamic_scatterer.temporary_scatterer_list)
            for gt_temporary_scatterer, estimator_temporary_scatterer in \
                    zip(gt_dynamic_scatterer.temporary_scatterer_list,
                                  est_temporary_scatterer_estimator_list):
                if estimator_temporary_scatterer.type =='MicrophysicalScatterer':
                    for wl in gt_temporary_scatterer.scatterer._wavelength:
                        gt_extinction = gt_temporary_scatterer.scatterer.get_extinction(wl)
                        est_extinction = estimator_temporary_scatterer.scatterer.get_extinction(wl).resample(gt_extinction.grid)
                        gt_extinction = gt_extinction.data.flatten()
                        est_extinction = est_extinction.data.flatten()

                        delta = (np.linalg.norm(est_extinction, 1) - np.linalg.norm(gt_extinction, 1)) / np.linalg.norm(gt_extinction, 1)
                        epsilon = np.linalg.norm((est_extinction - gt_extinction), 1) / np.linalg.norm(gt_extinction, 1)
                        self.tf_writer.add_scalar(
                            kwargs['title'][0].format(dynamic_scatterer_name, 'Extinction {}'.format(wl), gt_temporary_scatterer.time),
                            delta,
                            self.optimizer.iteration)
                        self.tf_writer.add_scalar(
                            kwargs['title'][1].format(dynamic_scatterer_name, 'Extinction {}'.format(wl), gt_temporary_scatterer.time),
                            epsilon,
                            self.optimizer.iteration)
                lwc_mask = None
                for parameter_name, parameter in estimator_temporary_scatterer.scatterer.estimators.items():
                    gt_param = getattr(gt_temporary_scatterer.scatterer, parameter_name)
                    est_param = parameter.resample(gt_param.grid).data.flatten()
                    gt_param = gt_param.data.flatten()
                    if parameter_name=='lwc':
                        lwc_mask = (est_param>0.01)
                    else:
                        if lwc_mask is not None:
                            est_param[lwc_mask] = gt_param[lwc_mask]

                # gt_param = gt_temporary_scatterer.scatterer.extinction
                # est_param = estimator_temporary_scatterer.scatterer.extinction.resample(gt_param.grid).data.flatten()


                    delta = (np.linalg.norm(est_param, 1) - np.linalg.norm(gt_param, 1)) / np.linalg.norm(gt_param, 1)
                    epsilon = np.linalg.norm((est_param - gt_param), 1) / np.linalg.norm(gt_param, 1)
                    self.tf_writer.add_scalar(kwargs['title'][0].format(dynamic_scatterer_name, parameter_name, gt_temporary_scatterer.time), delta,
                                              self.optimizer.iteration)
                    self.tf_writer.add_scalar(kwargs['title'][1].format(dynamic_scatterer_name, parameter_name, gt_temporary_scatterer.time), epsilon,
                                              self.optimizer.iteration)
                    eps_list.append(epsilon)
                    delta_list.append(delta)
            self.tf_writer.add_scalar(
                'eps', np.mean(eps_list),
                self.optimizer.iteration)
            self.tf_writer.add_scalar(
                'delta', np.mean(np.abs(delta_list)),
                self.optimizer.iteration)

    def domain_mean_cbfn(self, kwargs):
        """
        Callback function for monitoring domain averages of parameters.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """

        for dynamic_scatterer_name, gt_dynamic_scatterer in self._ground_truth.items():
            est_scatterer = self.optimizer.medium.get_scatterer(dynamic_scatterer_name)
            parm_scatterer = est_scatterer.temporary_scatterer_estimator_list[0]
            for parameter_name, parameter in parm_scatterer.scatterer.estimators.items():
                for ind, gt_temporary_scatterer in \
                        enumerate(gt_dynamic_scatterer.temporary_scatterer_list):
                    gt_param = getattr(gt_temporary_scatterer.scatterer, parameter_name)
                    gt_param_mean = gt_param.data.mean()
                    self.tf_writer.add_scalars(
                        main_tag=kwargs['title'].format(dynamic_scatterer_name, parameter_name),
                        tag_scalar_dict={'true{}'.format(ind): gt_param_mean},
                        global_step=self.optimizer.iteration
                    )
                for ind, estimator_temporary_scatterer in \
                        enumerate( est_scatterer.temporary_scatterer_estimator_list):
                    est_param = getattr(estimator_temporary_scatterer.scatterer, parameter_name).resample(gt_param.grid)
                    est_param_mean = est_param.data.mean()
                    self.tf_writer.add_scalars(
                        main_tag=kwargs['title'].format(dynamic_scatterer_name, parameter_name),
                        tag_scalar_dict={'estimated{}'.format(ind): est_param_mean},
                        global_step=self.optimizer.iteration
                    )

    def horizontal_mean_cbfn(self, kwargs):
        """
        Callback function for monitoring horizontal averages of parameters.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """

        for dynamic_scatterer_name, gt_dynamic_scatterer in self._ground_truth.items():
            est_scatterer = self.optimizer.medium.get_scatterer(dynamic_scatterer_name)
            parm_scatterer = est_scatterer.temporary_scatterer_estimator_list[0]
            for parameter_name, parameter in parm_scatterer.scatterer.estimators.items():
                est_temporary_scatterer_estimator_list = []
                est_temporary_scatterer_estimator_list +=( est_scatterer.temporary_scatterer_estimator_list)
                if len(est_temporary_scatterer_estimator_list)<len(gt_dynamic_scatterer.temporary_scatterer_list):
                    est_temporary_scatterer_estimator_list *= len(gt_dynamic_scatterer.temporary_scatterer_list)
                for ind, (gt_temporary_scatterer, estimator_temporary_scatterer) in \
                        enumerate(zip(gt_dynamic_scatterer.temporary_scatterer_list,
                                est_temporary_scatterer_estimator_list)):
                    common_grid = estimator_temporary_scatterer.scatterer.grid + gt_temporary_scatterer.scatterer.grid
                    a = estimator_temporary_scatterer.scatterer.get_mask(threshold=0.0).resample(common_grid, method='nearest')
                    b = gt_temporary_scatterer.scatterer.get_mask(threshold=0.0).resample(common_grid, method='nearest')
                    common_mask = shdom.GridData(data=np.bitwise_or(a.data, b.data), grid=common_grid)
                    gt_param = getattr(gt_temporary_scatterer.scatterer, parameter_name)
                    est_parameter = getattr(estimator_temporary_scatterer.scatterer, parameter_name)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)

                        est_parameter_masked = copy.deepcopy(est_parameter).resample(common_grid)
                        est_parameter_masked.apply_mask(common_mask)
                        est_param = est_parameter_masked.data
                        est_param[np.bitwise_not(common_mask.data)] = np.nan
                        est_param_mean = np.nan_to_num(np.nanmean(est_param, axis=(0, 1)))

                        gt_param_masked = copy.deepcopy(gt_param).resample(common_grid)
                        gt_param_masked.apply_mask(common_mask)
                        gt_param = gt_param_masked.data
                        gt_param[np.bitwise_not(common_mask.data)] = np.nan
                        gt_param_mean = np.nan_to_num(np.nanmean(gt_param, axis=(0, 1)))
                        # if parameter.type == 'Homogeneous' or parameter.type == '1D':
                        #     est_param = getattr(estimator_temporary_scatterer.scatterer, parameter_name).resample(
                        #         gt_param.grid)
                        #     est_param_mean = est_param.data
                        # else:
                        #     est_param = getattr(estimator_temporary_scatterer.scatterer, parameter_name).resample(
                        #         gt_param.grid)
                        #     est_param_data = copy.copy(est_param.data)
                        #     est_param_data[(estimator_temporary_scatterer.scatterer.mask.resample(gt_param.grid).data == False)] = np.nan
                        #     est_param_mean = np.nan_to_num(np.nanmean(est_param_data, axis=(0, 1)))
                        # if gt_param.type == 'Homogeneous' or gt_param.type == '1D':
                        #         gt_param_mean = gt_param.data
                        # else:
                        #
                        #     gt_param_data = copy.copy(gt_param.data)
                        #     if kwargs['mask']:
                        #         gt_param_data[kwargs['mask'][ind].data == False] = np.nan
                        #         # gt_param[estimator_temporary_scatterer.scatterer.mask.data == False] = np.nan
                        #     gt_param_mean = np.nan_to_num(np.nanmean(gt_param_data, axis=(0, 1)))

                        fig, ax = plt.subplots()
                        ax.set_title('{} {} {}'.format(dynamic_scatterer_name, parameter_name, ind), fontsize=16)
                        ax.plot(est_param_mean, common_grid.z, label='Estimated')
                        ax.plot(gt_param_mean, common_grid.z, label='True')
                        ax.legend()
                        ax.set_ylabel('Altitude [km]', fontsize=14)
                        self.tf_writer.add_figure(
                            tag=kwargs['title'].format(dynamic_scatterer_name, parameter_name, ind),
                            figure=fig,
                            global_step=self.optimizer.iteration
                        )

    def scatter_plot_cbfn(self, kwargs):
        """
        Callback function for monitoring scatter plot of parameters.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        for dynamic_scatterer_name, gt_dynamic_scatterer in self._ground_truth.items():
            est_scatterer = self.optimizer.medium.get_scatterer(dynamic_scatterer_name)
            parm_scatterer = est_scatterer.temporary_scatterer_estimator_list[0]
            for parameter_name, parameter in parm_scatterer.scatterer.estimators.items():
                est_temporary_scatterer_estimator_list = []
                est_temporary_scatterer_estimator_list += est_scatterer.temporary_scatterer_estimator_list
                if len(est_temporary_scatterer_estimator_list) < len(gt_dynamic_scatterer.temporary_scatterer_list):
                    est_temporary_scatterer_estimator_list *= len(gt_dynamic_scatterer.temporary_scatterer_list)
                for ind, (gt_temporary_scatterer, estimator_temporary_scatterer) in \
                        enumerate(zip(gt_dynamic_scatterer.temporary_scatterer_list,
                                      est_temporary_scatterer_estimator_list)):
                    gt_param = getattr(gt_temporary_scatterer.scatterer, parameter_name)
                    est_param = getattr(estimator_temporary_scatterer.scatterer, parameter_name).resample(
                        gt_param.grid)
                    est_param_data = copy.copy(est_param.data)
                    est_param_data = est_param_data[
                        (estimator_temporary_scatterer.scatterer.mask.resample(gt_param.grid).data == True)].ravel()

                    gt_param_data = copy.copy(gt_param.data)
                    gt_param_data = gt_param_data[
                        (estimator_temporary_scatterer.scatterer.mask.resample(gt_param.grid).data == True)].ravel()

                    rho = np.corrcoef(est_param_data, gt_param_data)[1, 0]
                    num_params = gt_param_data.size
                    rand_ind = np.unique(np.random.randint(0, num_params, int(kwargs['percent'] * num_params)))
                    max_val = max(gt_param_data.max(), est_param_data.max())
                    fig, ax = plt.subplots()
                    ax.set_title(r'{} {}{}: ${:1.0f}\%$ randomly sampled; $\rho={:1.2f}$'.format(dynamic_scatterer_name, parameter_name, ind, 100 * kwargs['percent'], rho),
                                 fontsize=16)
                    ax.scatter(gt_param_data[rand_ind], est_param_data[rand_ind], facecolors='none', edgecolors='b')
                    ax.set_xlim([0, 1.1*max_val])
                    ax.set_ylim([0, 1.1*max_val])
                    ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
                    ax.set_ylabel('Estimated', fontsize=14)
                    ax.set_xlabel('True', fontsize=14)

                    self.tf_writer.add_figure(
                        tag=kwargs['title'].format(dynamic_scatterer_name, parameter_name, ind),
                        figure=fig,
                        global_step=self.optimizer.iteration
                    )

    def write_image_list(self, global_step, images, titles, vmax=None):
        """
        Write an image list to tensorboardX.

        Parameters
        ----------
        global_step: integer,
            The global step of the optimizer.
        images: list
            List of images to be logged onto tensorboard.
        titles: list
            List of strings that will title the corresponding images on tensorboard.
        vmax: list or scalar, optional
            List or a single of scaling factor for the image contrast equalization
        """
        if np.isscalar(vmax) or vmax is None:
            vmax = [vmax] * len(images)

        assert len(images) == len(titles), 'len(images) != len(titles): {} != {}'.format(len(images), len(titles))
        assert len(vmax) == len(titles), 'len(vmax) != len(images): {} != {}'.format(len(vmax), len(titles))

        for image, title, vm in zip(images, titles, vmax):

            # for polarization
            if image.ndim == 4:
                stoke_title = ['V', 'U', 'Q', 'I']
                for v, stokes in zip(vm, image):
                    self.tf_writer.add_images(
                        tag=title + '/' + stoke_title.pop(),
                        img_tensor=(np.repeat(np.expand_dims(stokes, 2), 3, axis=2) / v),
                        dataformats='HWCN',
                        global_step=global_step
                    )

            # for polychromatic
            elif image.ndim == 3:
                self.tf_writer.add_images(
                    tag=title,
                    img_tensor=(np.repeat(np.expand_dims(image, 2), 3, axis=2) / vm),
                    dataformats='HWCN',
                    global_step=global_step
                )
            # for monochromatic
            else:
                self.tf_writer.add_image(
                    tag=title,
                    img_tensor=(image / vm),
                    dataformats='HW',
                    global_step=global_step
                )
    def monitor_images_scatter_plot(self, measurements, dilute_percent=0.1, ckpt_period=-1):
        """
        Monitor the Cross Validation process

        Parameters
        ----------
        cv_measurement: shdom.DynamicMeasurements
            The acquired images will be logged once onto tensorboard for comparison with the current state.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        acquired_images = measurements.images
        num_images = len(acquired_images)

        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': ['Image_Scatter_Plot/view{}'.format(view) for view in range(num_images)],
            'acquired_images': acquired_images,
            'percent': dilute_percent,
        }
        self.add_callback_fn(self.images_scatter_plot_cbfn, kwargs)

    def images_scatter_plot_cbfn(self, kwargs):
        for estimated_image, acquired_images,title in zip(self.optimizer.images, kwargs['acquired_images'],kwargs['title']):
            estimated_image = copy.copy(estimated_image).ravel()
            acquired_images = copy.copy(acquired_images).ravel()
            rho = np.corrcoef(estimated_image, acquired_images)[1, 0]
            num_params = acquired_images.size
            rand_ind = np.unique(np.random.randint(0, num_params, int(kwargs['percent'] * num_params)))
            max_val = max(acquired_images.max(), estimated_image.max())
            fig, ax = plt.subplots()
            ax.set_title(
                r'{}: ${:1.0f}\%$ randomly sampled; $\rho={:1.2f}$'.format(0, 100 * kwargs['percent'], rho),
                fontsize=16)
            ax.scatter(acquired_images[rand_ind], estimated_image[rand_ind], facecolors='none', edgecolors='b')
            ax.set_xlim([0, 1.1 * max_val])
            ax.set_ylim([0, 1.1 * max_val])
            ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
            ax.set_ylabel('Estimated', fontsize=14)
            ax.set_xlabel('True', fontsize=14)

            self.tf_writer.add_figure(
                tag=title,
                figure=fig,
                global_step=self.optimizer.iteration
            )

    def monitor_cross_validation(self, cv_measurement, dilute_percent=0.1, ckpt_period=-1):
        """
        Monitor the Cross Validation process

        Parameters
        ----------
        cv_measurement: shdom.DynamicMeasurements
            The acquired images will be logged once onto tensorboard for comparison with the current state.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        acquired_images = cv_measurement.images
        sensor_type = cv_measurement.camera.sensor.type
        num_images = len(acquired_images)

        if sensor_type == 'RadianceSensor':
            vmax = [image.max() * 1.25 for image in acquired_images]
        elif sensor_type == 'StokesSensor':
            vmax = [image.reshape(image.shape[0], -1).max(axis=-1) * 1.25 for image in acquired_images]

        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': ['Cross Validation Retrieval/view{}'.format(view) for view in range(num_images)],
            'vmax': vmax
        }
        self.add_callback_fn(self.estimated_cross_validation_images_cbfn, kwargs)
        acq_titles = ['Cross Validation Acquired/view{}'.format(view) for view in range(num_images)]
        self.write_image_list(0, acquired_images, acq_titles, vmax=kwargs['vmax'])

        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': 'Cross Validation/loss',
        }
        self.add_callback_fn(self.cv_loss_cbfn, kwargs)

        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': 'Cross Validation/image scatter plot',
            'percent': dilute_percent,
        }
        self.add_callback_fn(self.cv_image_scatter_plot_cbfn, kwargs)

    def cv_loss_cbfn(self, kwargs):
        """
        Callback function that is called (every optimizer iteration) for loss monitoring.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        self.tf_writer.add_scalar(kwargs['title'], self.optimizer.cv_loss, self.optimizer.iteration)

    def estimated_cross_validation_images_cbfn(self, kwargs):
        """
        Callback function the is called every optimizer iteration image monitoring is set.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        self.write_image_list(self.optimizer.iteration, self.optimizer.get_cross_validation_images(), kwargs['title'], kwargs['vmax'])

    def cv_image_scatter_plot_cbfn(self, kwargs):
        cv_image_estimated = np.array(self.optimizer.get_cross_validation_images()).ravel()
        cv_image = np.array(self.optimizer._cv_measurement.images).ravel()
        rho = np.corrcoef(cv_image_estimated, cv_image)[1, 0]
        num_params = cv_image.size
        rand_ind = np.unique(np.random.randint(0, num_params, int(kwargs['percent'] * num_params)))
        max_val = max(cv_image.max(), cv_image_estimated.max())
        fig, ax = plt.subplots()
        ax.set_title(
            r'{}: ${:1.0f}\%$ randomly sampled; $\rho={:1.2f}$'.format(0, 100 * kwargs['percent'], rho),
            fontsize=16)
        ax.scatter(cv_image[rand_ind], cv_image_estimated[rand_ind], facecolors='none', edgecolors='b')
        ax.set_xlim([0, 1.1 * max_val])
        ax.set_ylim([0, 1.1 * max_val])
        ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
        ax.set_ylabel('Estimated', fontsize=14)
        ax.set_xlabel('True', fontsize=14)

        self.tf_writer.add_figure(
            tag=kwargs['title'],
            figure=fig,
            global_step=self.optimizer.iteration
        )

    def monitor_cross_validation_scatterer_error(self, estimator_name, cv_ground_truth, ckpt_period=-1):
        """
        Monitor relative and overall mass error (epsilon, delta) as defined at:
          Amit Aides et al, "Multi sky-view 3D aerosol distribution recovery".

        Parameters
        ----------
        estimator_name: str
            The name of the scatterer to monitor
        ground_truth: shdom.Scatterer
            The ground truth medium.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': ['Cross Validation/delta {} at time {}', 'Cross Validation/epsilon {} at time {}']
        }
        self.add_callback_fn(self.cross_validation_scatterer_error_cbfn, kwargs)
        if hasattr(self, '_cv_ground_truth'):
            self._cv_ground_truth[estimator_name] = cv_ground_truth
        else:
            self._cv_ground_truth = OrderedDict({estimator_name: cv_ground_truth})

    def cross_validation_scatterer_error_cbfn(self, kwargs):
        """
        Callback function for monitoring parameter error measures.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        for dynamic_scatterer_name, gt_temporary_scatterer in self._cv_ground_truth.items():
            est_scatterer = self.optimizer.get_cross_validation_medium(dynamic_scatterer_name).medium_list[0].estimators[dynamic_scatterer_name]
            for parameter_name, parameter in est_scatterer.estimators.items():
                gt_param = getattr(gt_temporary_scatterer.scatterer, parameter_name)
                est_param = parameter.resample(gt_param.grid).data.flatten()
            # gt_param = gt_temporary_scatterer.scatterer.extinction
            # est_param = estimator_temporary_scatterer.scatterer.extinction.resample(gt_param.grid).data.flatten()
                gt_param = gt_param.data.flatten()

                delta = (np.linalg.norm(est_param, 1) - np.linalg.norm(gt_param, 1)) / np.linalg.norm(gt_param, 1)
                epsilon = np.linalg.norm((est_param - gt_param), 1) / np.linalg.norm(gt_param, 1)
                self.tf_writer.add_scalar(kwargs['title'][0].format(parameter_name, gt_temporary_scatterer.time), delta,
                                          self.optimizer.iteration)
                self.tf_writer.add_scalar(kwargs['title'][1].format(parameter_name, gt_temporary_scatterer.time), epsilon,
                                          self.optimizer.iteration)

    def monitor_cross_validation_scatter_plot(self, estimator_name, cv_ground_truth, dilute_percent=0.4, ckpt_period=-1, parameters='all'):
        """
        Monitor scatter plot of the parameters

        Parameters
        ----------
        estimator_name: str
            The name of the scatterer to monitor
        ground_truth: shdom.Scatterer
            The ground truth medium.
        dilute_precent: float [0,1]
            Precentage of (random) points that will be shown on the scatter plot.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        parameters: str,
           The parameters for which to monitor scatter plots. 'all' monitors all estimated parameters.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': 'Cross Validation {}/scatter_plot/{}{}',
            'percent': dilute_percent,
            'parameters': parameters
        }
        self.add_callback_fn(self.cross_validation_scatter_plot_cbfn, kwargs)
        if hasattr(self, '_ground_truth'):
            self._cv_ground_truth[estimator_name] = cv_ground_truth
        else:
            self._cv_ground_truth = OrderedDict({estimator_name: cv_ground_truth})

    def cross_validation_scatter_plot_cbfn(self, kwargs):
        """
        Callback function for monitoring scatter plot of parameters.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        for dynamic_scatterer_name, gt_temporary_scatterer in self._cv_ground_truth.items():
            est_scatterer = self.optimizer.get_cross_validation_medium(dynamic_scatterer_name).medium_list[0].estimators[dynamic_scatterer_name]
            for parameter_name, parameter in est_scatterer.estimators.items():

                    gt_param = getattr(gt_temporary_scatterer.scatterer, parameter_name)
                    est_param = parameter.resample(gt_param.grid)
                    est_param_data = copy.copy(est_param.data)
                    est_param_data = est_param_data[
                        (est_param.mask.resample(gt_param.grid).data == True)].ravel()

                    gt_param_data = copy.copy(gt_param.data)
                    gt_param_data = gt_param_data[
                        (est_param.mask.resample(gt_param.grid).data == True)].ravel()

                    rho = np.corrcoef(est_param_data, gt_param_data)[1, 0]
                    num_params = gt_param_data.size
                    rand_ind = np.unique(np.random.randint(0, num_params, int(kwargs['percent'] * num_params)))
                    max_val = max(gt_param_data.max(), est_param_data.max())
                    fig, ax = plt.subplots()
                    ax.set_title(r'{} {}{}: ${:1.0f}\%$ randomly sampled; $\rho={:1.2f}$'.format(dynamic_scatterer_name, parameter_name, 0, 100 * kwargs['percent'], rho),
                                 fontsize=16)
                    ax.scatter(gt_param_data[rand_ind], est_param_data[rand_ind], facecolors='none', edgecolors='b')
                    ax.set_xlim([0, 1.1*max_val])
                    ax.set_ylim([0, 1.1*max_val])
                    ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
                    ax.set_ylabel('Estimated', fontsize=14)
                    ax.set_xlabel('True', fontsize=14)

                    self.tf_writer.add_figure(
                        tag=kwargs['title'].format(dynamic_scatterer_name, parameter_name, 0),
                        figure=fig,
                        global_step=self.optimizer.iteration
                    )

    def monitor_save3d(self, ckpt_period=-1):
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
        }
        self.add_callback_fn(self.save3d_cbfn, kwargs)

    def save3d_cbfn(self, kwargs):
        estimated_extinction_stack = []
        estimated_gridx_stack = []
        estimated_gridy_stack = []
        estimated_gridz_stack = []

        estimated_dynamic_medium = self.optimizer.medium.medium_list
        for medium_estimator in estimated_dynamic_medium:
            for estimator_name, estimator in medium_estimator.estimators.items():
                estimated_extinction = estimator.extinction
                estimated_extinction_stack.append(estimated_extinction.data)
                estimated_gridx_stack.append(estimated_extinction.grid.x)
                estimated_gridy_stack.append(estimated_extinction.grid.y)
                estimated_gridz_stack.append(estimated_extinction.grid.z)

        estimated_extinction_stack = np.stack(estimated_extinction_stack, axis=3)
        estimated_gridx_stack = np.stack(estimated_gridx_stack, axis=1)
        estimated_gridy_stack = np.stack(estimated_gridy_stack, axis=1)
        estimated_gridz_stack = np.stack(estimated_gridz_stack, axis=1)

        sio.savemat(os.path.join(self._dir, 'FINAL_3D_{}.mat'.format('extinction')),
                    {'estimated_extinction': estimated_extinction_stack, 'x': estimated_gridx_stack, 'y': estimated_gridy_stack, 'z': estimated_gridz_stack, 'iteration': self.optimizer.iteration})

    @property
    def callback_fns(self):
        return self._callback_fns

    @property
    def dir(self):
        return self._dir

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def tf_writer(self):
        return self._tf_writer


class DynamicSpaceCarver(object):
    """
    SpaceCarver object recovers the convex hull of the cloud based on multi-view sensor geometry and pixel segmentation.

    Parameters
    ----------
    measurements: shdom.Measurements
        A measurements object storing the images and sensor geometry
    """

    def __init__(self, measurements):

        self._rte_solver = shdom.RteSolver(shdom.SceneParameters(), shdom.NumericalParameters())

        self._measurements = measurements

        # if isinstance(measurements.camera.projection, shdom.MultiViewProjection):
        #     self._projections = measurements.camera.dynamic_projection.multiview_projection_list
        # elif isinstance(measurements.camera.projection, list):
        #     self._projections = measurements.camera.projection
        # else:
        #     self._projections = [measurements.camera.projection]
        self._projections = measurements.camera.dynamic_projection.multiview_projection_list
        self._images = measurements.images

    def carve(self, grid, thresholds, time_list, agreement=0.75, vx_max=5, vy_max=5, gt_velocity = None, verbose=False):
        """
        Carves out the cloud geometry on the grid.
        A threshold on radiances is used to produce a pixel mask and preform space carving.

        Parameters
        ----------
        grid: shdom.Grid
            A grid object.
        thresholds: list or float
            Either a constant threshold or a list of len(thresholds)=num_projections is used as for masking.
        agreement: float
            the precentage of pixels that should agree on a cloudy voxels to set it to True in the mask

        Returns
        -------
        mask: shdom.GridData object
            A boolean mask with True marking cloudy voxels and False marking non-cloud region.

        Notes
        -----
        Currently ignores stokes/multispectral measurements and uses only I component and the last channel to retrieve a cloud mask.
        """


        thresholds = np.array(thresholds)
        if thresholds.size == 1:
            thresholds = np.repeat(thresholds, len(self._images))
        else:
            assert thresholds.size == len(self._images), 'thresholds (len={}) should be of the same' \
                                                         'length as the number of images (len={})'.format(
                thresholds.size, len(self._images))

        best_match = -np.inf

        if gt_velocity is None:
            vx_vec = np.linspace(-vx_max, vx_max, num=5)
            vy_vec = np.linspace(-vy_max, vy_max, num=5)
        else:
            vx_vec = [gt_velocity[0]]
            vy_vec = [gt_velocity[1]]
        first = True
        projections = []
        times = []
        for projection,time in zip(self._projections,time_list):
            if isinstance(projection,shdom.MultiViewProjection):
                projections += projection.projection_list
                times += [time]*len(projection.projection_list)
            else:
                projections += [projection]
                times += [time]
        for vx in vx_vec:
            for vy in vy_vec:
                dynamic_grid = []
                volume = np.zeros((grid.nx, grid.ny, grid.nz))
                for projection, image, threshold, time in zip(projections, self._images, thresholds,times):
                    shift = 1e-3 * time * np.array([vx,vy,0]) #km
                    shifted_grid = shdom.Grid(x=grid.x + shift[0], y=grid.y + shift[1],
                               z=grid.z + shift[2])
                    self._rte_solver.set_grid(shifted_grid)

                    if self._measurements.num_channels > 1:
                        image = image[..., -1]
                    if self._measurements.camera.sensor.type == 'StokesSensor':
                        image = image[0]
                    if threshold < 0:
                        threshold = filters.threshold_isodata(image)

                    image_mask = image > threshold
                    if verbose and first:
                        plt.imshow(image_mask)
                        plt.show()
                    projection = projection[image_mask.ravel(order='F')]

                    carved_volume = shdom.core.space_carve(
                        nx=grid.nx,
                        ny=grid.ny,
                        nz=grid.nz,
                        npts=self._rte_solver._npts,
                        ncells=self._rte_solver._ncells,
                        gridptr=self._rte_solver._gridptr,
                        neighptr=self._rte_solver._neighptr,
                        treeptr=self._rte_solver._treeptr,
                        cellflags=self._rte_solver._cellflags,
                        bcflag=self._rte_solver._bcflag,
                        ipflag=self._rte_solver._ipflag,
                        xgrid=self._rte_solver._xgrid,
                        ygrid=self._rte_solver._ygrid,
                        zgrid=self._rte_solver._zgrid,
                        gridpos=self._rte_solver._gridpos,
                        camx=projection.x,
                        camy=projection.y,
                        camz=projection.z,
                        cammu=projection.mu,
                        camphi=projection.phi,
                        npix=projection.npix,
                    )
                    volume += carved_volume.reshape(grid.nx, grid.ny, grid.nz)
                    dynamic_grid.append(shdom.GridData(shifted_grid, volume).grid)
                volume = volume * 1.0 / len(self._images)
                match = np.sum(volume > agreement)
                first = False

                if match > best_match:
                    best_match = match
                    cloud_velocity = [vx,vy,0]
                    mask = volume > agreement
                    mask[0, :, :] = 0
                    mask[:, 0, :] = 0
                    mask[:, :, 0] = 0
                    mask[-1, :, :] = 0
                    mask[:, -1, :] = 0
                    mask[:, :, -1] = 0
                    best_dynamic_grid = dynamic_grid

        mask_list = []
        for grid in best_dynamic_grid:
            mask_list.append(shdom.GridData(grid, mask))

        return mask_list, best_dynamic_grid, cloud_velocity


    def carve1(self, grid, thresholds, agreement=0.75, verbose=False):
        """
        Carves out the cloud geometry on the grid.
        A threshold on radiances is used to produce a pixel mask and preform space carving.

        Parameters
        ----------
        grid: shdom.Grid
            A grid object.
        thresholds: list or float
            Either a constant threshold or a list of len(thresholds)=num_projections is used as for masking.
        agreement: float
            the precentage of pixels that should agree on a cloudy voxels to set it to True in the mask

        Returns
        -------
        mask: shdom.GridData object
            A boolean mask with True marking cloudy voxels and False marking non-cloud region.

        Notes
        -----
        Currently ignores stokes/multispectral measurements and uses only I component and the last channel to retrieve a cloud mask.
        """


        thresholds = np.array(thresholds)
        if thresholds.size == 1:
            thresholds = np.repeat(thresholds, len(self._images))
        else:
            assert thresholds.size == len(self._images), 'thresholds (len={}) should be of the same' \
                                                         'length as the number of images (len={})'.format(
                thresholds.size, len(self._images))

        dynamic_grid = []
        volumes = []
        for projection, image, threshold in zip(self._projections, self._images, thresholds):
            self._rte_solver.set_grid(grid)

            if self._measurements.num_channels > 1:
                image = image[..., -1]
            if self._measurements.camera.sensor.type == 'StokesSensor':
                image = image[0]
            if threshold < 0:
                threshold = filters.threshold_isodata(image)

            image_mask = image > threshold
            if verbose:
                plt.imshow(image_mask)
                plt.show()
            projection = projection[image_mask.ravel(order='F')]

            carved_volume = shdom.core.space_carve(
                nx=grid.nx,
                ny=grid.ny,
                nz=grid.nz,
                npts=self._rte_solver._npts,
                ncells=self._rte_solver._ncells,
                gridptr=self._rte_solver._gridptr,
                neighptr=self._rte_solver._neighptr,
                treeptr=self._rte_solver._treeptr,
                cellflags=self._rte_solver._cellflags,
                bcflag=self._rte_solver._bcflag,
                ipflag=self._rte_solver._ipflag,
                xgrid=self._rte_solver._xgrid,
                ygrid=self._rte_solver._ygrid,
                zgrid=self._rte_solver._zgrid,
                gridpos=self._rte_solver._gridpos,
                camx=projection.x,
                camy=projection.y,
                camz=projection.z,
                cammu=projection.mu,
                camphi=projection.phi,
                npix=projection.npix,
            )
            volume = (carved_volume.reshape(grid.nx, grid.ny, grid.nz))
            volumes.append(volume)
        volume = np.sum(volumes,axis=0) * 1.0 / len(self._images)
        mask = volume > agreement
        mask[0, :, :] = 0
        mask[:, 0, :] = 0
        mask[:, :, 0] = 0
        mask[-1, :, :] = 0
        mask[:, -1, :] = 0
        mask[:, :, -1] = 0
        #
        #
        mask_list = []
        # for grid in best_dynamic_grid:
        #     mask_list.append(shdom.GridData(grid, mask))

        return mask

    @property
    def grid(self):
        return self._grid


def cloud_plot(a):
    x = np.arange(a.shape[0])[:, None, None]
    y = np.arange(a.shape[1])[None, :, None]
    z = np.arange(a.shape[2])[None, None, :]
    a = a.astype('float')
    a[a == 0] = float('nan')
    x, y, z = np.broadcast_arrays(x, y, z)
    fig = plt.figure()
    ax = Axes3D(fig)
    pnt3d = ax.scatter(x.ravel(),
                       y.ravel(),
                       z.ravel(),
                       c=a.ravel())
    cbar = plt.colorbar(pnt3d)
    cbar.set_label("Values (units)")
    plt.show()

