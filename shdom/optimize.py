"""
Optimization and related objects to monitor and log the optimization process.
"""
import numpy as np
import time, os, copy, shutil
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import shdom
from shdom import GridData, core, float_round
import dill as pickle
import itertools
from joblib import Parallel, delayed
from collections import OrderedDict
import tensorboardX as tb
import matplotlib.pyplot as plt
import warnings
from scipy import stats
from scipy import sparse


class OpticalScattererDerivative(shdom.OpticalScatterer):
    """
    An OpticalScattererDerivative object.
    Essentially identical to a shdom.OpticalScatterer with no restrictions on negative extinction or albedo values outside of [0,1].

    Parameters
    ----------
    wavelength: float
        A wavelength in microns
    extinction: shdom.GridData object
        A GridData object containing the extinction (1/km) on a grid
    albedo: shdom.GridData
        A GridData object containing the single scattering albedo [0,1] on a grid
    phase: shdom.GridPhase
        A GridPhase object containing the phase function on a grid
    """
    def __init__(self, wavelength, extinction=None, albedo=None, phase=None):
        super().__init__(wavelength, extinction, albedo, phase)

    def resample(self, grid):
        """
        The resample method resamples the OpticalScatterer (extinction, albedo, phase).

        Parameters
        ----------
        grid: shdom.Grid
            The new grid to which the data will be resampled

        Returns
        -------
        scatterer: shdom.OpticalScatterer
            An optical scatterer resampled onto the input grid
        """
        extinction = self.extinction.resample(grid)
        albedo = self.albedo.resample(grid)
        phase = self.phase.resample(grid)
        return shdom.OpticalScattererDerivative(self.wavelength, extinction, albedo, phase)

    @property
    def extinction(self):
        return self._extinction

    @extinction.setter
    def extinction(self, val):
        self._extinction = val

    @property
    def albedo(self):
        return self._albedo

    @albedo.setter
    def albedo(self, val):
        self._albedo = val


class GridPhaseEstimator(shdom.GridPhase):
    """
    A GridPhaseEstimator.

    Notes
    -----
    A dummy class, currently not implemented.
    """
    def __init__(self, legendre_table, index):
        super().__init__(legendre_table, index)


class GridDataEstimator(shdom.GridData):
    """
    A GridDataEstimator defines unknown shdom.GridData to be estimated.

    Parameters
    ----------
    grid_data: shdom.GridData
        The initial guess for the estimator
    min_bound: float, optional
        A lower bound for the parameter values
    max_bound: float, optional
        An upper bound for the parameter values
    """
    def __init__(self, grid_data, min_bound=None, max_bound=None, precondition_scale_factor=1.0):
        super().__init__(grid_data.grid, grid_data.data)
        self._min_bound = min_bound
        self._max_bound = max_bound
        self._mask = None
        self._num_parameters = self.init_num_parameters()
        self._precondition_scale_factor = precondition_scale_factor

    def set_state(self, state):
        """
        Set the estimator state.

        Parameters
        ----------
        state: np.array(dtype=np.float64)
            The state to set the estimator data (grid is left unchanged)

        Notes
        -----
        The state is scaled back by the preconditioning scale factor
        If the estimator has a mask, data point outside of the mask are left uneffected.
        """
        state = state / self.precondition_scale_factor
        if self.mask is None:
            self._data = np.reshape(state, self.shape)
        else:
            self._data[self.mask.data] = state

    def get_state(self):
        """
        Retrieve the medium state.

        Returns
        -------
        preconditioned_state: np.array(dtype=np.float64)
            The preconditioned state scaled by a preconditioning scale factor.

        Notes
        -----
        If the estimator has a mask, data point outside of the mask are not retrieved.
        """
        if self.mask is None:
            data = self.data.ravel()
        else:
            data = self.data[self.mask.data]
        preconditioned_state = data * self.precondition_scale_factor
        return preconditioned_state

    def init_num_parameters(self):
        """
        Initialize the number of parameters to be estimated.

        Returns
        -------
        num_parameters: int
            The number of parameters to be estimated.

        Notes
        -----
        If the estimator has a mask, data point outside of the mask are not counted.
        """
        if self.mask is None:
            num_parameters = self.data.size
        else:
            num_parameters = np.count_nonzero(self.mask.data)
        return num_parameters

    def set_mask(self, mask):
        """
        Set a mask for GridDataEstimator data points and zero points outside of the masked region.

        Parameters
        ----------
        mask: shdom.GridData object
            A GridData object with boolean data. True is for data points that will be estimated.
        """
        self._mask = mask.resample(self.grid, method='nearest')
        self._num_parameters = self.init_num_parameters()
        super().apply_mask(self.mask)

    def get_bounds(self):
        """
        Retrieve the bounds for every parameter (used by scipy.minimize)

        Returns
        -------
        bounds: list of tuples
            The lower and upper bound of each parameter
        """
        min_bound = self.min_bound * self.precondition_scale_factor if self.min_bound is not None else None
        max_bound = self.max_bound * self.precondition_scale_factor if self.max_bound is not None else None
        return [(min_bound, max_bound)] * self.num_parameters

    def project_gradient(self, gradient, grid):
        """
        Project gradient onto the state representation.

        Parameters
        ----------
        grid: shdom.Grid
            The internal shdom grid upon which the gradient was computed.
        gradient: np.array(dtype=np.float64)
            An array containing the gradient of the cost function with respect to the parameters.

        Returns
        -------
        state_gradient: np.array(dtype=np.float64)
            State gradient representation
        """
        gradient = gradient.squeeze(axis=-1)
        state_gradient = shdom.GridData(grid, gradient).resample(self.grid)
        if self.mask is None:
            return state_gradient.data.ravel()
        else:
            return state_gradient.data[self.mask.data] / self.precondition_scale_factor

    @property
    def mask(self):
        return self._mask

    @property
    def precondition_scale_factor(self):
        return self._precondition_scale_factor

    @property
    def num_parameters(self):
        return self._num_parameters

    @property
    def min_bound(self):
        return self._min_bound

    @property
    def max_bound(self):
        return self._max_bound


class ScattererEstimator(object):
    """
    A ScattererEstimator defines an unknown shdom.Scatterer to be estimated.
    A scatterer estimator contains more basic estimators such as GridDataEstimators which define the parameters of the Scatterer that are to be estimated.
    This is an abstract method that is inherited by a specific type of scatterer estimator (e.g. OpticalScatterEstimator, MicrophysicalScattererEstimator)
    """
    def __init__(self):
        self._mask = None
        self._estimators = self.init_estimators()
        self._derivatives = self.init_derivatives()
        self._num_parameters = self.init_num_parameters()
        self._num_estimators = len(self.estimators)

    def init_estimators(self):
        """
        Initialize the internal estimators.

        Returns
        -------
        estimators: OrderedDict
            A dictionary of more basic estimators that define the ScattererEstimator.

        Notes
        -----
        This is a dummy method that is overwritten by inheritance.
        """
        return OrderedDict()

    def init_derivatives(self):
        """
        Initialize the internal derivatives.
        The internal derivatives are of the optical fields (extinction, albedo, phase) with respect to the internal estimators.

        Returns
        -------
        derivatives: OrderedDict
            A dictionary of derivatives that define the ScattererEstimator.

        Notes
        -----
        This is a dummy method that is overwritten by inheritance.
        """
        return OrderedDict()

    def init_num_parameters(self):
        """
        Initialize the number of parameters to be estimated by accumulating all the internal estimator parameters.

        Returns
        -------
        num_parameters: int
            The number of parameters to be estimated.
        """
        num_parameters = []
        for estimator in self.estimators.values():
            num_parameters.append(estimator.num_parameters)
        return num_parameters

    def set_mask(self, mask):
        """
        Set a mask for data points that will be estimated.

        Parameters
        ----------
        mask: shdom.GridData object
            A GridData object with boolean data. True is for data points that will be estimated.
        """
        self._mask = mask.resample(self.grid, method='nearest')
        for estimator in self.estimators.values():
            estimator.set_mask(mask)
        self._num_parameters = self.init_num_parameters()

    def set_state(self, state):
        """
        Set the estimator state by setting all the internal estimators states.

        Parameters
        ----------
        state: np.array(dtype=np.float64)
            The combined state of all the internal estimators
        """
        states = np.split(state, np.cumsum(self.num_parameters[:-1]))
        for estimator, state in zip(self.estimators.values(), states):
            estimator.set_state(state)

    def get_state(self):
        """
        Retrieve the estimator state by joining all the internal estimators states.

        Returns
        -------
        state: np.array(dtype=np.float64)
            The combined state of all the internal estimators
        """
        state = np.empty(shape=(0), dtype=np.float64)
        for estimator in self.estimators.values():
            state = np.concatenate((state, estimator.get_state()))
        return state

    def get_bounds(self):
        """
        Retrieve the bounds for every parameter by accumulating from all internal estimators (used by scipy.minimize)

        Returns
        -------
        bounds: list of tuples
            The lower and upper bound of each parameter
        """
        bounds = []
        for estimator in self.estimators.values():
            bounds.extend(estimator.get_bounds())
        return bounds

    def project_gradient(self, gradient, grid):
        """
        Project gradient onto the combined state representation.

        Parameters
        ----------
        grid: shdom.Grid
            The internal shdom grid upon which the gradient was computed.
        gradient: np.array(dtype=np.float64)
            An array containing the gradient of the cost function with respect to the parameters.
        """
        gradient = np.split(gradient, self.num_estimators, axis=-1)
        state_gradient = np.empty(shape=(0), dtype=np.float64)
        for estimator, gradient in zip(self.estimators.values(), gradient):
            state_gradient = np.concatenate((state_gradient, estimator.project_gradient(gradient, grid)))
        return state_gradient

    @property
    def estimators(self):
        return self._estimators

    @property
    def num_estimators(self):
        return self._num_estimators

    @property
    def derivatives(self):
        return self._derivatives

    @property
    def num_parameters(self):
        return self._num_parameters

    @property
    def mask(self):
        return self._mask


class OpticalScattererEstimator(shdom.OpticalScatterer, ScattererEstimator):
    """
    An OpticalScattererEstimator defines an unknown shdom.OpticalScatterer to be estimated.
    The internal estimators which define an OpticalScatterer are: extinction, albedo, phase.

    Parameters
    ----------
    wavelength: float
        A wavelength in microns
    extinction: shdom.GridData or shdom.GridDataEstimator
        A GridData or GridDataEstimator object containing the extinction (1/km) on a grid
    albedo: shdom.GridData or shdom.GridDataEstimator
        A GridData or GridDataEstimator object containing the single scattering albedo [0,1] on a grid
    phase: shdom.GridPhase or shdom.GridPhaseEstimator
        A GridPhase or GridPhaseEstimator object containing the phase function on a grid

    Notes
    -----
    albedo and phase estimation is not implemented
    """
    def __init__(self, wavelength, extinction, albedo, phase):
        shdom.OpticalScatterer.__init__(self, wavelength, extinction, albedo, phase)
        ScattererEstimator.__init__(self)

    def init_estimators(self):
        """
        Initialize the internal estimators: extinction, albedo, phase

        Returns
        -------
        estimators: OrderedDict
            A dictionary with optional GridDataEstimators and/or GridPhaseEstimators

        Notes
        -----
        albedo and phase estimation is not implemented
        """
        estimators = OrderedDict()
        if isinstance(self.extinction, shdom.GridDataEstimator):
            estimators['extinction'] = self.extinction
        if isinstance(self.albedo, shdom.GridDataEstimator):
            raise NotImplementedError("Albedo estimation not implemented")
        if isinstance(self.phase, shdom.GridPhaseEstimator):
            raise NotImplementedError("Phase estimation not implemented")
        return estimators

    def init_derivatives(self):
        """
        Initialize the internal derivatives.
        The internal derivatives are of the optical fields (extinction, albedo, phase) with respect to the internal estimators.
        For this estimator, the internal parameters are the optical fields themselves, hence the derivatives are indicator functions.

        Returns
        -------
        derivatives: OrderedDict of shdom.OpticalScattererDerivative
            A dictionary of derivatives that define the ScattererEstimator.
        """
        derivatives = OrderedDict()
        if isinstance(self.extinction, shdom.GridDataEstimator):
            derivatives['extinction'] = self.init_extinction_derivative()
        if isinstance(self.albedo, shdom.GridDataEstimator):
            derivatives['albedo'] = self.init_albedo_derivative()
        if isinstance(self.phase, shdom.GridPhaseEstimator):
            derivatives['phase'] = self.init_phase_derivative()
        return derivatives

    def init_extinction_derivative(self):
        """
        Initialize the derivatives of the optical fields with respect to the optical extinction.
        This is an indicator function, as only the extinction depends on the extinction parameter.

        Returns
        -------
        derivative: shdom.OpticalScattererDerivative
            An OpticalScattererDerivative object with the optical derivatives with respect to extinction.
        """
        extinction = shdom.GridData(self.extinction.grid, np.ones_like(self.extinction.data))
        albedo = shdom.GridData(self.albedo.grid, np.zeros_like(self.albedo.data))
        if self.phase.legendre_table.table_type == 'SCALAR':
            legcoef = np.zeros((self.phase.legendre_table.maxleg + 1), dtype=np.float32)
        elif self.phase.legendre_table.table_type == 'VECTOR':
            legcoef = np.zeros((self.phase.legendre_table.nstleg, self.phase.legendre_table.maxleg + 1), dtype=np.float32)
        legen_table = shdom.LegendreTable(legcoef, table_type=self.phase.legendre_table.table_type)
        phase = shdom.GridPhase(legen_table, shdom.GridData(self.phase.index.grid, np.ones_like(self.phase.index.data)))
        derivative = shdom.OpticalScattererDerivative(self.wavelength, extinction, albedo, phase)
        return derivative

    def init_albedo_derivative(self):
        """
        Initialize the derivatives of the optical fields with respect to the single scattering albedo.
        This is an indicator function, as only the albedo depends on the albedo parameter.

        Returns
        -------
        derivative: shdom.OpticalScattererDerivative
            An OpticalScattererDerivative object with the optical derivatives with respect to albedo.
        """
        extinction = shdom.GridData(self.extinction.grid, np.zeros_like(self.extinction.data))
        albedo = shdom.GridData(self.albedo.grid, np.ones_like(self.albedo.data))
        if self.phase.legendre_table.table_type == 'SCALAR':
            legcoef = np.zeros((self.phase.legendre_table.maxleg + 1), dtype=np.float32)
        elif self.phase.legendre_table.table_type == 'VECTOR':
            legcoef = np.zeros((self.phase.legendre_table.nstleg, self.phase.legendre_table.maxleg + 1), dtype=np.float32)
        legen_table = shdom.LegendreTable(legcoef, table_type=self.phase.legendre_table.table_type)
        phase = shdom.GridPhase(legen_table, shdom.GridData(self.phase.index.grid, np.ones_like(self.phase.index.data)))
        derivative = shdom.OpticalScattererDerivative(self.wavelength, extinction, albedo, phase)
        return derivative

    def init_phase_derivative(self):
        """
        Initialize the derivatives of the optical fields with respect to the phase function.

        Notes
        -----
        This is a dummy method which is not implemented.
        """
        raise NotImplementedError("Phase estimation not implemented")

    def get_derivative(self, derivative_type, wavelength):
        """
        Retrieve the relevant derivative at a single wavelength.

        Parameters
        ----------
        derivative_type: str
            'extinction' for the extinction derivative
            'albedo' for the albedo derivative
            'phase' for the phase derivative
        wavelength: float
            A wavelength in microns

        Returns
        -------
        derivative: shdom.OpticalScattererDerivative
            An OpticalScattererDerivative with respect to the type requested

        Notes
        -----
        Wavelength here is a dummy. The optical derivatives with respect to the optical parameters are indicator functions and not a function of wavelength
        """
        if derivative_type == 'extinction':
            derivative = self.derivatives['extinction']
        elif derivative_type == 'albedo':
            derivative = self.derivatives['albedo']
        elif derivative_type == 'phase':
            derivative = self.derivatives['phase']
        else:
            raise AttributeError('derivative type {} not supported'.format(derivative_type))
        return derivative


class MicrophysicalScattererEstimator(shdom.MicrophysicalScatterer, ScattererEstimator):
    """
    An MicrophysicalScattererEstimator defines an unknown shdom.MicrophysicalScatterer to be estimated.
    The internal estimators which define an MicrophysicalScatterer are: lwc, reff, veff.

    Parameters
    ----------
    mie: shdom.MiePolydisperse or list of shdom.MiePolydisperse
        Using the Mie model microphyical properties are transformed into optical properties (see get_optical_scatterer method)
    lwc: shdom.GridData or shdom.GridDataEstimator
        A GridData object containing liquid water content (g/m^3) on a 3D grid.
    reff: shdom.GridData or shdom.GridDataEstimator
        A GridData object containing effective radii (micron) on a 3D grid.
    veff: shdom.GridDatao r shdom.GridDataEstimator
        A GridData object containing effective variances on a 3D grid.
    """
    def __init__(self, mie, lwc, reff, veff):
        shdom.MicrophysicalScatterer.__init__(self, lwc, reff, veff)
        self.add_mie(mie)
        ScattererEstimator.__init__(self)

    def init_estimators(self):
        """
        Initialize the internal estimators: lwc, reff, veff

        Returns
        -------
        estimators: OrderedDict
            A dictionary with optional GridDataEstimators
        """
        estimators = OrderedDict()
        if isinstance(self.lwc, shdom.GridDataEstimator):
            estimators['lwc'] = self.lwc
        if isinstance(self.reff, shdom.GridDataEstimator):
            estimators['reff'] = self.reff
        if isinstance(self.veff, shdom.GridDataEstimator):
            estimators['veff'] = self.veff
        return estimators

    def init_derivatives(self):
        """
        Initialize the internal derivatives.
        The internal derivatives are of the optical fields (extinction, albedo, phase) with respect to the internal estimators.
        For this estimator, the internal parameters are: lwc, reff, veff

        Returns
        -------
        derivatives: OrderedDict of shdom.OpticalScattererDerivative
            A dictionary of derivatives that define the ScattererEstimator.
        """
        derivatives = OrderedDict()
        if isinstance(self.lwc, shdom.GridDataEstimator):
            derivatives['lwc'] = self.init_lwc_derivative()
        if isinstance(self.reff, shdom.GridDataEstimator):
            derivatives['reff'] = self.init_mie_derivative(derivative_type='reff')
        if isinstance(self.veff, shdom.GridDataEstimator):
            derivatives['veff'] = self.init_mie_derivative(derivative_type='veff')
        return derivatives

    def init_lwc_derivative(self):
        """
        Initialize the derivatives of the optical fields with respect to the liquid water content.

        Returns
        -------
        derivative: OrderedDict
            A dictionary of properties that define the the optical derivatives with respect to lwc.

        Notes
        -----
        Only the optical extinction depends on the lwc, however, since it also depends on wavelength it is not initialized here.
        """
        derivative = OrderedDict()
        derivative['lwc'] = shdom.GridData(self.lwc.grid, np.ones_like(self.lwc.data))
        derivative['albedo'] = shdom.GridData(self.grid, np.zeros(self.grid.shape))
        return derivative

    def init_mie_derivative(self, derivative_type):
        """
        Initialize the Mie derivatives of the optical fields with respect to the effective radius/variance.
        This means the optical cross-section, single scattering albedo and phase function derivatives.

        Returns
        -------
        derivatives: OrderedDict
            A dictionary of shdom.MiePolydisperse (at every wavelength).
        derivative_type: str
            The derivative type: 'reff' or 'veff'

        Notes
        -----
        Derivatives are computed numerically thus small enough spacing of reff/veff in the Mie tables is required.
        """
        derivatives = OrderedDict()
        for wavelength, mie in self.mie.items():
            extinct = mie.extinct.reshape((mie.size_distribution.nretab, mie.size_distribution.nvetab), order='F')
            ssalb = mie.ssalb.reshape((mie.size_distribution.nretab, mie.size_distribution.nvetab), order='F')

            if derivative_type == 'reff':
                dextinct = np.gradient(extinct, mie.size_distribution.reff, axis=-2)
                dssalb = np.gradient(ssalb, mie.size_distribution.reff, axis=-2)
                dlegcoef = np.gradient(mie.legcoef_2d, mie.size_distribution.reff, axis=-2)

            elif derivative_type == 'veff':
                dextinct = np.gradient(extinct, mie.size_distribution.veff, axis=-1)
                dssalb = np.gradient(ssalb, mie.size_distribution.veff, axis=-1)
                dlegcoef = np.gradient(mie.legcoef_2d, mie.size_distribution.veff, axis=-1)

            # Define a derivative Mie object, last derivative is duplicated
            derivative = copy.deepcopy(mie)
            derivative._extinct = dextinct.ravel(order='F')
            derivative._ssalb = dssalb.ravel(order='F')
            if mie.table_type == 'SCALAR':
                derivative.legcoef = dlegcoef.reshape((mie.maxleg+1, -1), order='F')
            elif mie.table_type == 'VECTOR':
                derivative.legcoef = dlegcoef.reshape((mie.legendre_table.nstleg, mie.maxleg+1, -1), order='F')

            derivative.init_intepolators()
            derivatives[float_round(wavelength)] = derivative
        return derivatives

    def get_lwc_derivative(self, wavelength):
        """
        Retrieve the liquid water content derivative at a single wavelength.

        Parameters
        ----------
        wavelength: float
            Wavelength in microns. A Mie table at this wavelength should be added prior

        Returns
        -------
        scatterer: shdom.OpticalScatterer
            The derivative with respect to lwc at a single wavelength
        """
        mie = self.mie[float_round(wavelength)]
        index = shdom.GridData(self.grid, np.ones(self.grid.shape, dtype=np.int32))
        if mie.table_type == 'SCALAR':
            legen_table = shdom.LegendreTable(np.zeros((mie.maxleg+1), dtype=np.float32), mie.table_type)
        elif mie.table_type == 'VECTOR':
            legen_table = shdom.LegendreTable(np.zeros((mie.legendre_table.nstleg, mie.maxleg + 1), dtype=np.float32), mie.table_type)
        derivative = self.derivatives['lwc']
        scatterer = shdom.OpticalScattererDerivative(
            wavelength,
            extinction=mie.get_extinction(derivative['lwc'], self.reff, self.veff),
            albedo=derivative['albedo'],
            phase=shdom.GridPhase(legen_table, index))
        return scatterer

    def get_mie_derivative(self, derivative, wavelength):
        """
        Retrieve mie scattering derivatives at a single wavelength.

        Parameters
        ----------
        derivative: OrderedDict
            A dictionary of shdom.MiePolydisperse at multiple wavelengths
        wavelength: float
            Wavelength in microns. A Mie derivative at this wavelength should be added prior

        Returns
        -------
        scatterer: shdom.OpticalScatterer
            The Mie derivative at a single wavelength
        """
        scatterer = shdom.OpticalScattererDerivative(
            wavelength,
            extinction=derivative[float_round(wavelength)].get_extinction(self.lwc, self.reff, self.veff),
            albedo=derivative[float_round(wavelength)].get_albedo(self.reff, self.veff),
            phase=derivative[float_round(wavelength)].get_phase(self.reff, self.veff))
        return scatterer

    def get_derivative(self, derivative_type, wavelength):
        """
        Retrieve the relevant derivative at a single wavelength.

        Parameters
        ----------
        derivative_type: str
            'lwc' for the lwc derivative
            'reff' for the reff derivative
            'veff' for the veff derivative
        wavelength: float
            A wavelength in microns

        Returns
        -------
        derivative: shdom.OpticalScattererDerivative
            An OpticalScattererDerivative with respect to the type requested
        """
        if derivative_type == 'lwc':
            derivative = self.get_lwc_derivative(wavelength)
        elif derivative_type == 'reff' or derivative_type == 'veff':
            derivative = self.get_mie_derivative(self.derivatives[derivative_type], wavelength)
        else:
            raise AttributeError('derivative type {} not supported'.format(derivative_type))
        return derivative


class MediumEstimator(shdom.Medium):
    """
    A MediumEstimator defines an unknown shdom.Medium to be estimated.
    A medium estimator is a shdom.Medium with unknown (and optionally known) scatterers.
    Unknown scatterers are defined by internal estimators (shdom.ScattererEstimator instances).

    Parameters
    ----------
    grid: shdom.Grid, optional
        A grid for the Medium object. All scatterers will be resampled to this grid.
                loss_type: str,
    loss_type: 'l2' or 'normcorr'.
        l2 - used for l2 norm between the acquired and synthetic (rendered) measurements
        normcorr - used for the normalized correlation between the acquired and synthetic (rendered) measurements
    exact_single_scatter: bool
        True will compute the exact single scattering gradient along a broken-ray trajectory (using the direct solar beam)
    stokes_weights: list of floats
        Loss function weights for stokes vector components [I,Q,U,V].
    """
    def __init__(self, grid=None, loss_type='l2', exact_single_scatter=True, stokes_weights=None):
        super().__init__(grid)
        self._estimators = OrderedDict()
        self._num_parameters = []
        self._unknown_scatterers_indices = np.empty(shape=(0), dtype=np.int32)
        self._num_derivatives = 0
        self._num_estimators = 0
        self._exact_single_scatter = exact_single_scatter
        self._core_grad, self._output_transform = self.init_loss_function(loss_type)
        self._stokes_weights = stokes_weights if stokes_weights is not None else np.array([1.0,0.0,0.0,0.0], dtype=np.float32)

    def init_loss_function(self, loss_type):
        """
        Initialized the loss function and corresponding gradient computation.
        This includes how a transformation of the core Fortran output parameters.

        Parameters
        ----------
        loss_type: 'l2' or 'normcorr'.
            l2 - used for l2 norm between the acquired and synthetic (rendered) measurements
            normcorr - used for the normalized correlation between the acquired and synthetic (rendered) measurements

        Returns
        -------
        core_grad: function
             The core gradient and loss computation routine
        output_transform: function
             The transformation to the output parameters.
        """
        if loss_type == 'l2':
            core_grad = self.grad_l2

            def output_transform(output, projection, sensor, num_wavelengths, masks=None):
                loss = np.sum(list(map(lambda x: x[1], output)))
                gradient = np.sum(list(map(lambda x: x[0], output)), axis=0)
                if masks is not None:
                    images = sensor.make_images(np.concatenate(list(map(lambda x: x[2], output)), axis=-1),
                                                projection,
                                                num_wavelengths, masks)
                else:
                    images = sensor.make_images(np.concatenate(list(map(lambda x: x[2], output)), axis=-1),
                                                projection,
                                                num_wavelengths)
                return loss, gradient, images

        elif loss_type == 'l2_weighted':
            core_grad = self.grad_l2_weighted

            def output_transform(output, projection, sensor, num_wavelengths):
                loss = np.sum(list(map(lambda x: x[1], output)))
                gradient = np.sum(list(map(lambda x: x[0], output)), axis=0)
                images = sensor.make_images(np.concatenate(list(map(lambda x: x[2], output)), axis=-1),
                                            projection,
                                            num_wavelengths)
                return loss, gradient, images

        elif loss_type == 'normcorr':
            core_grad = self.grad_normcorr

            def output_transform(output, projection, sensor, num_wavelengths):
                norm1 = np.sum(list(map(lambda x: x[2], output)), axis=0)[:, None, None]
                norm2 = np.sum(list(map(lambda x: x[3], output)), axis=0)[:, None, None]
                norm = np.sqrt(norm1 * norm2)
                loss = np.sum(list(map(lambda x: x[4], output)), axis=0)[:, None, None]
                grad_fn = lambda x: ((loss * x[0]) / norm1 - x[1]) / norm
                gradient = np.mean(np.sum(list(map(grad_fn, output)), axis=0), axis=0)
                loss = -np.mean(loss / norm, dtype=np.float64)
                images = sensor.make_images(np.concatenate(list(map(lambda x: x[5], output)), axis=-1),
                                            projection,
                                            num_wavelengths)
                return loss, gradient, images
        else:
            raise NotImplementedError('Loss type {} not implemented'.format(loss_type))

        return core_grad, output_transform

    def add_scatterer(self, scatterer, name=None):
        """
        Add a Scatterer to the medium. If the scatterer is a ScattererEstimator it will enter the estimator list.

        Parameters
        ----------
        scatterer: shdom.Scatterer or shdom.ScattererEstimator
            A known or unknown scattering particle distribution
            (e.g. MicrophysicalScatterer, MicrophysicalScattererEstimator, OpticalScatterer, OpticalScattererEstimator)
        name: string, optional
            A name for the scatterer that will be used to retrieve it (see get_scatterer method).
            If no name is specified the default name is scatterer# where # is the number in which it was input (i.e. scatterer1 for the first scatterer).
        """
        super().add_scatterer(scatterer, name)
        if issubclass(type(scatterer), shdom.ScattererEstimator):
            name = 'scatterer{:d}'.format(self._num_scatterers) if name is None else name
            self._num_estimators += 1
            total_num_estimators = len(scatterer.estimators)
            self._estimators[name] = scatterer
            self._num_parameters.append(np.sum(scatterer.num_parameters))
            self._unknown_scatterers_indices = np.concatenate((
                self.unknown_scatterers_indices,
                np.full(total_num_estimators, self.num_scatterers, dtype=np.int32)))
            self._num_derivatives += total_num_estimators

    def set_state(self, state):
        """
        Set the estimator state by setting all the internal estimators states.

        Parameters
        ----------
        state: np.array(dtype=np.float64)
            The combined state of all the internal estimators
        """
        states = np.split(state, np.cumsum(self.num_parameters[:-1]))
        for (name, estimator), state in zip(self.estimators.items(), states):
            estimator.set_state(state)
            self.scatterers[name] = estimator

    def get_state(self):
        """
        Retrieve the estimator state by joining all the internal estimators states.

        Returns
        -------
        state: np.array(dtype=np.float64)
            The combined state of all the internal estimators
        """
        state = np.empty(shape=(0),dtype=np.float64)
        for estimator in self.estimators.values():
            state = np.concatenate((state, estimator.get_state()))
        return state

    def get_bounds(self):
        """
        Retrieve the bounds for every parameter by accumulating from all internal estimators (used by scipy.minimize)

        Returns
        -------
        bounds: list of tuples
            The lower and upper bound of each parameter
        """
        bounds = []
        for estimator in self.estimators.values():
            bounds.extend(estimator.get_bounds())
        return bounds

    def get_derivatives(self, rte_solver):
        """
        Retrieve the relevant derivatives for a given RteSolver.

        Parameters
        ----------
        rte_solver: shdom.RteSolver
            The RteSolver object (at a given wavelength)

        Returns
        -------
        dext: np.array(dtype=np.float32)
            The derivative of the optical extinction with respect to the parameters
        dalb: np.array(dtype=np.float32)
            The derivative of the single scattering albedo with respect to the parameters
        diphase: np.array(dtype=np.int32)
            A pointer to the derivative of the phase function respect to the parameters
        dleg: np.array(dtype=np.float32)
             The derivative of the phase function legendre coefficients with respect to the parameters
        dphasetab: np.array(dtype=np.float32)
             The derivative of the phase function at pre-determined angles with respect to the parameters
        dnumphase: np.array(dtype=np.float32)
             The number of phase function derivatives
        """
        dext = np.zeros(shape=[rte_solver._nbpts, self.num_derivatives], dtype=np.float32)
        dalb = np.zeros(shape=[rte_solver._nbpts, self.num_derivatives], dtype=np.float32)
        diphase = np.zeros(shape=[rte_solver._nbpts, self.num_derivatives], dtype=np.int32)

        i=0
        for estimator in self.estimators.values():
            for dtype in estimator.derivatives.keys():
                derivative = estimator.get_derivative(dtype, rte_solver.wavelength)
                resampled_derivative = derivative.resample(self.grid)
                dext[:, i] = resampled_derivative.extinction.data.ravel()
                dalb[:, i] = resampled_derivative.albedo.data.ravel()
                diphase[:, i] = resampled_derivative.phase.iphasep.ravel() + diphase.max()

                if i == 0:
                    leg_table = copy.deepcopy(resampled_derivative.phase.legendre_table)
                else:
                    leg_table.append(copy.deepcopy(resampled_derivative.phase.legendre_table))
                i += 1

        leg_table.pad(rte_solver._nleg)
        dleg = leg_table.data
        dnumphase = leg_table.numphase

        # zero the first term of the first component of the phase function
        # gradient. Pre-scale the legendre moments by 1/(2*l+1) which
        # is done in the forward problem in TRILIN_INTERP_PROP
        scaling_factor =np.array([2.0*i+1.0 for i in range(0,rte_solver._nleg+1)])
        if dleg.ndim == 2:
            dleg[0,:] = 0.0
            dleg /= scaling_factor[:,np.newaxis]
        elif dleg.ndim ==3:
            dleg[0,0,:] = 0.0
            dleg /= scaling_factor[np.newaxis,:,np.newaxis]
            dleg = dleg[:rte_solver._nstleg]

        dphasetab = core.precompute_phase_check_grad(
                                                     negcheck=False,
                                                     nstphase=rte_solver._nstphase,
                                                     nstleg=rte_solver._nstleg,
                                                     nscatangle=rte_solver._nscatangle,
                                                     nstokes=rte_solver._nstokes,
                                                     dnumphase=dnumphase,
                                                     ml=rte_solver._ml,
                                                     nlm=rte_solver._nlm,
                                                     nleg=rte_solver._nleg,
                                                     dleg=dleg,
                                                     deltam=rte_solver._deltam
                                                     )

        return dext, dalb, diphase, dleg, dphasetab, dnumphase

    def compute_direct_derivative(self, rte_solver):
        """
        Compute the derivative with respect to the direct solar beam. This is a ray for every point in 3D space.
        Internally this method stores for every point the indices and paths traversed by the direct solar beam to reach it.

        Parameters
        ----------
        rte_solver: shdom.RteSolver
            The RteSolver object (at a given wavelength)
        """

        # There is no optical information stored here, only paths and indices
        # Therefor, there is no important for the specific RteSolver which is used
        if isinstance(rte_solver, shdom.RteSolverArray):
            uniformzlev = max([solver._uniformzlev for solver in rte_solver])
            rte_solver = rte_solver[0]
        else:
            uniformzlev = rte_solver._uniformzlev

        self._direct_derivative_path, self._direct_derivative_ptr = \
            core.make_direct_derivative(
                npts=rte_solver._npts,
                bcflag=rte_solver._bcflag,
                gridpos=rte_solver._gridpos,
                npx=rte_solver._pa.npx,
                npy=rte_solver._pa.npy,
                npz=rte_solver._pa.npz,
                delx=rte_solver._pa.delx,
                dely=rte_solver._pa.dely,
                xstart=rte_solver._pa.xstart,
                ystart=rte_solver._pa.ystart,
                zlevels=rte_solver._pa.zlevels,
                ipdirect=rte_solver._ipdirect,
                di=rte_solver._di,
                dj=rte_solver._dj,
                dk=rte_solver._dk,
                epss=rte_solver._epss,
                epsz=rte_solver._epsz,
                xdomain=rte_solver._xdomain,
                ydomain=rte_solver._ydomain,
                cx=rte_solver._cx,
                cy=rte_solver._cy,
                cz=rte_solver._cz,
                cxinv=rte_solver._cxinv,
                cyinv=rte_solver._cyinv,
                czinv=rte_solver._czinv,
                uniformzlev=uniformzlev,
                delxd=rte_solver._delxd,
                delyd=rte_solver._delyd
            )

    def grad_normcorr(self, rte_solver, projection, pixels, sigma):
        """
        The core normalized correlation gradient method.

        Parameters
        ----------
        rte_solver: shdom.RteSolver
            A solver with all the associated parameters and the solution to the RTE
        projection: shdom.Projection
            A projection model which specified the position and direction of each and every pixel
        pixels: np.array(shape=(projection.npix), dtype=np.float32)
            The acquired pixels driving the error and optimization.
        sigma: np.array(shape=(nstokes, nstokes), dtype=np.float32)
            The noise correlation matrix.

        Returns
        -------
        grad1: np.array(shape=(rte_solver._nstokes, rte_solver._nbpts, self.num_derivatives), dtype=np.float32)
            A part of the gradient with respect to all parameters at every grid base point
        grad2: np.array(shape=(rte_solver._nstokes, rte_solver._nbpts, self.num_derivatives), dtype=np.float32)
            A part of the gradient with respect to all parameters at every grid base point
        norm1: np.array(shape=(rte_solver._nstokes), dtype=np.float32)
            The per stokes component norm of all synthetic pixels
        norm2: np.array(shape=(rte_solver._nstokes), dtype=np.float32)
            The per stokes component norm of all measurements
        loss: float32
            The per stokes component correlations of the measruements and synthetic pixels
        images: np.array(shape=(rte_solver._nstokes, projection.npix), dtype=np.float32)
            The rendered (synthetic) images.
        """
        if isinstance(projection.npix, list):
            total_pix = np.sum(projection.npix)
        else:
            total_pix = projection.npix

        grad1, grad2, norm1, norm2, loss, images = core.gradient_normcorr(
            weights=self._stokes_weights[:rte_solver._nstokes],
            exact_single_scatter=self._exact_single_scatter,
            nstphase=rte_solver._nstphase,
            dpath=self._direct_derivative_path,
            dptr=self._direct_derivative_ptr,
            npx=rte_solver._pa.npx,
            npy=rte_solver._pa.npy,
            npz=rte_solver._pa.npz,
            delx=rte_solver._pa.delx,
            dely=rte_solver._pa.dely,
            xstart=rte_solver._pa.xstart,
            ystart=rte_solver._pa.ystart,
            zlevels=rte_solver._pa.zlevels,
            extdirp=rte_solver._pa.extdirp,
            uniformzlev=rte_solver._uniformzlev,
            partder=self.unknown_scatterers_indices,
            numder=self.num_derivatives,
            dext=rte_solver._dext,
            dalb=rte_solver._dalb,
            diphase=rte_solver._diphase,
            dleg=rte_solver._dleg,
            dphasetab=rte_solver._dphasetab,
            dnumphase=rte_solver._dnumphase,
            nscatangle=rte_solver._nscatangle,
            phasetab=rte_solver._phasetab,
            ylmsun=rte_solver._ylmsun,
            nstokes=rte_solver._nstokes,
            nstleg=rte_solver._nstleg,
            nx=rte_solver._nx,
            ny=rte_solver._ny,
            nz=rte_solver._nz,
            bcflag=rte_solver._bcflag,
            ipflag=rte_solver._ipflag,
            npts=rte_solver._npts,
            nbpts=rte_solver._nbpts,
            ncells=rte_solver._ncells,
            nbcells=rte_solver._nbcells,
            ml=rte_solver._ml,
            mm=rte_solver._mm,
            ncs=rte_solver._ncs,
            nlm=rte_solver._nlm,
            numphase=rte_solver._pa.numphase,
            nmu=rte_solver._nmu,
            nphi0max=rte_solver._nphi0max,
            nphi0=rte_solver._nphi0,
            maxnbc=rte_solver._maxnbc,
            ntoppts=rte_solver._ntoppts,
            nbotpts=rte_solver._nbotpts,
            nsfcpar=rte_solver._nsfcpar,
            gridptr=rte_solver._gridptr,
            neighptr=rte_solver._neighptr,
            treeptr=rte_solver._treeptr,
            shptr=rte_solver._shptr,
            bcptr=rte_solver._bcptr,
            cellflags=rte_solver._cellflags,
            iphase=rte_solver._iphase[:rte_solver._npts],
            deltam=rte_solver._deltam,
            solarflux=rte_solver._solarflux,
            solarmu=rte_solver._solarmu,
            solaraz=rte_solver._solaraz,
            gndtemp=rte_solver._gndtemp,
            gndalbedo=rte_solver._gndalbedo,
            skyrad=rte_solver._skyrad,
            waveno=rte_solver._waveno,
            wavelen=rte_solver._wavelen,
            mu=rte_solver._mu,
            phi=rte_solver._phi,
            wtdo=rte_solver._wtdo,
            xgrid=rte_solver._xgrid,
            ygrid=rte_solver._ygrid,
            zgrid=rte_solver._zgrid,
            gridpos=rte_solver._gridpos,
            sfcgridparms=rte_solver._sfcgridparms,
            bcrad=rte_solver._bcrad,
            extinct=rte_solver._extinct[:rte_solver._npts],
            albedo=rte_solver._albedo[:rte_solver._npts],
            legen=rte_solver._legen,
            dirflux=rte_solver._dirflux[:rte_solver._npts],
            fluxes=rte_solver._fluxes,
            source=rte_solver._source,
            camx=projection.x,
            camy=projection.y,
            camz=projection.z,
            cammu=projection.mu,
            camphi=projection.phi,
            npix=total_pix,
            srctype=rte_solver._srctype,
            sfctype=rte_solver._sfctype,
            units=rte_solver._units,
            measurements=pixels,
            rshptr=rte_solver._rshptr,
            radiance=rte_solver._radiance,
            total_ext=rte_solver._total_ext[:rte_solver._npts]
        )
        return grad1, grad2, norm1, norm2, loss, images

    def grad_l2(self, rte_solver, projection, pixels, uncertainties,
                jacobian_flag=False):
        """
        The core l2 gradient method.

        Parameters
        ----------
        rte_solver: shdom.RteSolver
            A solver with all the associated parameters and the solution to the RTE
        projection: shdom.Projection
            A projection model which specified the position and direction of each and every pixel
        pixels: np.array(shape=(projection.npix), dtype=np.float32)
            The acquired pixels driving the error and optimization.
        uncertainties: np.array(shape=(projection.npix), dtype=np.float32)
            The pixel uncertainties.

        Returns
        -------
        gradient: np.array(shape=(rte_solver._nbpts, self.num_derivatives), dtype=np.float64)
            The gradient with respect to all parameters at every grid base point
        loss: float64
            The total loss accumulated over all pixels
        images: np.array(shape=(rte_solver._nstokes, projection.npix), dtype=np.float32)
            The rendered (synthetic) images.
        """
        if isinstance(projection.npix, list):
            total_pix = np.sum(projection.npix)
        else:
            total_pix = projection.npix

        if jacobian_flag:
            #maximum size is hard coded - should be roughly an order of magnitude
            #larger than the maximum necssary size but is possible source of seg faults.
            largest_dim = int(100*np.sqrt(rte_solver._nx**2+rte_solver._ny**2+rte_solver._nz**2))
            jacobian = np.empty((rte_solver._nstokes,self.num_derivatives,total_pix*largest_dim),order='F',dtype=np.float32)
            jacobian_ptr = np.empty((2,total_pix*largest_dim),order='F',dtype=np.int32)
        else:
            jacobian = np.empty((rte_solver._nstokes,self.num_derivatives,1),order='F',dtype=np.float32)
            jacobian_ptr = np.empty((2,1),order='F',dtype=np.int32)

        gradient, loss, images, jacobian, jacobian_ptr, counter = core.gradient_l2(
            uncertainties=uncertainties,
            weights=self._stokes_weights[:rte_solver._nstokes],
            exact_single_scatter=self._exact_single_scatter,
            nstphase=rte_solver._nstphase,
            dpath=self._direct_derivative_path,
            dptr=self._direct_derivative_ptr,
            npx=rte_solver._pa.npx,
            npy=rte_solver._pa.npy,
            npz=rte_solver._pa.npz,
            delx=rte_solver._pa.delx,
            dely=rte_solver._pa.dely,
            xstart=rte_solver._pa.xstart,
            ystart=rte_solver._pa.ystart,
            zlevels=rte_solver._pa.zlevels,
            extdirp=rte_solver._pa.extdirp,
            uniformzlev=rte_solver._uniformzlev,
            partder=self.unknown_scatterers_indices,
            numder=self.num_derivatives,
            dext=rte_solver._dext,
            dalb=rte_solver._dalb,
            diphase=rte_solver._diphase,
            dleg=rte_solver._dleg,
            dphasetab=rte_solver._dphasetab,
            dnumphase=rte_solver._dnumphase,
            nscatangle=rte_solver._nscatangle,
            phasetab=rte_solver._phasetab,
            ylmsun=rte_solver._ylmsun,
            nstokes=rte_solver._nstokes,
            nstleg=rte_solver._nstleg,
            nx=rte_solver._nx,
            ny=rte_solver._ny,
            nz=rte_solver._nz,
            bcflag=rte_solver._bcflag,
            ipflag=rte_solver._ipflag,
            npts=rte_solver._npts,
            nbpts=rte_solver._nbpts,
            ncells=rte_solver._ncells,
            nbcells=rte_solver._nbcells,
            ml=rte_solver._ml,
            mm=rte_solver._mm,
            ncs=rte_solver._ncs,
            nlm=rte_solver._nlm,
            numphase=rte_solver._pa.numphase,
            nmu=rte_solver._nmu,
            nphi0max=rte_solver._nphi0max,
            nphi0=rte_solver._nphi0,
            maxnbc=rte_solver._maxnbc,
            ntoppts=rte_solver._ntoppts,
            nbotpts=rte_solver._nbotpts,
            nsfcpar=rte_solver._nsfcpar,
            gridptr=rte_solver._gridptr,
            neighptr=rte_solver._neighptr,
            treeptr=rte_solver._treeptr,
            shptr=rte_solver._shptr,
            bcptr=rte_solver._bcptr,
            cellflags=rte_solver._cellflags,
            iphase=rte_solver._iphase[:rte_solver._npts],
            deltam=rte_solver._deltam,
            solarflux=rte_solver._solarflux,
            solarmu=rte_solver._solarmu,
            solaraz=rte_solver._solaraz,
            gndtemp=rte_solver._gndtemp,
            gndalbedo=rte_solver._gndalbedo,
            skyrad=rte_solver._skyrad,
            waveno=rte_solver._waveno,
            wavelen=rte_solver._wavelen,
            mu=rte_solver._mu,
            phi=rte_solver._phi,
            wtdo=rte_solver._wtdo,
            xgrid=rte_solver._xgrid,
            ygrid=rte_solver._ygrid,
            zgrid=rte_solver._zgrid,
            gridpos=rte_solver._gridpos,
            sfcgridparms=rte_solver._sfcgridparms,
            bcrad=rte_solver._bcrad,
            extinct=rte_solver._extinct[:rte_solver._npts],
            albedo=rte_solver._albedo[:rte_solver._npts],
            legen=rte_solver._legen,
            dirflux=rte_solver._dirflux[:rte_solver._npts],
            fluxes=rte_solver._fluxes,
            source=rte_solver._source,
            camx=projection.x,
            camy=projection.y,
            camz=projection.z,
            cammu=projection.mu,
            camphi=projection.phi,
            npix=total_pix,
            srctype=rte_solver._srctype,
            sfctype=rte_solver._sfctype,
            units=rte_solver._units,
            measurements=pixels,
            rshptr=rte_solver._rshptr,
            radiance=rte_solver._radiance,
            total_ext=rte_solver._total_ext[:rte_solver._npts],
            jacobian=jacobian,
            jacobianptr=jacobian_ptr,
            makejacobian=jacobian_flag
        )
        jacobian = jacobian[:,:,:counter-1]
        jacobian_ptr = jacobian_ptr[:,:counter-1]
        if jacobian_flag:
            return gradient, loss, images, jacobian, jacobian_ptr, np.array([counter-1])
        else:
            return gradient, loss, images

    def compute_gradient(self, rte_solvers, measurements, n_jobs, measurement_weight=None, pixels_mask=None, jacobian_flag=False):
        """
        Compute the gradient with respect to the current state.
        If n_jobs>1 than parallel gradient computation is used with pixels are distributed amongst all workers

        Parameters
        ----------
        rte_solvers: shdom.RteSolverArray
            A solver array with all the associated parameters and the solution to the RTE
        measurements: shdom.Measurements
            A measurements object storing the acquired images and sensor geometry
        n_jobs: int,
            The number of jobs to divide the gradient computation into.

        Returns
        -------
        state_gradient: np.array(shape=(self.num_parameters), dtype=np.float64)
            The gradient of the loss function with respect to the state parameters
        loss: np.float64
            The total loss accumulated over all pixels
        images: list of np.array(shape=(measurements.projection.resolution), dtype=np.float32)
            A list of the rendered (synthetic) images, used for display purposes.
        """
        # Pre-computation of phase-function and derivatives for all solvers.
        for rte_solver in rte_solvers.solver_list:
            rte_solver.precompute_phase()
            rte_solver._dext, rte_solver._dalb, rte_solver._diphase, \
                rte_solver._dleg, rte_solver._dphasetab, rte_solver._dnumphase = self.get_derivatives(rte_solver)

        projection = measurements.camera.projection
        sensor = measurements.camera.sensor
        pixels = measurements.pixels
        if pixels_mask is not None:
            pixels_mask = pixels[0,:,0]>0
            projection = projection[pixels_mask]
            pixels = pixels[:,pixels_mask,:]
        if measurements.uncertainties is None:
            uncertainties = np.repeat(np.ones(pixels.shape,order='F',dtype=np.float32)[np.newaxis,:],rte_solver._nstokes,axis=0)
        else:
            uncertainties = measurements.uncertainties
        # Sequential or parallel processing using multithreading (threadsafe Fortran)
        if measurement_weight is not None:
            if n_jobs > 1:
                output = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
                    delayed(self.core_grad, check_pickle=False)(
                        rte_solver=rte_solvers[channel],
                        projection=projection,
                        pixels=spectral_pixels[..., channel],
                        uncertainties=spectral_uncertainties[...,channel],
                        pixels_weight=pixels_weight,
                        jacobian_flag=jacobian_flag,
                    ) for channel, (projection, spectral_pixels, pixels_weight, spectral_uncertainties) in
                    itertools.product(range(self.num_wavelengths), zip(projection.split(n_jobs),
                                                                       np.array_split(pixels, n_jobs, axis=-2),
                                                                       np.array_split(measurement_weight, n_jobs),
                                                                       np.array_split(uncertainties, n_jobs, axis=-2)))
                )
            else:
                output = [
                    self.core_grad(rte_solvers[channel], projection, pixels[..., channel], uncertainties[...,channel],jacobian_flag=jacobian_flag)
                    for channel in range(self.num_wavelengths)
                ]
        else:
            if n_jobs > 1:
                output = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
                    delayed(self.core_grad, check_pickle=False)(
                        rte_solver=rte_solvers[channel],
                        projection=projection,
                        pixels=spectral_pixels[..., channel],
                        uncertainties=spectral_uncertainties[..., channel],
                        jacobian_flag=jacobian_flag,
                    ) for channel, (projection, spectral_pixels, spectral_uncertainties) in
                    itertools.product(range(self.num_wavelengths), zip(projection.split(n_jobs),
                                                                       np.array_split(pixels, n_jobs, axis=-2),
                                                                       np.array_split(uncertainties, n_jobs, axis=-2)))
                )
            else:
                output = [
                    self.core_grad(rte_solvers[channel], projection, pixels[..., channel], uncertainties[..., channel],
                                   jacobian_flag=jacobian_flag)
                    for channel in range(self.num_wavelengths)
                ]

        # Sum over all the losses of the different channels
        loss, gradient, images = self.output_transform(output, measurements.camera.projection, sensor, self.num_wavelengths, pixels_mask)

        if jacobian_flag:
            assert self._core_grad ==self.grad_l2, 'jacobian is currently only generated by the gradient_l2 function'
            jacobian = Jacobians(output=output,projection=projection,rte_solver=rte_solvers, estimators=self.estimators)
        else:
            jacobian=jacobian_flag

        gradient = gradient.reshape(self.grid.shape + tuple([self.num_derivatives]))
        gradient = np.split(gradient, self.num_estimators, axis=-1)
        state_gradient = np.empty(shape=(0), dtype=np.float64)
        for estimator, gradient in zip(self.estimators.values(), gradient):
            state_gradient = np.concatenate((state_gradient, estimator.project_gradient(gradient, self.grid)))

        return state_gradient, loss, images, jacobian

    @property
    def estimators(self):
        return self._estimators

    @property
    def num_parameters(self):
        return self._num_parameters

    @property
    def unknown_scatterers_indices(self):
        return self._unknown_scatterers_indices

    @property
    def num_derivatives(self):
        return self._num_derivatives

    @property
    def num_estimators(self):
        return self._num_estimators

    @property
    def core_grad(self):
        return self._core_grad

    @property
    def output_transform(self):
        return self._output_transform


class SummaryWriter(object):
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
            'title': 'loss'
        }
        self.add_callback_fn(self.loss_cbfn, kwargs)

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
            'title': ['{}/delta/{}', '{}/epsilon/{}']
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
            'title': '{}/scatter_plot/{}',
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
            'title': '{}/horizontal_mean/{}',
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
        elif sensor_type == 'StokesSensor':
            vmax = [image.reshape(image.shape[0], -1).max(axis=-1) * 1.25 for image in acquired_images]
        elif sensor_type == 'HybridSensor':
            vmax = [image.max() * 1.25 if type == 'Radiance'
                    else image.reshape(image.shape[0], -1).max(axis=-1) * 1.25 for image,type in zip(acquired_images,measurements.camera.projection.type)]

        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title':  ['Retrieval/view{}'.format(view) for view in range(num_images)],
            'vmax': vmax
        }
        self.add_callback_fn(self.estimated_images_cbfn, kwargs)
        acq_titles = ['Acquired/view{}'.format(view) for view in range(num_images)]
        self.write_image_list(0, acquired_images, acq_titles, vmax=kwargs['vmax'])

    def monitor_power_spectrum(self, estimator_name, ground_truth, ckpt_period=-1):
        """
        TODO
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': '{}/isotropic_power_spectrum/{}',
        }
        self.add_callback_fn(self.isotropic_power_spectrum_cbfn, kwargs)
        if hasattr(self, '_ground_truth'):
            self._ground_truth[estimator_name] = ground_truth
        else:
            self._ground_truth = OrderedDict({estimator_name: ground_truth})

    def save_ckpt_cbfn(self, kwargs=None):
        """
        Callback function that saves checkpoints .

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        timestr = time.strftime("%H%M%S")
        path = os.path.join(self.dir,  timestr + '.ckpt')
        self.optimizer.save_state(path)

    def loss_cbfn(self, kwargs):
        """
        Callback function that is called (every optimizer iteration) for loss monitoring.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        self.tf_writer.add_scalar(kwargs['title'], self.optimizer.loss, self.optimizer.iteration)

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
        for scatterer_name, gt_scatterer in self._ground_truth.items():
            est_scatterer = self.optimizer.medium.get_scatterer(scatterer_name)

            common_grid = est_scatterer.grid + gt_scatterer.grid
            a = est_scatterer.get_mask(threshold=0.0).resample(common_grid,method='nearest')
            b = gt_scatterer.get_mask(threshold=0.0).resample(common_grid,method='nearest')
            common_mask = shdom.GridData(data=np.bitwise_or(a.data,b.data),grid=common_grid)

            for parameter_name, parameter in est_scatterer.estimators.items():
                ground_truth = getattr(gt_scatterer, parameter_name)

                est_parameter_masked = copy.copy(parameter).resample(common_grid)
                est_parameter_masked.apply_mask(common_mask)
                est_param = est_parameter_masked.data.ravel()

                gt_param_masked = copy.copy(ground_truth).resample(common_grid)
                gt_param_masked.apply_mask(common_mask)
                gt_param = gt_param_masked.data.ravel()

                delta = (np.linalg.norm(est_param, 1) - np.linalg.norm(gt_param, 1)) / np.linalg.norm(gt_param, 1)
                epsilon = np.linalg.norm((est_param - gt_param), 1) / np.linalg.norm(gt_param,1)
                self.tf_writer.add_scalar(kwargs['title'][0].format(scatterer_name, parameter_name), delta, self.optimizer.iteration)
                self.tf_writer.add_scalar(kwargs['title'][1].format(scatterer_name, parameter_name), epsilon, self.optimizer.iteration)

    def domain_mean_cbfn(self, kwargs):
        """
        Callback function for monitoring domain averages of parameters.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        for scatterer_name, gt_scatterer in self._ground_truth.items():
            est_scatterer = self.optimizer.medium.get_scatterer(scatterer_name)
            for parameter_name, parameter in est_scatterer.estimators.items():
                if parameter.type == 'Homogeneous':
                    est_param = parameter.data
                else:
                    est_param = parameter.data.mean()

                ground_truth = getattr(gt_scatterer, parameter_name)
                if ground_truth.type == 'Homogeneous':
                    gt_param = ground_truth.data
                else:
                    gt_param = ground_truth.data.mean()

                self.tf_writer.add_scalars(
                    main_tag=kwargs['title'].format(scatterer_name, parameter_name),
                    tag_scalar_dict={'estimated': est_param, 'true': gt_param},
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
        for scatterer_name, gt_scatterer in self._ground_truth.items():
            est_scatterer = self.optimizer.medium.get_scatterer(scatterer_name)

            common_grid = est_scatterer.grid + gt_scatterer.grid
            a = est_scatterer.get_mask(threshold=0.0).resample(common_grid,method='nearest')
            b = gt_scatterer.get_mask(threshold=0.0).resample(common_grid,method='nearest')
            common_mask = shdom.GridData(data=np.bitwise_or(a.data,b.data),grid=common_grid)

            for parameter_name, parameter in est_scatterer.estimators.items():
                ground_truth = getattr(gt_scatterer, parameter_name)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)


                    est_parameter_masked = copy.deepcopy(parameter).resample(common_grid)
                    est_parameter_masked.apply_mask(common_mask)
                    est_param = est_parameter_masked.data
                    est_param[np.bitwise_not(common_mask.data)] = np.nan
                    est_param = np.nan_to_num(np.nanmean(est_param,axis=(0,1)))

                    gt_param_masked = copy.deepcopy(ground_truth).resample(common_grid)
                    gt_param_masked.apply_mask(common_mask)
                    gt_param = gt_param_masked.data
                    gt_param[np.bitwise_not(common_mask.data)] = np.nan
                    gt_param = np.nan_to_num(np.nanmean(gt_param,axis=(0,1)))

                fig, ax = plt.subplots()
                ax.set_title('{} {}'.format(scatterer_name, parameter_name), fontsize=16)
                ax.plot(est_param, common_grid.z, label='Estimated')
                ax.plot(gt_param, common_grid.z, label='True')
                ax.legend()
                ax.set_ylabel('Altitude [km]', fontsize=14)
                self.tf_writer.add_figure(
                    tag=kwargs['title'].format(scatterer_name, parameter_name),
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
        for scatterer_name, gt_scatterer in self._ground_truth.items():

            est_scatterer = self.optimizer.medium.get_scatterer(scatterer_name)
            common_grid = est_scatterer.grid + gt_scatterer.grid
            a = est_scatterer.get_mask(threshold=0.0).resample(common_grid,method='nearest')
            b = gt_scatterer.get_mask(threshold=0.0).resample(common_grid,method='nearest')
            common_mask = shdom.GridData(data=np.bitwise_or(a.data,b.data),grid=common_grid)

            parameters = est_scatterer.estimators.keys() if kwargs['parameters']=='all' else kwargs['parameters']
            for parameter_name in parameters:
                if parameter_name not in est_scatterer.estimators.keys():
                    continue
                parameter = est_scatterer.estimators[parameter_name]
                ground_truth = getattr(gt_scatterer, parameter_name)

                est_parameter_masked = copy.copy(parameter).resample(common_grid)
                est_parameter_masked.apply_mask(common_mask)
                est_param = est_parameter_masked.data.ravel()

                gt_param_masked = copy.copy(ground_truth).resample(common_grid)
                gt_param_masked.apply_mask(common_mask)
                gt_param = gt_param_masked.data.ravel()

                rho = np.corrcoef(est_param, gt_param)[1, 0]
                num_params = gt_param.size
                rand_ind = np.unique(np.random.randint(0, num_params, int(kwargs['percent'] * num_params)))
                max_val = max(gt_param.max(), est_param.max())
                fig, ax = plt.subplots()
                ax.set_title(r'{} {}: ${:1.0f}\%$ randomly sampled; $\rho={:1.2f}$'.format(scatterer_name, parameter_name, 100 * kwargs['percent'], rho),
                             fontsize=16)
                ax.scatter(gt_param[rand_ind], est_param[rand_ind], facecolors='none', edgecolors='b')
                ax.set_xlim([0, 1.1*max_val])
                ax.set_ylim([0, 1.1*max_val])
                ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
                ax.set_ylabel('Estimated', fontsize=14)
                ax.set_xlabel('True', fontsize=14)

                self.tf_writer.add_figure(
                    tag=kwargs['title'].format(scatterer_name, parameter_name),
                    figure=fig,
                    global_step=self.optimizer.iteration
                )

    def isotropic_power_spectrum_cbfn(self, kwargs):
        """
        TODO
        """
        for scatterer_name, gt_scatterer in self._ground_truth.items():

            est_scatterer = self.optimizer.medium.get_scatterer(scatterer_name)
            grid = est_scatterer.grid + gt_scatterer.grid
            a = est_scatterer.get_mask(threshold=0.0).resample(grid,method='nearest')
            b = gt_scatterer.get_mask(threshold=0.0).resample(grid,method='nearest')
            common_mask = shdom.GridData(data=np.bitwise_or(a.data,b.data),grid=grid)

            x,y,z = np.fft.fftfreq(grid.nx,d=grid.x[1]-grid.x[0]),np.fft.fftfreq(grid.ny,d=grid.y[1]-grid.y[0]), \
                np.fft.fftfreq(grid.nz,d=grid.z[1]-grid.z[0])
            X,Y,Z = np.meshgrid(np.fft.fftshift(y),np.fft.fftshift(x),np.fft.fftshift(z))
            isotropic_wavenumber = np.sqrt(X**2+Y**2+Z**2)
            nyquist_mask = (np.abs(X)<0.5/(grid.x[1]-grid.x[0])) & (np.abs(Y)<0.5/(grid.y[1]-grid.y[0])) & (np.abs(Z)<0.5/(grid.z[1]-grid.z[0]))
            bins = stats.mstats.mquantiles(isotropic_wavenumber[nyquist_mask],[i/20 for i in range(21)])
            bin_centres = np.array([(bins[i+1]+bins[i])/2.0 for i in range(len(bins)-1)])


            for parameter_name, parameter in est_scatterer.estimators.items():
                ground_truth = getattr(gt_scatterer, parameter_name)

                grid_gt = copy.copy(ground_truth).resample(grid)
                grid_gt.apply_mask(common_mask)
                grid_parameter = copy.copy(parameter).resample(grid)
                grid_parameter.apply_mask(common_mask)

                gt_spec = np.abs(np.fft.fftshift(np.fft.fftn((grid_gt.data - grid_gt.data.mean())/grid_gt.data.std())))**2
                param_spec = np.abs(np.fft.fftshift(np.fft.fftn((grid_parameter.data - grid_parameter.data.mean())/grid_parameter.data.std())))**2

                gt_resampled, bin_edge, number = stats.binned_statistic_dd(isotropic_wavenumber[nyquist_mask], gt_spec[nyquist_mask],
                                bins=[bins,],statistic='mean')
                param_resampled, bin_edge, number = stats.binned_statistic_dd(isotropic_wavenumber[nyquist_mask], param_spec[nyquist_mask],
                                bins=[bins,],statistic='mean')

                fig, ax = plt.subplots()
                ax.set_title(r'{} {}: Isotropic power spectrum'.format(scatterer_name, parameter_name ),
                             fontsize=16)
                ax.loglog(bin_centres, gt_resampled, '-o',label='True')
                ax.loglog(bin_centres, param_resampled, 'x-', label='Estimated')
                ax.legend()
                self.tf_writer.add_figure(
                    tag=kwargs['title'].format(scatterer_name, parameter_name),
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
            vmax = [vmax]*len(images)

        assert len(images) == len(titles), 'len(images) != len(titles): {} != {}'.format(len(images), len(titles))
        assert len(vmax) == len(titles), 'len(vmax) != len(images): {} != {}'.format(len(vmax), len(titles))

        for image, title, vm in zip(images, titles, vmax):
            if (image.shape[0] in (1,3,4)): #polarized
                #there is some overlap in this condition with a multispectral unpolarized case
                # with a very small number of pixels in the first dimension.
                image = image[0]

            if image.ndim==3: # polychromatic
                img_tensor = image[:,:,:]/ image.max()

            else:
                img_tensor = image[:,:,np.newaxis] / image.max()
            self.tf_writer.add_images(tag=title,
                img_tensor=img_tensor,
                dataformats='HWN',
                global_step=global_step
                )

#    def jacobian

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


class SpaceCarver(object):
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

        if isinstance(measurements.camera.projection, shdom.MultiViewProjection):
            self._projections = measurements.camera.projection.projection_list
        else:
            self._projections = [measurements.camera.projection]
        self._images = measurements.images

    def carve(self, grid, thresholds, agreement=0.75):
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
        self._rte_solver.set_grid(grid)
        volume = np.zeros((grid.nx, grid.ny, grid.nz))

        thresholds = np.array(thresholds)
        if thresholds.size == 1:
            thresholds = np.repeat(thresholds, len(self._images))
        else:
            assert thresholds.size == len(self._images), 'thresholds (len={}) should be of the same' \
                   'length as the number of images (len={})'.format(thresholds.size,  len(self._images))

        for projection, image, threshold in zip(self._projections, self._images, thresholds):

            if self._measurements.num_channels > 1:
                image = image[..., -1]
            if self._measurements.camera.sensor.type == 'StokesSensor':
                image = image[0]

            image_mask = image > threshold

            projection = projection[image_mask.ravel(order='F') == 1]

            carved_volume = core.space_carve(
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

        volume = volume * 1.0 / len(self._images)
        mask = GridData(grid, volume > agreement)
        return mask

    @property
    def grid(self):
        return self._grid


class OpticalDepth(object):
    """
    TODO
    """

    def __init__(self, rte_solver):
        self._rte_solver = rte_solver

    def integrate(self, projection):
        """
        TODO
        """
        if isinstance(projection, shdom.MultiViewProjection):
            projections = projection.projection_list
        else:
            projections = [projection]
        optical_depth = np.full(self._rte_solver._npts, 999)
        for projection in projections:
            tau = core.optical_depth(
                nx=self._rte_solver._nx,
                ny=self._rte_solver._ny,
                nz=self._rte_solver._nz,
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
                total_ext=self._rte_solver._total_ext[:self._rte_solver._npts]
            )
            optical_depth = np.minimum(tau, optical_depth)
        return optical_depth.reshape(self._rte_solver._nx, self._rte_solver._ny, self._rte_solver._nz)


class LocalOptimizer(object):
    """
    The LocalOptimizer class takes care of the under the hood of the optimization process.
    To run the optimization the following methods should be called:
       [required] optimizer.set_measurements()
       [required] optimizer.set_rte_solver()
       [required] optimizer.set_medium_estimator()
       [optional] optimizer.set_writer()

    Parameters
    ----------
    options: dict
        The option dictionary for the optimizer
    n_jobs: int, default=1
        The number of jobs to divide the gradient computation into.
    init_solution: bool
        True will re-initialize the solution process every iteration.
        False will use the previous step RTE solution to initialize the current RTE solution.
    method: str, default='L-BFGS-B'
        The optimizer solution method: 'L-BFGS-B', 'TNC'

    Notes
    -----
    Currently only L-BFGS-B optimization method is supported.
    For documentation:
        https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
    """
    def __init__(self, method, options={}, n_jobs=1, init_solution=True, jacobian_flag=False):
        self._medium = None
        self._rte_solver = None
        self._measurements = None
        self._writer = None
        self._images = None
        self._iteration = 0
        self._loss = None
        self._jacobian = None
        self._jacobian_flag = jacobian_flag
        self._n_jobs = n_jobs
        self._init_solution = init_solution
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
        Set the MediumEstimator for the optimizer.

        Parameters
        ----------
        medium_estimator: shdom.MediumEstimator
            The MediumEstimator
        """
        self._medium = medium_estimator

    def set_rte_solver(self, rte_solver):
        """
        Set the RteSolver for the SHDOM iterations.

        Parameters
        ----------
        rte_solver: shdom.RteSolver
            The RteSolver
        """
        if isinstance(rte_solver, shdom.RteSolverArray):
            self._rte_solver = rte_solver
        else:
            self._rte_solver = shdom.RteSolverArray([rte_solver])

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
        self.set_state(state)
        gradient, loss, images, jacobian = self.medium.compute_gradient(
            rte_solvers=self.rte_solver,
            measurements=self.measurements,
            n_jobs=self.n_jobs,
            jacobian_flag=self._jacobian_flag
        )
        # print(state,gradient,loss)
        self._loss = loss
        self._images = images
        if self._jacobian_flag:
            self._jacobian = jacobian

        return loss, gradient

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

        assert self.rte_solver.num_solvers == self.measurements.num_channels == self.medium.num_wavelengths, \
            'RteSolver has {} solvers, Measurements have {} channels and Medium has {} wavelengths'.format(self.rte_solver.num_solvers, self.measurements.num_channels, self.medium.num_wavelengths)

        self.rte_solver.set_medium(self.medium)
        self.rte_solver.init_solution()
        self.medium.compute_direct_derivative(self.rte_solver)
        self._num_parameters = self.medium.num_parameters

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
        self.rte_solver.set_medium(self.medium)
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
        #self.medium

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
        file = open(path,'rb')
        data = file.read()
        file.close()
        state = pickle.loads(data)
        self.set_state(state)

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
    def jacobian(self):
        return self._jacobian

    @property
    def jacobian_flag(self):
        return self._jacobian_flag

class ProximalProjection(object):
    """TODO"""
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def __call__(self):
        if self.optimizer.writer is not None:
            self.optimizer.writer.attach_optimizer(self.optimizer)
        self.optimizer.minimize()


class LocalOptimizerADMM(LocalOptimizer):
    """"TODO"""
    def __init__(self, method, options={}, n_jobs=1):
        self._options = options
        self._n_jobs = n_jobs
        self._method = method
        self._proximal_projections = []

    def init_optimizer(self):
        """TODO"""
        self._iter = 0
        if self.medium.num_estimators > 1:
            raise NotImplementedError('Multiple medium estimators not implemented')

        scatterer_estimator = next(iter(self.medium.estimators.values()))
        for param, param_estimator in scatterer_estimator.estimators.items():
            optimizer = shdom.LocalOptimizer(method=self.method, options=self.options, n_jobs=self.n_jobs)
            optimizer.set_measurements(self.measurements)
            optimizer.set_rte_solver(self.rte_solver)
            optimizer.set_writer(self.writer)
            medium_estimator = shdom.MediumEstimator(
                grid=self.medium.grid,
                loss_type=self.medium._loss_type,
                exact_single_scatter=self.medium._exact_single_scatter,
                stokes_weights=self.medium._stokes_weights
            )

            for name, scatterer in self.medium.scatterers.items():
                if scatterer == scatterer_estimator:
                    if param == 'lwc':
                        lwc = scatterer.lwc
                        reff = shdom.GridData(scatterer.reff.grid, scatterer.reff.data)
                        veff = shdom.GridData(scatterer.veff.grid, scatterer.veff.data)
                    elif param == 'reff':
                        lwc = shdom.GridData(scatterer.lwc.grid, scatterer.lwc.data)
                        reff = scatterer.reff
                        veff = shdom.GridData(scatterer.veff.grid, scatterer.veff.data)
                    elif param == 'veff':
                        lwc = shdom.GridData(scatterer.lwc.grid, scatterer.lwc.data)
                        reff = shdom.GridData(scatterer.reff.grid, scatterer.reff.data)
                        veff = scatterer.veff
                    proximal_scatterer = shdom.MicrophysicalScattererEstimator(scatterer.mie, lwc, reff, veff)
                    medium_estimator.add_scatterer(proximal_scatterer, name)
                else:
                    medium_estimator.add_scatterer(scatterer, name)

            optimizer.set_medium_estimator(medium_estimator)
            self.proximal_projections.append(shdom.ProximalProjection(optimizer))

    def minimize(self):
        """TODO"""
        self.init_optimizer()
        iter = 0
        while (iter < maxiter):
            [proximal() for proximal in self.proximal_projections]
            iter += 1

    @property
    def proximal_projections(self):
        return self._proximal_projections

class GlobalOptimizer(object):
    """
    The GlobalOptimizer class takes care of the under the hood of the global optimization process.
    To run the optimization a local optimizer should be set (see set_local_optimizer method)

    Parameters
    ----------
    local_optimizer: shdom.LocalOptimizer, optional
        A local optimizer object, after all initializations (see LocalOptimizer class)
    method: str
        The global optimization method.

    Notes
    -----
    Currently only basin-hopping global optimization is supported.

    For documentation:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html
    """
    def __init__(self, local_optimizer=None, method='basin-hopping'):
        self._iteration = 1
        self._best_minimum_f = np.Inf
        self._best_minimum_iteration = 0
        self._best_minimum_x = None
        self._loss = None
        self._take_step = None
        self._tf_writer = None
        if method != 'basin-hopping':
            raise NotImplementedError('Optimization method [{}] not implemented'.format(method))
        self._method = method
        self.set_local_optimizer(local_optimizer)

    def set_local_optimizer(self, local_optimizer):
        """
        Set the local optimizer.

        Parameters
        ----------
        local_optimizer: shdom.LocalOptimizer
            A local optimizer object, after all initializations (see LocalOptimizer class)
        """
        self._local_optimizer = local_optimizer
        self._take_step = RandomStep(local_optimizer.medium)
        self.init_local_optimizer()

    def minimize(self, niter_success, T=1e-3, maxiter=100, stepsize=0.5, interval=10, disp=True):
        """
        Global minimization with respect to the parameters defined.

        Parameters
        ----------
        niter_success: int
            Stop the run if the global minimum candidate remains the same for this number of iterations.
        T: float
            The “temperature” parameter for the accept or reject criterion.
            Higher “temperatures” mean that larger jumps in function value will be accepted.
            For best results T should be comparable to the separation (in function value) between local minima
        maxiter : int,
            The number of basin hopping iterations
        stepsize: float,
            Maximum step size for use in the random displacement. See RandomStep object for more info.
        interval: int,
            interval for how often to update the stepsize
        disp: bool
            Display information of the optimization process.
        """
        if self.method == 'basin-hopping':
            result = basinhopping(func=self.local_optimizer.objective_fun,
                                  x0=self.local_optimizer.get_state(),
                                  minimizer_kwargs=self.local_minimizer_kwargs,
                                  disp=disp,
                                  niter=maxiter,
                                  take_step=self.take_step,
                                  stepsize=stepsize,
                                  callback=self.callback,
                                  T=T,
                                  interval=interval,
                                  niter_success=niter_success)
        return result

    def init_local_optimizer(self):
        """Initialize the local optimizer and writer (if any)."""
        self.local_optimizer.init_optimizer()
        self._best_minimum_x = self.local_optimizer.get_state()
        local_options = self.local_optimizer.options
        local_options['disp'] = False
        self._local_minimizer_kwargs = {
            'method': self.local_optimizer.method,
            'jac': True,
            'bounds': self.local_optimizer.get_bounds(),
            'options': local_options,
            'callback': self.local_optimizer.callback
        }

        # If local writer exists, modify checkpoint saving for global optimization
        self._save_checkpoints = False
        if self.local_optimizer.writer is not None:
            self._tf_writer = tb.SummaryWriter(self.local_optimizer.writer.dir)
            self.update_writer(self.iteration)
            self.update_ckpt_saving()

    def callback(self, state, loss, accept):
        """
        The callback function invokes the callbacks defined by the writer (if any).
        Additionally it keeps track of the iteration number.

        Parameters
        ----------
        state: np.array(shape=(self.num_parameters, dtype=np.float64)
            The state vector of the local minimum found
        loss: float64
            The loss of the local minimum found
        accept: bool
            Whether of not the minimum was accepted
        """
        self.local_optimizer._iteration = 0
        if loss < self._best_minimum_f:
            self._best_minimum_f = loss
            self._best_minimum_iteration = self.iteration
            self._best_minimum_x = state
            if self.tf_writer is not None:
                self.tf_writer.add_scalar('Basin loss', loss, self.iteration)

            if self._save_checkpoints:
                time_passed = time.time() - self._ckpt_time
                if time_passed > self._ckpt_period:
                    self.local_optimizer.writer.save_ckpt_cbfn()
        else:
            shutil.rmtree(self.local_optimizer.writer.tf_writer.log_dir)

        self._iteration += 1
        if self.tf_writer is not None:
            self.update_writer(self.iteration)

    def update_writer(self, iteration):
        """
        Update summary writer to global iteration index.
        This method will place Basin<iteration> at the beginning of the path.

        Parameters
        ----------
        iteration: int
            The current iteration number.
        """
        log_dir = os.path.join(self.local_optimizer.writer.dir, 'Basin{}'.format(iteration))
        self.local_optimizer.writer._tf_writer = tb.SummaryWriter(log_dir)

    def update_ckpt_saving(self):
        """If checkpoint saving is defined then update to save only global minima."""
        cbfn_names = [cbfn.__name__ for cbfn in self.local_optimizer.writer.callback_fns]
        if 'save_ckpt_cbfn' in cbfn_names:
            self._save_checkpoints = True
            cbfn_index = cbfn_names.index('save_ckpt_cbfn')
            self._ckpt_period = self.local_optimizer.writer.kwargs[cbfn_index]['ckpt_period']
            self._ckpt_time = self.local_optimizer.writer.kwargs[cbfn_index]['ckpt_time']
            self.local_optimizer.writer.kwargs[cbfn_index]['ckpt_period'] = np.Inf

    @property
    def method(self):
        return self._method

    @property
    def take_step(self):
        return self._take_step

    @property
    def local_minimizer_kwargs(self):
        return self._local_minimizer_kwargs

    @property
    def local_optimizer(self):
        return self._local_optimizer

    @property
    def iteration(self):
        return self._iteration

    @property
    def tf_writer(self):
        return self._tf_writer


class RandomStep(object):
    """"
    Replaces the default step taking routine of the basin hopping minimizer.
    The default step taking routine is a random displacement of the coordinates, but other step taking algorithms may be better for some systems.
    Here a custume step taking procedure is defined taking into account the parameter bounds.

    Parameters
    ----------
    medium: shdom.MediumEstimator
        A MediumEstimator object
    stepsize: float
        A factor to the per-parameter stepsize that is optimized throughout iterations

    Notes
    -----
    stepsize should be on the order of the scaled parameters (see preconditioning scale factor in GridDataEstimator)
    """
    def __init__(self, medium, stepsize=0.5):
        self.stepsize = stepsize
        bounds = medium.get_bounds()
        self.min_bound = [bound[0] for bound in bounds]
        self.max_bound = [bound[1] for bound in bounds]

    def __call__(self, x):
        x += np.random.uniform(-self.stepsize, self.stepsize)
        return np.clip(x, self.min_bound, self.max_bound)


class Jacobians(object):
    """
    Object that stores the Jacobian as a function of scatterer, unknown parameter, wavelength
    and stokes vector.
    """
    def __init__(self,projection=None, rte_solver=None, estimators=None,output=None):

        #JACOBIAN ERROR WHEN RTE_SOVER GRID IS NOT SAME AS ESTIMATOR GRID
        #
        self._projection = None
        self._total_pix = None
        self._npts = None
        self._wavelengths = None
        self._jacobian_tree = OrderedDict()
        self._masks = OrderedDict()
        self._nstokes = None
        self._stokes_strings = None
        if projection is not None:
            self.set_projection(projection)
        if rte_solver is not None:
            self.set_rte_solver(rte_solver)
        if estimators is not None:
            self.set_estimator(estimators)

        self._split_jacobian = None
        self._split_ptrs = None
        self._image_ptrs = None

        if output is not None:
            self.add_jacobian(output=output)

    def set_projection(self,projection):
        """
        """
        if isinstance(projection.npix, list):
            total_pix = np.sum(projection.npix)
        else:
            total_pix = projection.npix

        self._projection = projection
        self._total_pix = total_pix

    def set_rte_solver(self,rte_solver):
        """
        """
        if isinstance(rte_solver,shdom.RteSolver):
            rte_solver = shdom.RteSolverArray(solver_list=[rte_solver])

        self._npts = rte_solver.solver_list[0]._nbpts
        self._wavelengths = np.atleast_1d(rte_solver.wavelength)
        self._nstokes = [rte_solver._nstokes for rte_solver in rte_solver.solver_list]
        self._stokes_strings = [['I','Q','U','V'][:nstokes] for nstokes in self._nstokes]

    def set_estimator(self,estimators):
        """
        """
        #map the derivatives to indices in the jacobian array.
        i=0
        for estimator_name in estimators.keys():
            derivatives = estimators[estimator_name].derivatives.keys()
            self._masks[estimator_name] = estimators[estimator_name].get_mask(threshold=0.0)
            for parameter_name in derivatives:
                self._jacobian_tree[estimator_name+'/'+parameter_name] = i
                i+=1

    def add_jacobian(self,output=None,split_jacobian=None,split_ptrs=None,image_ptrs=None):
        """
        """
        if output is not None:
            #extract jacobian from gradient_l2 output.
            jacobian = np.concatenate(list(map(lambda x: x[3], output)),axis=-1)
            jacobian_ptr = np.concatenate(list(map(lambda x: x[4], output)), axis=-1)
            counters = np.concatenate(list(map(lambda x: x[5], output)),axis=-1)

            #deal with the fact that jacobian_ptr[0] only contains relative ptrs
            #not absolute due n_job projection split.
            total = []
            for i,proj in enumerate(self.projection.split(len(counters)//len(self.wavelengths))):
                total.append(np.sum(proj.npix))
            total = np.tile(np.cumsum(np.array([0]+total),axis=-1)[:-1],[len(self.wavelengths)])
            total = np.array([[total[i]]*counters[i] for i in range(len(counters))])
            total = np.concatenate(total,axis=-1).astype(np.int32)
            jacobian_ptr[0] += total
            jacobian_ptr -= 1 # change to zero indexing. . .

            #deal with the fact that different wavelengths can have different numbers of non-sparse entries.
            counter_reduced = np.array(np.split(counters,len(self.wavelengths),axis=-1)).sum(axis=-1)
            self._split_jacobian = np.split(jacobian, np.cumsum(counter_reduced),axis=-1)
            self._split_ptrs = np.split(jacobian_ptr,np.cumsum(counter_reduced),axis=-1)
            #pointers to each image (rather than each pixel)
            self._image_ptrs = [np.digitize(jacobian_ptr[0],bins=np.cumsum(np.array([0]+self.npix)))-1 for jacobian_ptr in self._split_ptrs]
        else:
            self._split_jacobian = split_jacobian
            self._split_ptrs = split_ptrs
            self._image_ptrs = image_ptrs

    def save(self, path):
        file = open(path, 'wb')
        file.write(pickle.dumps(self.__dic__, -1))
        file.close()

    def load(self,path):
        file = open(path, 'wb')
        data = file.read()
        file.close()
        self.__dic__ = data

    def get_jacobian(self,wavelength=None,stokes='I',parameter_name='lwc',estimator_name='cloud',dense=False,
                mode='csr',reduce_wavelengths=False, reduce_stokes=False,reduce_angles=False,reduce_images=False
                ,gradout=False):
        """
        A function that retrieves the desired jacobian or simplified version. The various axes for each
        unknown parameter (wavelength, stokes, images, angles) can be summed over by setting the appropriate reduce
        variable to True. This overrides the selection of individual indices given by other keyword arguments.
        """

        if gradout:
            reduce_wavelengths=reduce_stokes=reduce_angles=True

        derivative_ptr = self._jacobian_tree[estimator_name+'/'+parameter_name]

        if reduce_wavelengths:
            wavelength_list = self.wavelengths
        else:
            wavelength_list = np.atleast_1d(self.wavelengths[np.where(self.wavelengths==wavelength)])

        sparse_matrix_full = None
        for wv in wavelength_list:
            wv_ptr = list(self.wavelengths).index(wv)
            stokes_ptr = self._stokes_strings[wv_ptr].index(stokes)
            col_ptr = self._split_ptrs[wv_ptr][1]

            if reduce_images:
                row_ptr = self._image_ptrs[wv_ptr]
                row_len = len(self.npix)
            else:
                row_ptr = self._split_ptrs[wv_ptr][0]
                row_len = self.total_pix

            if reduce_stokes:
                jacobian = np.sum(self._split_jacobian[wv_ptr][:,derivative_ptr],axis=0)
            else:
                jacobian = self._split_jacobian[wv_ptr][stokes_ptr,derivative_ptr]
            #initialize as coo to ensure image reduce can occur.
            sparse_matrix = sparse.coo_matrix((jacobian,(row_ptr,col_ptr)),shape=(row_len,self.npts))
            sparse_matrix = sparse_matrix.tocsr() #change to csr for arithmetic.
            if sparse_matrix_full is None:
                sparse_matrix_full = sparse_matrix
            else:
                sparse_matrix_full += sparse_matrix

        if reduce_angles:
            output = sparse_matrix_full.tocsr().sum(axis=0)
        else:
            if mode == 'coo':
                output = sparse_matrix_full.tocoo()
            elif mode == 'csr':
                output = sparse_matrix_full.tocsr()
            elif mode =='csc':
                output = sparse_matrix_full.tocsc()
        if dense:
            return output.todense().A
        else:
            return output

    # def rank_images(self,estimator_name,parameter_name,wavelength,stokes,subset_num):
    #     """
    #     Rank subsets of the images according to how well the jacobian is approximated
    #     measured by the Frobenius norm. This can be used to optimize the viewing configuration.
    #     """
    #     num_images = len(self.npix)
    #     assert subset_num < num_images, 'image subset {} must be less than the number of images {}'.format(subset_num,num_images)
    #     image_indices = [i for i in range(num_images)]
    #     combinations = []
    #     for comb in itertools.combinations(image_indices, subset_num):
    #         combinations.append(comb)
    #         #print(comb)
    #
    #     sparse_matrix = self.get_jacobian(wavelength,stokes=stokes,parameter_name=parameter_name,estimator_name=estimator_name)
    #     data = sparse_matrix.data
    #     image_ptrs = self._image_ptrs[list(self.wavelengths).index(wavelength)]
    #     approximation_performance = np.zeros(len(combinations))
    #     print(len(combinations))
    #     for i,combination in enumerate(combinations):
    #         print(i)
    #         data_copy = np.copy(data)
    #         condition = np.zeros(image_ptrs.shape,dtype=bool)
    #         for index in combination:
    #              condition[np.where(image_ptrs == index)] = True
    #         data_copy[np.where(np.bitwise_not(condition))] = 0.0
    #         approximation_performance[i] = np.sqrt(np.sum((data_copy - data)**2))
    #
    #     return combinations, approximation_performance

    # def max_vol_approximation(self,estimator,scatterer,wavelength,stokes,subset_num):
    #     """
    #     Algorithm:
    #     for k in range(subset_num):
    #         Find the image column block with the maximum Frobenius norm.
    #         For each column in the column block:
    #             subtract the projection of that column on columns in other column blocks from those columns
    #             Add the column to the approximated matrix as part of its column block.
    #     end do.
    #     """
    #     num_images = len(self.npix)
    #     assert subset_num < num_images, 'image subset {} must be less than the number of images {}'.format(subset_num,num_images)
    #
    #     sparse_matrix = self.jacobians[estimator][scatterer][wavelength][stokes]
    #
    #     print(sparse_matrix.getrow(0))
    #     #csc = sparse_matrix.tocsc(copy=True)
    #     data = sparse_matrix.data
    #     approximation = []
    #
    #     for i in range(subset_num):
    #         minimum_vector = []
    #         for j in range(self._image_ptrs):
    #             pass
    #         #find
    #     return None

    def get_mask(self, estimator_name='cloud'):
        """
        TODO
        """
        return self._masks[estimator_name]

    def get_grid(self, estimator_name='cloud'):
        """
        TODO
        """
        return self._masks[estimator_name].grid

    def row_as_grid_data(self,jacobian, row_index, estimator_name='cloud',apply_mask=True):
        """
        TODO
        """
        column_data = jacobian.getrow(row_index).todense().A
        grid = self.get_grid(estimator_name=estimator_name)
        data=column_data.reshape(grid.nx,grid.ny,grid.nz)
        grid_data = shdom.GridData(data=data,grid=grid)
        if apply_mask:
            mask = self.get_mask(estimator_name=estimator_name)
            grid_data.apply_mask(mask)
        return grid_data

    def column_as_images(self,grid_indices,estimator_name='cloud',parameter_name='lwc'):
        """
        TODO
        """
        grid = self.get_grid(estimator_name=estimator_name)
        column_index = grid_indices[0]*grid.ny*grid.nz + grid_indices[1]*grid.nz+grid_indices[2]

        pixels = []
        for i,wavelength in enumerate(self.wavelengths):
            monochromatic_pixels = []
            for j,stokes in enumerate(self._stokes_strings[i]):

                column_data = self.get_jacobian(wavelength=wavelength,
                                                stokes=stokes,
                                                estimator_name=estimator_name,
                                                parameter_name=parameter_name,
                                                ).getcol(column_index).todense().A.squeeze()
                monochromatic_pixels.append(column_data)
            monochromatic_pixels = np.stack(monochromatic_pixels,axis=0)
            pixels.append(monochromatic_pixels)
        pixels = np.concatenate(pixels,axis=-1)
        sensor = shdom.StokesSensor()
        images = sensor.make_images(pixels,self.projection,len(self.wavelengths))
        return images

    def squeeze_jacobians(self,estimator_name='cloud'):
        """
        Generates the subset of Jacobian's data and coords corresponding to a matrix of size:
        number_of_valid_pixels x number_of_state_elements.
        """
        mask = self.get_mask(estimator_name=estimator_name)
        raveled_mask = mask.data.ravel().astype(bool)
        number_of_state_elements = raveled_mask[np.where(raveled_mask==True)].size
        valid_state_indices = np.where(raveled_mask==True)[0]

        squeezed_ptrs = []
        squeezed_jacobian = []
        squeezed_image_ptrs = []
        new_ptrs = []
        new_shape = []

        for i,(wv,jac,ptrs,img_ptrs) in enumerate(zip(self.wavelengths,
                                                    self._split_jacobian,
                                                    self._split_ptrs,
                                                    self._image_ptrs)):
            print('blah1')
            valid_pixel_indices,new_pixel_ptr = np.unique(ptrs[0],return_inverse=True)#list(set(ptrs[0]))
            number_of_valid_pixels = len(valid_pixel_indices)
            print('blah2')
            # new_pixel_ptr = np.ones(ptrs[0].shape,dtype=np.int)*-1
            # for j,valid_pixel in enumerate(valid_pixel_indices):
            #     new_pixel_ptr[np.where(ptrs[0]==valid_pixel)] = j
            #valid_state_indices, new_state_ptr = np.unique(ptrs[1],return_inverse=True)

            data_condition = raveled_mask[ptrs[1].astype(np.int)]
            ptrs_update = ptrs[:,data_condition]
            valid_state_indices, new_state_ptr = np.unique(ptrs_update[1],return_inverse=True)
            # new_state_ptr = np.ones(ptrs[1].shape,dtype=np.int)*-1
            # for k,valid_state in enumerate(valid_state_indices):
            #      new_state_ptr[np.where(ptrs[1]==valid_state)] = k
            #      print(k)
            # print('blah3')
            # data_condition = np.where((new_state_ptr>=0))[0]

            squeezed_ptrs.append(ptrs_update)
            squeezed_jacobian.append(jac[:,:,data_condition])
            squeezed_image_ptrs.append(img_ptrs[data_condition])
            new_ptrs.append(np.stack([new_pixel_ptr[data_condition],new_state_ptr],axis=0))
            new_shape.append((number_of_valid_pixels,number_of_state_elements))
        return squeezed_ptrs, squeezed_jacobian, squeezed_image_ptrs, new_ptrs, new_shape

    # def svd(self, estimator,scatterer, wavelength,**kwargs):
    #     """
    #     """
    #     sparse_matrix = self._jacobians[estimator][scatterer][wavelength]
    #     #time1 = time.time()
    #     #import scipy.linalg as linalg
    #     #out = linalg.svds(sparse_matrix.todense(),full_matrices=False)#,**kwargs)
    #     #print('1',kwargs['k'],time.time()-time1)
    #     #time2 = time.time()
    #     out2 = sparse.linalg.svds(sparse_matrix,**kwargs)
    #     #out2 = sparsesvd.sparsesvd(sparse_matrix,**kwargs)
    #     #print('2',kwargs['k'],time.time()-time2)
    #     return out2

    @property
    def jacobian_tree(self):
        return list(self._jacobian_tree.keys())

    @property
    def projection(self):
        return self._projection

    @property
    def npix(self):
        return self.projection.npix

    @property
    def wavelengths(self):
        return self._wavelengths

    @property
    def npts(self):
        return self._npts

    @property
    def total_pix(self):
        return self._total_pix
