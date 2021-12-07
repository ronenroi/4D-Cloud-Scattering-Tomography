import os, time
import argparse
import numpy as np
import shdom
from optimize_dynamic_extinction_lbfgs import OptimizationScript as ExtinctionOptimizationScript
import sys
from os.path import dirname
sys.path.append(dirname(__file__))

class OptimizationScript(object):
    """
    Optimize: Micro-physics
    ----------------------
    Estimate micro-physical properties based on multi-spectral radiance/polarization measurements.
    Note that for convergence a fine enough sampling of effective radii and variances should be pre-computed in the
    Mie tables used by the forward model. This is due to the linearization of the phase-function and it's derivatives.

    Measurements are simulated measurements using a forward rendering script
    (e.g. scripts/render_radiance_toa.py).

    For example usage see the README.md

    For information about the command line flags see:
      python scripts/optimize_dynamic_microphysics_lbfgs.py --help

    Parameters
    ----------
    scatterer_name: str
        The name of the scatterer that will be optimized.
    """
    def __init__(self, scatterer_name='cloud'):
        self.scatterer_name = scatterer_name


    def optimization_args(self, parser):
        """
        Add common optimization arguments that may be shared across scripts.

        Parameters
        ----------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.

        Returns
        -------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.
        """
        parser.add_argument('--input_dir',
                            help='Path to an input directory where the forward modeling parameters are be saved. \
                                  This directory will be used to save the optimization results and progress.')
        parser.add_argument('--reload_path',
                            help='Reload an optimizer or checkpoint and continue optimizing from that point.')
        parser.add_argument('--log',
                            help='Write intermediate TensorBoardX results. \
                                  The provided string is added as a comment to the specific run.')
        parser.add_argument('--use_forward_grid',
                            action='store_true',
                            help='Use the same grid for the reconstruction. This is a sort of inverse crime which is \
                                  usefull for debugging/development.')
        parser.add_argument('--use_forward_mask',
                            action='store_true',
                            help='Use the ground-truth cloud mask. This is an inverse crime which is \
                                  usefull for debugging/development.')
        parser.add_argument('--n_jobs',
                            default=1,
                            type=int,
                            help='(default value: %(default)s) Number of jobs for parallel rendering. n_jobs=1 uses no parallelization')
        parser.add_argument('--maxiter',
                            default=10,
                            type=int,
                            help='(default value: %(default)s) Maximum number of Line-searcg iterations.')
        parser.add_argument('--stokes_weights',
                            nargs=4,
                            default=[1.0, 0.0, 0.0, 0.0],
                            type=float,
                            help='(default value: %(default)s) Loss function weights for stokes vector components [I, Q, U, V]')#TODO
        parser.add_argument('--loss_type',
                            choices=['l2', 'normcorr', 'l2_weight'],
                            default='l2',
                            help='Different loss functions for optimization. Currently only l2 is supported.')
        parser.add_argument('--num_mediums',
                            default=30,
                            type=int,
                            help='Number of different mediums to be reconstructed simultaneously.')

        return parser

    def medium_args(self, parser):
        """
        Add common medium arguments that may be shared across scripts.

        Parameters
        ----------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.

        Returns
        -------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.
        """
        parser.add_argument('--use_forward_lwc',
                            action='store_true',
                            help='Use the ground-truth LWC.')
        parser.add_argument('--use_forward_reff',
                            action='store_true',
                            help='Use the ground-truth effective radius.')
        parser.add_argument('--use_forward_veff',
                            action='store_true',
                            help='Use the ground-truth effective variance.')
        parser.add_argument('--const_lwc',
                            action='store_true',
                            help='Keep liquid water content constant at a specified value (not optimized).')
        parser.add_argument('--const_reff',
                            action='store_true',
                            help='Keep effective radius constant at a specified value (not optimized).')
        parser.add_argument('--const_veff',
                            action='store_true',
                            help='Keep effective variance constant at a specified value (not optimized).')
        parser.add_argument('--radiance_threshold',
                            default=[0.03],
                            nargs='+',
                            type=np.float32,
                            help='(default value: %(default)s) Threshold for the radiance to create a cloud mask.'
                                 'Threshold is either a scalar or a list of length of measurements.')
        return parser

    def parse_arguments(self):
        """
        Handle all the argument parsing needed for this script.

        Returns
        -------
        args: arguments from argparse.ArgumentParser()
            Arguments required for this script.
        cloud_generator: a shdom.CloudGenerator object.
            Creates the cloudy medium. The loading of Mie tables takes place at this point.
        air_generator: a shdom.AirGenerator object
            Creates the scattering due to air molecules
        """
        parser = argparse.ArgumentParser()
        parser = self.optimization_args(parser)
        parser = self.medium_args(parser)

        # Additional arguments to the parser
        subparser = argparse.ArgumentParser(add_help=False)
        subparser.add_argument('--init')
        subparser.add_argument('--add_rayleigh', action='store_true')
        parser.add_argument('--init',
                            default='Homogeneous',
                            help='(default value: %(default)s) Name of the generator used to initialize the atmosphere. \
                                  for additional generator arguments: python scripts/optimize_extinction_lbgfs.py --generator GENERATOR --help. \
                                  See generate.py for more documentation.')
        parser.add_argument('--add_rayleigh',
                            action='store_true',
                            help='Overlay the atmosphere with (known) Rayleigh scattering due to air molecules. \
                                  Temperature profile is taken from AFGL measurements of summer mid-lat.')

        init = subparser.parse_known_args()[0].init
        add_rayleigh = subparser.parse_known_args()[0].add_rayleigh

        CloudGenerator = None
        if init:
            CloudGenerator = getattr(shdom.dynamic_scene, init)
            parser = CloudGenerator.update_parser(parser)

        AirGenerator = None
        if add_rayleigh:
            AirGenerator = shdom.generate.AFGLSummerMidLatAir
            parser = AirGenerator.update_parser(parser)

        self.args = parser.parse_args()
        self.cloud_generator = CloudGenerator(self.args) if CloudGenerator is not None else None
        self.air_generator = AirGenerator(self.args) if AirGenerator is not None else None

    def get_medium_estimator(self, measurements: shdom.DynamicMeasurements, ground_truth: shdom.DynamicScatterer):
        """
        Generate the medium estimator for optimization.

        Parameters
        ----------
        measurements: shdom.DynamicMeasurements
            The acquired measurements.
        ground_truth: shdom.DynamicScatterer


        Returns
        -------
        medium_estimator: shdom.MediumEstimator
            A medium estimator object which defines the optimized parameters.
        """

        num_of_mediums = 1
        time_list = measurements.time_list
        temporary_scatterer_list = ground_truth._temporary_scatterer_list


        assert isinstance(num_of_mediums, int) and num_of_mediums <= len(time_list)
        time_list = np.mean(np.split(np.array(time_list), num_of_mediums), 1)

        wavelength = ground_truth.wavelength
        if not isinstance(wavelength, list):
            wavelength = [wavelength]
        # Define the grid for reconstruction
        if self.args.use_forward_grid:
            lwc_grid = []
            reff_grid = []
            veff_grid = []
            l_grid = []
            r_grid = []
            v_grid = []
            for temporary_scatterer in temporary_scatterer_list:
                l_grid.append(temporary_scatterer.scatterer.lwc.grid)
                r_grid.append(temporary_scatterer.scatterer.reff.grid)
                v_grid.append(temporary_scatterer.scatterer.veff.grid)

            l_grid = np.split(np.array(l_grid), num_of_mediums)
            r_grid = np.split(np.array(r_grid), num_of_mediums)
            v_grid = np.split(np.array(v_grid), num_of_mediums)
            for l, r, v in zip(l_grid, r_grid, v_grid):
                lwc_grid.append(np.sum(l))
                reff_grid.append(np.sum(r))
                veff_grid.append(np.sum(v))
            grid = lwc_grid[0]

        else:
            grid = self.cloud_generator.get_grid()
            grid = shdom.Grid(x=grid.x + temporary_scatterer_list[0].scatterer.lwc.grid.xmin, y=grid.y + temporary_scatterer_list[0].scatterer.lwc.grid.ymin, z=grid.z)
            lwc_grid = reff_grid = veff_grid = [grid]*num_of_mediums
        if 1:#self.args.one_dim_reff:
            for i, grid in enumerate(reff_grid):
                reff_grid[i] = shdom.Grid(z=grid.z)

        num_of_mediums = self.args.num_mediums

        # Find a cloud mask for non-cloudy grid points
        self.thr = 1e-3

        if self.args.use_forward_mask:
            mask_list = ground_truth.get_mask(threshold=self.thr)
            show_mask = 0
            if show_mask:
                a = (mask_list[0].data).astype(int)
                print(np.sum(a))
                shdom.cloud_plot(a)

        else:
            dynamic_carver = shdom.DynamicSpaceCarver(measurements)
            mask_list, dynamic_grid, cloud_velocity = dynamic_carver.carve(grid, agreement=0.9,
                                                                           time_list=measurements.time_list,
                                                                           thresholds=self.args.radiance_threshold,
                                                                           vx_max=0, vy_max=0,
                                                                           gt_velocity=[0,0,0],
                                                                           verbose = False)
            show_mask = 1
            if show_mask:
                a = (mask_list[0].data).astype(int)
                b = ((ground_truth.get_mask(threshold=self.thr)[0].resample(dynamic_grid[0]).data)).astype(int)
                print(np.sum((a > b)))
                print(np.sum((a < b)))
                shdom.cloud_plot(a)
                shdom.cloud_plot(b)
        z0_vec = np.arange(1, 2)
        # Define micro-physical parameters: either optimize, keep constant at a specified value or use ground-truth
        if self.args.use_forward_lwc:
            lwc = ground_truth.get_lwc()[:1]#TODO use average
            for ind in range(len(lwc)):
                lwc[ind] = lwc[ind].resample(lwc_grid[ind])
        elif self.args.const_lwc:
            lwc = self.cloud_generator.get_lwc(lwc_grid)

        else:
            rr = []
            lwc_slope = np.linspace(0.5,1.3,8)
            m = mask_list[0].resample(lwc_grid[0])
            for slope in lwc_slope:
                for z0 in z0_vec:
                    rr.append(self.get_monotonous_lwc(m, slope,z0=z0))
            lwc = shdom.DynamicGridDataEstimator(rr,
                                                 min_bound=1e-5,
                                                 max_bound=2.0
                                                 )
            # lwc = shdom.DynamicGridDataEstimator(self.cloud_generator.get_lwc(lwc_grid),
            #                               min_bound=1e-5,
            #                               max_bound=2.0,
            #                               precondition_scale_factor=self.args.lwc_scaling)
            lwc = lwc.dynamic_data


        if self.args.use_forward_reff:
            reff = ground_truth.get_reff()[:1] #TODO use average
            for ind in range(len(reff)):
                reff[ind] = reff[ind].resample(reff_grid[ind])

        elif self.args.const_reff:
            reff = self.cloud_generator.get_reff(reff_grid)
          
        else:
            rr = []
            reff_slope = np.linspace(5, 5, 1)
            r0_vec = np.arange(2, 3)
            m = mask_list[0].resample(reff_grid[0])
            z0_vec = [0]
            for z0 in z0_vec:
                for r0 in r0_vec:
                    for slope in reff_slope:
                            rr.append(self.get_monotonous_reff(m, slope, r0, z0))
            reff = shdom.DynamicGridDataEstimator(rr,
                                                  min_bound=0.01,
                                                  max_bound=35
                                                  )
            # reff = shdom.DynamicGridDataEstimator(self.cloud_generator.get_reff(reff_grid),
            #                                min_bound=0.01,
            #                                max_bound=35,
            #                                precondition_scale_factor=self.args.reff_scaling)
            reff = reff.dynamic_data

        if self.args.use_forward_veff:
            veff = ground_truth.get_veff()[:1] #TODO use average
            for ind in range(len(veff)):
                veff[ind] = veff[ind].resample(veff_grid[ind])
        elif self.args.const_veff:
            veff = self.cloud_generator.get_veff(veff_grid)
        else:
            veff = shdom.DynamicGridDataEstimator(self.cloud_generator.get_veff(veff_grid),
                                           min_bound=0.01,
                                           max_bound=0.3
                                           )
            veff = veff.dynamic_data
        veff = [veff[0]]*num_of_mediums


        for lwc_i in lwc:
            lwc_i.apply_mask(mask_list[0])

        for reff_i in reff:
            reff_i.apply_mask(mask_list[0])

        for veff_i in veff:
            veff_i.apply_mask(mask_list[0])

        len_lwc = len(lwc)
        lwc = lwc*len(reff)
        reff = np.repeat(reff, len_lwc).tolist()
        veff = [veff[0]] * len(reff)
        time_list = [time_list] * len(reff)
        # Define a MicrophysicalScattererEstimator object
        kw_microphysical_scatterer = {"lwc": lwc, "reff": reff, "veff": veff}
        cloud_estimator = shdom.DynamicScattererEstimator(wavelength=wavelength, time_list=time_list, **kw_microphysical_scatterer)
        cloud_estimator.set_mask([mask_list[0]]* len(reff))

        # Create a medium estimator object (optional Rayleigh scattering)

        air = self.air_generator.get_scatterer(cloud_estimator.wavelength)
        medium_estimator = shdom.DynamicMediumEstimator(cloud_estimator, air, [0,0,0], loss_type=self.args.loss_type,

                                                        stokes_weights=self.args.stokes_weights)
        return medium_estimator

    def get_monotonous_reff(self, old_reff, slope, reff0, z0=None):
        mask = old_reff.data > 0
        grid = old_reff.grid
        if z0 is None:
            z0 = grid.z[mask][0]

        Z = grid.z - z0
        Z[Z < 0] = 0
        reff_data = (slope * Z ** (1. / 3.)) + reff0

        # reff_data[Z == 0] = 0
        reff_data[mask == 0] = 0
        return shdom.GridData(grid, reff_data)

    def get_monotonous_lwc(self, old_lwc, slope, z0=0):
        mask = old_lwc.data > 0
        mask_z = np.sum(mask,(0,1))>0
        grid = old_lwc.grid
        z0 = grid.z[mask_z][z0]

        Z = grid.z - z0
        # Z[Z < 0] = -1
        lwc_profile = (slope * Z )
        lwc_profile[lwc_profile>0] += 0.01
        lwc_profile[lwc_profile<0] = 0

        lwc_data = np.tile(lwc_profile[np.newaxis, np.newaxis, :], (grid.nx, grid.ny, 1))

        lwc_data[mask==0] = 0
        return shdom.GridData(grid, lwc_data)

    def load_forward_model(self, input_directory):
        """
        Load the ground-truth medium, rte_solver and measurements which define the forward model

        Parameters
        ----------
        input_directory: str
            The input directory where the forward model is saved

        Returns
        -------
        ground_truth: shdom.OpticalScatterer
            The ground truth scatterer
        rte_solver: shdom.RteSolverArray
            The rte solver with the numerical and scene parameters
        measurements: shdom.Measurements
            The acquired measurements
        """
        # Load forward model and measurements
        dynamic_medium, dynamic_solver, measurements = shdom.load_dynamic_forward_model(input_directory)

        # Get micro-physical medium ground-truth
        ground_truth = dynamic_medium.get_dynamic_scatterer()
        return ground_truth, dynamic_solver, measurements

    def get_summary_writer(self, measurements, ground_truth):
        """
        Define a SummaryWriter object

        Parameters
        ----------
        measurements: shdom.Measurements object
            The acquired measurements.
        ground_truth: shdom.Scatterer
            The ground-truth scatterer for monitoring

        Returns
        -------
        writer: shdom.SummaryWriter object
            A logger for the TensorboardX.
        """
        writer = None
        if self.args.log is not None:
            log_dir = os.path.join(self.args.input_dir, 'logs', self.args.log + '-' + time.strftime("%d-%b-%Y-%H:%M:%S"))
            writer = shdom.DynamicSummaryWriter(log_dir)
            writer.save_checkpoints()
            writer.monitor_loss()
            writer.monitor_images(measurements=measurements)

            # writer.monitor_state()

            # Compare estimator to ground-truth
            writer.monitor_scatterer_error(estimator_name=self.scatterer_name, ground_truth=ground_truth)
            writer.monitor_domain_mean(estimator_name=self.scatterer_name, ground_truth=ground_truth)
            writer.monitor_scatter_plot(estimator_name=self.scatterer_name, ground_truth=ground_truth, dilute_percent=0.8)
            writer.monitor_horizontal_mean(estimator_name=self.scatterer_name, ground_truth=ground_truth, ground_truth_mask=ground_truth.get_mask(threshold=self.thr))

            self.save_args(log_dir)
        return writer

    def get_optimizer(self):
        """
        Define an Optimizer object

        Returns
        -------
        optimizer: shdom.Optimizer object
            An optimizer object.
        """
        self.parse_arguments()
        ground_truth, dynamic_solver, measurements = self.load_forward_model(self.args.input_dir)

        # Initialize a Medium Estimator
        medium_estimator = self.get_medium_estimator(measurements, ground_truth)
        self.args.num_mediums = medium_estimator.num_mediums
        # Initialize a RTESolver
        dynamic_solver = self.get_rte_solver(dynamic_solver)
        dynamic_solver.set_dynamic_medium(medium_estimator)
        measurements = measurements.downsample_viewed_mediums(1)



        # Initialize a LocalOptimizer
        options = {
            'maxiter': self.args.maxiter,
        }
        optimizer = shdom.ParametersOptimizer('L-BFGS-B', options=options, n_jobs=self.args.n_jobs
                                                )



        optimizer.set_measurements(measurements)
        optimizer.set_dynamic_solver(dynamic_solver)
        optimizer.set_medium_estimator(medium_estimator)

        reff_slope, reff0, z0 = optimizer.minimize()
        return optimizer

    def get_rte_solver(self, dynamic_solver):

        num_mediums = self.args.num_mediums
        dynamic_solver_out = shdom.DynamicRteSolver([dynamic_solver._scene_params[0]]*num_mediums,
                                                    [dynamic_solver._numerical_params[0]]*num_mediums)


        return dynamic_solver_out

    def main(self):
        """
        Main optimization script
        """


        local_optimizer = self.get_optimizer()
        t = time.time()
        # Optimization process
        num_global_iter = 1
        if self.args.globalopt:
            global_optimizer = shdom.GlobalOptimizer(local_optimizer=local_optimizer)
            result = global_optimizer.minimize(niter_success=3, T=1e-3)
            num_global_iter = result.nit
            result = result.lowest_optimization_result
            local_optimizer.set_state(result.x)
        else:
            result = local_optimizer.minimize()

        print('\n------------------ Optimization Finished ------------------\n')
        print('Number global iterations: {}'.format(num_global_iter))
        print('Success: {}'.format(result.success))
        print('Message: {}'.format(result.message))
        print('Final loss: {}'.format(result.fun))
        print('Number iterations: {}'.format(result.nit))
        print(time.time() - t)
        # Save optimizer state
        save_dir = local_optimizer.writer.dir if self.args.log is not None else self.args.input_dir
        local_optimizer.save_state(os.path.join(save_dir, 'final_state.ckpt'))

if __name__ == "__main__":
    script = OptimizationScript(scatterer_name='cloud')
    script.main()



