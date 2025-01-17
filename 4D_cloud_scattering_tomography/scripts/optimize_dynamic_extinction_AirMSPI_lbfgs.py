﻿import os, time
import numpy as np
import argparse
import shdom
import scipy.ndimage as sci
import scipy.stats as sci_stat

class OptimizationScript(object):
    """
    Optimize: Extinction
    --------------------
    Estimate the extinction coefficient based on monochrome radiance measurements.
    In this script, the phase function, albedo and rayleigh scattering are assumed known and are not estimated.

    Measurements are simulated measurements using a forward rendering script
    (e.g. scripts/render_radiance_toa.py).

    For example usage see the README.md

    For information about the command line flags see:
      python scripts/optimize_extinction_lbfgs.py --help

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
        parser.add_argument('--reg_const',
                            default=0,
                            type=float,
                            help='(default value: %(default)s) Regularization constant. reg_const=0 uses no regularization')
        parser.add_argument('--n_jobs',
                            default=1,
                            type=int,
                            help='(default value: %(default)s) Number of jobs for parallel rendering. n_jobs=1 uses no parallelization')
        parser.add_argument('--globalopt',
                            action='store_true',
                            help='Global optimization with basin-hopping.'
                                 'For more info see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html')
        parser.add_argument('--maxiter',
                            default=1000,
                            type=int,
                            help='(default value: %(default)s) Maximum number of L-BFGS iterations.'
                                 'For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html')
        parser.add_argument('--maxls',
                            default=30,
                            type=int,
                            help='(default value: %(default)s) Maximum number of line search steps (per iteration).'
                                 'For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html')
        parser.add_argument('--disp',
                            choices=[True, False],
                            default=True,
                            type=np.bool,
                            help='(default value: %(default)s) Display optimization progression.'
                                 'For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html')
        parser.add_argument('--gtol',
                            default=1e-16,
                            type=np.float32,
                            help='(default value: %(default)s) Stop criteria for the maximum projected gradient.'
                                 'For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html')
        parser.add_argument('--ftol',
                            default=1e-16,
                            type=np.float32,
                            help='(default value: %(default)s) Stop criteria for the relative change in loss function.'
                                 'For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html')
        parser.add_argument('--stokes_weights',
                            nargs=4,
                            default=[1.0, 0.0, 0.0, 0.0],
                            type=float,
                            help='(default value: %(default)s) Loss function weights for stokes vector components [I, Q, U, V]')
        parser.add_argument('--loss_type',
                            choices=['l2', 'l2_weighted', 'normcorr'],
                            default='l2',
                            help='Different loss functions for optimization. Currently only l2 is supported.')
        parser.add_argument('--num_mu_bins',
                            default=8,
                            type=np.int,
                            )
        parser.add_argument('--num_phi_bins',
                            default=16,
                            type=np.int,
                            )
        parser.add_argument('--assume_moving_cloud',
                            default=False,
                            action='store_true',
                            help='Use the ground truth cloud velocity.')
        parser.add_argument('--use_cross_validation',
                            default=-1,
                            type=int,
                            help='Reconstruct on base of all the view-points except mentioned,'
                                 ' if negative run without cross validation')
        parser.add_argument('--num_mediums',
                            default=-1,
                            type=int,
                            help='Number of different mediums to be reconstructed,'
                                 ' if negative run without cross validation')
        parser.add_argument('--sigma',
                            default=20,
                            type=float)

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
        parser.add_argument('--use_forward_albedo',
                            action='store_true',
                            help='Use the ground truth albedo.')
        parser.add_argument('--use_forward_phase',
                            action='store_true',
                            help='Use the ground-truth phase reconstruction.')
        parser.add_argument('--radiance_threshold',
                            default=[-0.03],
                            nargs='+',
                            type=np.float32,
                            help='(default value: %(default)s) Threshold for the radiance to create a cloud mask.' \
                            'Threshold is either a scalar or a list of length of measurements.')
        parser.add_argument('--space_carve_agreement',
                            default=0.9,
                            type=np.float32)
        parser.add_argument('--mie_base_path',
                            default='mie_tables/polydisperse/Water_<wavelength>nm.scat',
                            help='(default value: %(default)s) Mie table base file name. ' \
                                 '<wavelength> will be replaced by the corresponding wavelength.')
        parser.add_argument('--surface_type',
                            default='Lambertian',
                            type=str,
                            help='')

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

    def get_medium_estimator(self, measurements):
        """
        """

        num_of_mediums = self.args.num_mediums
        cv_index = self.args.use_cross_validation
        time_list = measurements.time_list
        images_weight = None
        if cv_index >= 0:
            cv_time = time_list[cv_index]
            time_list = np.delete(time_list, cv_index)
            if self.args.loss_type == 'l2_weighted':
                delta_time = np.abs(np.array(time_list) - cv_time)
                images_weight = 1 / delta_time
                images_weight /= np.sum(images_weight)
                images_weight += 1
                images_weight /= np.sum(images_weight)
                images_weight *= images_weight.shape

        assert isinstance(num_of_mediums, int) and num_of_mediums <= len(time_list)
        time_list = np.mean(np.split(time_list, num_of_mediums), 1)

        wavelength = measurements.wavelength
        if not isinstance(wavelength,list):
            wavelength = [wavelength]

        # Define the grid for reconstruction
        grid = albedo_grid = phase_grid = shdom.Grid(bounding_box=measurements.bb,nx=self.args.nx,ny=self.args.ny,nz=self.args.nz)
        mask_grid = shdom.Grid(bounding_box=measurements.bb,nx=np.round(self.args.nx/2).astype(np.int),ny=np.round(self.args.ny/2).astype(np.int)
                               ,nz=np.round(self.args.nz/2).astype(np.int))
        if self.args.assume_moving_cloud:
            cloud_velocity = None
        else:
            cloud_velocity = [0,0,0]


        # Find a cloud mask for non-cloudy grid points
        dynamic_carver = shdom.DynamicSpaceCarver(measurements)
        mask_list, dynamic_grid, cloud_velocity = dynamic_carver.carve(mask_grid, agreement=self.args.space_carve_agreement,
                            time_list = measurements.time_list, thresholds=self.args.radiance_threshold,
                            vx_max = 0, vy_max=0, gt_velocity = cloud_velocity,verbose=False)
        mask = mask_list[0]
        mask.resample(grid)
        show_mask=1
        if show_mask:
            a = mask.data.astype(int)
            shdom.cloud_plot(a)
            print(cloud_velocity)
            print(sum(sum(sum(a))))
        table_path = self.args.mie_base_path.replace('<wavelength>', '{}'.format(shdom.int_round(wavelength[0])))
        self.cloud_generator.add_mie(table_path)
        albedo = self.cloud_generator.get_albedo(wavelength[0], [albedo_grid] * num_of_mediums)
        phase = self.cloud_generator.get_phase(wavelength[0], [phase_grid] * num_of_mediums)


        # grid.xmin
        extinction = shdom.DynamicGridDataEstimator(self.cloud_generator.get_extinction(wavelength[0], [grid] * num_of_mediums),
                                                    min_bound=1e-5,
                                                    max_bound=2e2)
        kw_optical_scatterer = {"extinction": extinction, "albedo": albedo, "phase": phase}
        cloud_estimator = shdom.DynamicScattererEstimator(wavelength=wavelength, time_list=time_list, **kw_optical_scatterer)
        cloud_estimator.set_mask([mask] * num_of_mediums)

        # Create a medium estimator object (optional Rayleigh scattering)

        air = self.air_generator.get_scatterer(wavelength)
        medium_estimator = shdom.DynamicMediumEstimator(cloud_estimator, air, cloud_velocity,images_weight=images_weight, loss_type=self.args.loss_type,sigma=self.args.sigma)
        return medium_estimator

    def get_summary_writer(self, measurements):
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
            writer.save_checkpoints(ckpt_period=-1)
            writer.monitor_loss()
            writer.monitor_shdom_iterations()
            writer.monitor_images(measurements=measurements, ckpt_period=-1)
            writer.monitor_save3d(ckpt_period=-1)
            writer.monitor_images_scatter_plot(measurements=measurements)
            # save parse_arguments
            self.save_args(log_dir)
        return writer

    def save_args(self,log_dir):
        text_file = open(log_dir+"/Input_args.txt", "w")
        for data in self.args.__dict__:
            text_file.write("{} : {}\n".format(data, self.args.__dict__[data]))
        text_file.close()

    def get_optimizer(self):
        """
        Define an Optimizer object

        Returns
        -------
        optimizer: shdom.Optimizer object
            An optimizer object.
        """
        self.parse_arguments()

        measurements = shdom.AirMSPIDynamicMeasurements()
        measurements.load_airmspi_measurements(self.args.input_dir)
        cv_index = self.args.use_cross_validation
        if self.args.num_mediums < 0:
            self.args.num_mediums = len(measurements.time_list)
            if cv_index >= 0:
                self.args.num_mediums -= 1



        # Initialize a Medium Estimator
        medium_estimator = self.get_medium_estimator(measurements)

        # Initialize a RTESolver
        dynamic_solver, cv_rte_solver = self.get_rte_solver(measurements, medium_estimator)

        if cv_index >= 0:
            cv_measurement, measurements = measurements.get_cross_validation_measurements(cv_index)
        measurements = measurements.downsample_viewed_mediums(self.args.num_mediums)

        # Initialize TensorboardX logger
        writer = self.get_summary_writer(measurements)
        if cv_index >= 0:
            writer.monitor_cross_validation(cv_measurement=cv_measurement, ckpt_period=-1)

        # Initialize a LocalOptimizer
        options = {
            'maxiter': self.args.maxiter,
            'maxls': self.args.maxls,
            'disp': self.args.disp,
            'gtol': self.args.gtol,
            'ftol': self.args.ftol,
        }
        optimizer = shdom.DynamicLocalOptimizer('L-BFGS-B', options=options, n_jobs=self.args.n_jobs,
                                                regularization_const=self.args.reg_const)
        optimizer.set_measurements(measurements)
        optimizer.set_dynamic_solver(dynamic_solver)
        optimizer.set_medium_estimator(medium_estimator)
        optimizer.set_writer(writer)
        if cv_index >= 0:
            optimizer.set_cross_validation_param(cv_rte_solver, cv_measurement, cv_index)
        # Reload previous state
        if self.args.reload_path is not None:
            optimizer.load_state(self.args.reload_path)
        return optimizer

    def get_rte_solver(self,measurements, medium_estimator):
        wavelengths = measurements.wavelength
        if not isinstance(wavelengths,list):
            wavelengths = [wavelengths]
        sun_azimuth_list = measurements.sun_azimuth_list
        sun_zenith_list = measurements.sun_zenith_list
        if self.args.surface_type == 'Lambertian':
            # measurements.calc_albedo(n_jobs=self.args.n_jobs)
            surface_params = measurements.est_albedo_list
            surface_params = [0.04]*len(sun_zenith_list)
        elif self.args.surface_type == 'Ocean':
            # measurements.calc_wind(n_jobs=self.args.n_jobs)
            surface_params = measurements.est_wind_list
            surface_params = [20] * len(sun_zenith_list)
        else:
            assert False, 'unknown surface type'

        cv_index = self.args.use_cross_validation
        num_of_mediums = self.args.num_mediums

        cv_dynamic_solver = None
        if cv_index >= 0:
            cv_sun_azimuth = sun_azimuth_list[cv_index]
            cv_sun_zenith = sun_zenith_list[cv_index]
            cv_param = surface_params[cv_index]
            sun_azimuth_list = np.delete(sun_azimuth_list, cv_index)
            sun_zenith_list = np.delete(sun_zenith_list, cv_index)
            surface_params = np.delete(surface_params, cv_index)
        sun_azimuth_list = sci_stat.circmean(np.split(np.array(sun_azimuth_list), num_of_mediums), axis=1,high=360)
        sun_zenith_list = sci_stat.circmean(np.split(np.array(sun_zenith_list), num_of_mediums), axis=1,high=360)
        surface_params = np.mean(np.split(np.array(surface_params), num_of_mediums), 1)

        wl_scene_params_list =[]
        wl_numerical_params_list =[]
        for sun_azimuth, sun_zenith, surface_param in zip(sun_azimuth_list, sun_zenith_list, surface_params):
            if self.args.surface_type == 'Lambertian':
                surface = shdom.LambertianSurface(albedo=surface_param)
            elif self.args.surface_type == 'Ocean':
                surface = shdom.OceanSurface(wind_speed=surface_param)
            else:
                assert False, 'unknown surface type'
            scene_params_list = []
            numerical_params_list = []
            for wavelength in wavelengths:
                scene_params = shdom.SceneParameters(
                    wavelength=wavelength,
                    surface=surface,
                    source=shdom.SolarSource(azimuth=sun_azimuth, zenith=sun_zenith)
                )
                scene_params_list.append(scene_params)
                numerical_params = shdom.NumericalParameters(num_mu_bins=self.args.num_mu_bins, num_phi_bins=self.args.num_phi_bins)
                numerical_params_list.append(numerical_params)
            wl_scene_params_list.append(scene_params_list)
            wl_numerical_params_list.append(numerical_params_list)
        dynamic_solver = shdom.DynamicRteSolver(scene_params=wl_scene_params_list, numerical_params=wl_numerical_params_list)
        dynamic_solver.set_dynamic_medium(medium_estimator)
        if cv_index >= 0:
            wl_scene_params_list = []
            wl_numerical_params_list = []
            scene_params_list = []
            numerical_params_list = []
            if self.args.surface_type == 'Lambertian':
                surface = shdom.LambertianSurface(albedo=cv_param)
            elif self.args.surface_type == 'Ocean':
                surface = shdom.OceanSurface(wind_speed=cv_param)
            else:
                assert False, 'unknown surface type'
            for wavelength in wavelengths:
                scene_params = shdom.SceneParameters(
                    wavelength=wavelength,
                    surface=surface,
                    source=shdom.SolarSource(azimuth=cv_sun_azimuth, zenith=cv_sun_zenith)
                )
                scene_params_list.append(scene_params)
                numerical_params = shdom.NumericalParameters(num_mu_bins=self.args.num_mu_bins, num_phi_bins=self.args.num_phi_bins)
                numerical_params_list.append(numerical_params)
            wl_scene_params_list.append(scene_params_list)
            wl_numerical_params_list.append(numerical_params_list)
            cv_dynamic_solver = shdom.DynamicRteSolver(scene_params=wl_scene_params_list,
                                                    numerical_params=wl_numerical_params_list)

        return dynamic_solver, cv_dynamic_solver


    def main(self):
        """
        Main optimization script
        """
        local_optimizer = self.get_optimizer()

        # Optimization process
        num_global_iter = 1
        if self.args.globalopt:
            global_optimizer = shdom.GlobalOptimizer(local_optimizer=local_optimizer)
            result = global_optimizer.minimize(niter_success=20, T=1e-3)
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

        # Save optimizer state
        save_dir = local_optimizer.writer.dir if self.args.log is not None else self.args.input_dir
        local_optimizer.save_state(os.path.join(save_dir, 'final_state.ckpt'))


if __name__ == "__main__":
    script = OptimizationScript(scatterer_name='cloud')
    script.main()




