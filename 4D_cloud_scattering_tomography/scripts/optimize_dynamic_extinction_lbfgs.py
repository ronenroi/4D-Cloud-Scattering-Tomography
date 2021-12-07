import os, time
import numpy as np
import argparse
import shdom


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
        parser.add_argument('--use_forward_grid',
                            action='store_true',
                            help='Use the same grid for the reconstruction. This is a sort of inverse crime which is \
                                  usefull for debugging/development.')
        parser.add_argument('--use_forward_mask',
                            action='store_true',
                            help='Use the ground-truth cloud mask. This is an inverse crime which is \
                                  usefull for debugging/development.')
        parser.add_argument('--reg_const',
                            default=0,
                            type=float,
                            help='(default value: %(default)s) Regularization constant. reg_const=0 uses no regularization')
        parser.add_argument('--add_noise',
                            action='store_true',
                            help='currently only supports AirMSPI noise model. \
                                  See shdom.AirMSPINoise object for more info.')
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
                            choices=['l2', 'normcorr', 'l2_weight'],
                            default='l2',
                            help='Different loss functions for optimization. Currently only l2 is supported.')
        parser.add_argument('--use_forward_cloud_velocity',
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
                            default=[0.03],
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
        parser.add_argument('--sigma',
                            default=20,
                            type=float)

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

    def get_medium_estimator(self, measurements, ground_truth):
        """
        """
        num_of_mediums = self.args.num_mediums
        cv_index = self.args.use_cross_validation
        time_list = measurements.time_list
        temporary_scatterer_list = ground_truth._temporary_scatterer_list

        if cv_index >= 0:
            cv_time = time_list[cv_index]
            time_list = np.delete(time_list, cv_index)
            temporary_scatterer_list = np.delete(temporary_scatterer_list, cv_index)
            if self.args.loss_type == 'l2_weighted':
                delta_time = np.abs(np.array(time_list) - cv_time)
                images_weight = 1 / delta_time
                images_weight /= np.sum(images_weight)
                images_weight += 1
                images_weight /= np.sum(images_weight)
                images_weight *= images_weight.shape

        assert isinstance(num_of_mediums, int) and num_of_mediums <= len(time_list)
        time_list = np.mean(np.split(np.array(time_list), num_of_mediums), 1)

        wavelength = ground_truth.wavelength
        if not isinstance(wavelength, list):
            wavelength = [wavelength]
        # Define the grid for reconstruction
        if self.args.use_forward_grid:
            extinction_grid = []
            albedo_grid = []
            phase_grid = []
            ext_grid = []
            alb_grid = []
            pha_grid = []
            for temporary_scatterer in temporary_scatterer_list:
                ext_grid.append(temporary_scatterer.scatterer.extinction.grid)
                alb_grid.append(temporary_scatterer.scatterer.albedo.grid)
                pha_grid.append(temporary_scatterer.scatterer.phase.grid)

            ext_grid = np.split(np.array(ext_grid), num_of_mediums)
            alb_grid = np.split(np.array(alb_grid), num_of_mediums)
            pha_grid = np.split(np.array(pha_grid), num_of_mediums)
            for ext, alb, pha in zip(ext_grid, alb_grid, pha_grid):
                extinction_grid.append(np.sum(ext))
                albedo_grid.append(np.sum(alb))
                phase_grid.append(np.sum(pha))

            grid = extinction_grid[0]

            # grid = shdom.Grid(x = grid.x - grid.xmin, y = grid.y - grid.ymin, z = grid.z)
        else:
            # grid = self.cloud_generator.get_grid()
            old_grid = temporary_scatterer_list[0].scatterer.extinction.grid
            grid = shdom.Grid(x=np.linspace(old_grid.xmin,old_grid.xmax,self.args.nx),
                              y=np.linspace(old_grid.ymin,old_grid.ymax,self.args.ny),
                              z=np.linspace(old_grid.zmin,old_grid.zmax,self.args.nz))
            # grid = shdom.Grid(x=grid.x + temporary_scatterer_list[0].scatterer.extinction.grid.xmin, y=grid.y + temporary_scatterer_list[0].scatterer.extinction.grid.ymin, z=grid.z)
            extinction_grid = albedo_grid = phase_grid = [grid]*num_of_mediums



        if self.args.use_forward_cloud_velocity:
            cloud_velocity = ground_truth.get_velocity()
            if len(cloud_velocity)>0:
                cloud_velocity = cloud_velocity[0]*1000 #km/sec to m/sec
            else:
                cloud_velocity = [0,0,0]
        else:
            cloud_velocity = None

        # Find a cloud mask for non-cloudy grid points
        if self.args.use_forward_mask:
            mask_list = ground_truth.get_mask(threshold=0.000001)
            a = (mask_list[0].data).astype(int)
            shdom.cloud_plot(a)
        else:
            dynamic_carver = shdom.DynamicSpaceCarver(measurements)
            mask_list, ext_grid, cloud_velocity = dynamic_carver.carve(grid, agreement=self.args.space_carve_agreement,
                                time_list = measurements.time_list, thresholds=self.args.radiance_threshold,
                                vx_max = 0, vy_max=0, gt_velocity = cloud_velocity,verbose=False)
            ext_grid = np.split(np.array(ext_grid), num_of_mediums)
            extinction_grid =[]
            for ext in ext_grid:
                extinction_grid.append(np.sum(ext))
            show_mask=1
            if show_mask:
                a = (mask_list[0].data).astype(int)
                b = ((ground_truth.get_mask(threshold=0.0000001)[0].resample(extinction_grid[0]).data)).astype(int)
                print(np.sum((a > b)))
                print(np.sum((a < b))/np.sum(b))
                shdom.cloud_plot(a*2+b)
                shdom.cloud_plot(b)

        table_path = self.args.mie_base_path.replace('<wavelength>', '{}'.format(shdom.int_round(wavelength[0])))
        self.cloud_generator.add_mie(table_path)
        if self.args.use_forward_albedo:
            albedo = ground_truth.get_albedo()[:num_of_mediums]
            for ind in range(len(albedo)):
                albedo[ind] = albedo[ind].resample(grid)
        else:

            albedo = self.cloud_generator.get_albedo(wavelength[0], albedo_grid)

        if self.args.use_forward_phase:
            phase = ground_truth.get_phase()[:num_of_mediums]
            for ind in range(len(phase)):
                phase[ind] = phase[ind].resample(grid)
        else:
            phase = self.cloud_generator.get_phase(wavelength[0], phase_grid)

        if cv_index >= 0:
            del extinction_grid[cv_index]
            del mask_list[cv_index]
            del albedo[cv_index]
            del phase[cv_index]
        extinction = shdom.DynamicGridDataEstimator(self.cloud_generator.get_extinction(measurements.wavelength, extinction_grid),
                                                    min_bound=1e-3,
                                                    max_bound=2e2)


        kw_optical_scatterer = {"extinction": extinction, "albedo": albedo, "phase": phase}
        cloud_estimator = shdom.DynamicScattererEstimator(wavelength=wavelength, time_list=time_list, **kw_optical_scatterer)
        cloud_estimator.set_mask(mask_list)

        # Create a medium estimator object (optional Rayleigh scattering)
        air = self.air_generator.get_scatterer(wavelength)
        medium_estimator = shdom.DynamicMediumEstimator(cloud_estimator, air, cloud_velocity,sigma=self.args.sigma, regularization_const=self.args.reg_const)

        return medium_estimator

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
            writer.save_checkpoints(ckpt_period=20 * 60)
            writer.monitor_loss()
            writer.monitor_shdom_iterations()
            writer.monitor_time_smoothness()

            writer.monitor_save3d(ckpt_period=-1)
            writer.monitor_images(measurements=measurements, ckpt_period=1 * 60)

            # Compare estimator to ground-truth
            writer.monitor_scatterer_error(estimator_name=self.scatterer_name, ground_truth=ground_truth)
            writer.monitor_domain_mean(estimator_name=self.scatterer_name, ground_truth=ground_truth)
            writer.monitor_scatter_plot(estimator_name=self.scatterer_name, ground_truth=ground_truth, dilute_percent=0.8)
            writer.monitor_horizontal_mean(estimator_name=self.scatterer_name, ground_truth=ground_truth, ground_truth_mask=ground_truth.get_mask(threshold=0.000001))

            # save parse_arguments
            self.save_args(log_dir)
        return writer

    def save_args(self,log_dir):
        text_file = open(log_dir+"/Input_args.txt", "w")
        for data in self.args.__dict__:
            text_file.write("{} : {}\n".format(data, self.args.__dict__[data]))
        text_file.close()

    def load_forward_model(self, input_directory):
        """
        Load the ground-truth medium, rte_solver and measurements which define the forward model

        Parameters
        ----------
        input_directory: str
            The input directory where the forward model is saved

        Returns
        -------
        ground_truth: shdom.DynamicScatterer
            The ground truth scatterer
        rte_solver: shdom.RteSolverArray
            The rte solver with the numerical and scene parameters
        measurements: shdom.Measurements
            The acquired measurements
        """
        # Load forward model and measurements
        dynamic_medium, dynamic_solver, measurements = shdom.load_dynamic_forward_model(input_directory)

        # Get optical medium ground-truth
        dynamic_scatterer = dynamic_medium.get_dynamic_scatterer()
        if dynamic_scatterer.type == 'MicrophysicalScatterer':
            ground_truth = dynamic_scatterer.get_dynamic_optical_scatterer(dynamic_medium.wavelength)
        else:
            ground_truth=dynamic_scatterer
        return ground_truth, dynamic_solver, measurements

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
        import scipy.io as sio
        gt_ext =[]
        # for i in ground_truth._temporary_scatterer_list:
        #     gt_ext.append(i.scatterer.extinction.data)
        # sio.savemat(os.path.join(self.args.input_dir, 'GT_3D_{}.mat'.format('extinction')),
        #             {'gt_extinction': gt_ext})

        cv_index = self.args.use_cross_validation
        if self.args.num_mediums < 0:
            self.args.num_mediums = len(measurements.time_list)
            if cv_index >= 0:
                self.args.num_mediums -= 1
        # Add noise (currently only supports AirMSPI noise model)
        if self.args.add_noise:
            assert measurements.noise is None, 'The images were already noised'
            measurements.set_noise(shdom.AirMSPINoise())

        # Initialize a Medium Estimator
        medium_estimator = self.get_medium_estimator(measurements, ground_truth)


        # Initialize a RTESolver
        dynamic_solver, cv_rte_solver = self.get_rte_solver(dynamic_solver)
        dynamic_solver.set_dynamic_medium(medium_estimator)

        if cv_index >= 0:
            cv_ground_truth = ground_truth.pop_temporary_scatterer(cv_index)
            cv_measurement, measurements = measurements.get_cross_validation_measurements(cv_index)
        measurements = measurements.downsample_viewed_mediums(self.args.num_mediums)


        # Initialize TensorboardX logger
        writer = self.get_summary_writer(measurements, ground_truth)
        if cv_index >= 0:
            writer.monitor_cross_validation(cv_measurement=cv_measurement, ckpt_period=-1)
            writer.monitor_cross_validation_scatterer_error(estimator_name=self.scatterer_name, cv_ground_truth=cv_ground_truth)
            writer.monitor_cross_validation_scatter_plot(estimator_name=self.scatterer_name, cv_ground_truth=cv_ground_truth, dilute_percent=0.8)

        # Initialize a LocalOptimizer
        options = {
            'maxiter': self.args.maxiter,
            'maxls': self.args.maxls,
            'disp': self.args.disp,
            'gtol': self.args.gtol,
            'ftol': self.args.ftol,
        }
        optimizer = shdom.DynamicLocalOptimizer('L-BFGS-B', options=options, n_jobs=self.args.n_jobs,
                                                regularization_const=self.args.reg_const
                                                )



        optimizer.set_measurements(measurements)
        optimizer.set_dynamic_solver(dynamic_solver)
        optimizer.set_medium_estimator(medium_estimator)
        optimizer.set_writer(writer)
        if cv_index >= 0:
            optimizer.set_cross_validation_param(cv_rte_solver, cv_measurement, cv_index)
        # Reload previous state
        if self.args.reload_path is not None:
            import scipy.io as sio

            optimizer.load_state(self.args.reload_path)
            est_l =[]
            est_r =[]
            gt_r = []
            gt_l = []
            for m,t in zip(medium_estimator.medium_list, ground_truth._temporary_scatterer_list):
                c = m.scatterers['cloud']
                est_r.append( c.reff.data)
                est_l.append(c.lwc.data)
                gt_r.append(t.scatterer.reff.data)
                gt_l.append(t.scatterer.lwc.data)

            sio.savemat(os.path.join(self.args.input_dir, 'est_r.mat'),
                        {'est_r': est_r})
            sio.savemat(os.path.join(self.args.input_dir, 'est_l.mat'),
                        {'est_l': est_l})
            sio.savemat(os.path.join(self.args.input_dir, 'gt_r.mat'),
                        {'gt_r': gt_r})
        # import scipy.io as sio
        # gt_r = []
        # gt_l = []
        # for m, t in zip(medium_estimator.medium_list, ground_truth._temporary_scatterer_list):
        #     c = m.scatterers['cloud']
        #     gt_r.append(t.scatterer.reff.data)
        #     gt_l.append(t.scatterer.lwc.data)
        #
        # sio.savemat(os.path.join(self.args.input_dir, 'gt_r.mat'),
        #             {'gt_r': gt_r})
        # sio.savemat(os.path.join(self.args.input_dir, 'gt_l.mat'),
        #             {'gt_l': gt_l})
        return optimizer

    def get_rte_solver(self, dynamic_solver):

        num_mediums = self.args.num_mediums
        dynamic_solver_out = shdom.DynamicRteSolver(dynamic_solver._scene_params[:num_mediums],
                                                    dynamic_solver._numerical_params[:num_mediums])

        cv_index = self.args.use_cross_validation
        cv_dynamic_solver = None
        if cv_index >= 0:
            cv_dynamic_solver = shdom.DynamicRteSolver(dynamic_solver._scene_params[cv_index],
                                                        dynamic_solver._numerical_params[cv_index])
        return dynamic_solver_out, cv_dynamic_solver

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




