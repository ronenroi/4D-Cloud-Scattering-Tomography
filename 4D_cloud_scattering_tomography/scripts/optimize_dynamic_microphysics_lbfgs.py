import os, time
import numpy as np
import shdom
from optimize_dynamic_extinction_lbfgs import OptimizationScript as ExtinctionOptimizationScript
import sys
from os.path import dirname
sys.path.append(dirname(__file__))

class OptimizationScript(ExtinctionOptimizationScript):
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
        super().__init__(scatterer_name)

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
        parser.add_argument('--one_dim_reff',
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
        parser.add_argument('--lwc_scaling',
                            default=10,
                            type=np.float32,
                            help='(default value: %(default)s) Pre-conditioning scale factor for liquid water content estimation')
        parser.add_argument('--reff_scaling',
                            default=0.1,
                            type=np.float32,
                            help='(default value: %(default)s) Pre-conditioning scale factor for effective radius estimation')
        parser.add_argument('--veff_scaling',
                            default=1.0,
                            type=np.float32,
                            help='(default value: %(default)s) Pre-conditioning scale factor for effective variance estimation')
        parser.add_argument('--sigma',
                            # default=[20, np.inf])
                            default=20,
                            type=np.float32)

        return parser

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
        if self.args.one_dim_reff:
            for i, grid in enumerate(reff_grid):
                reff_grid[i] = shdom.Grid(z=grid.z)

        if self.args.use_forward_cloud_velocity:
            if ground_truth.num_scatterers > 1:
                cloud_velocity = ground_truth.get_velocity()[0]
            else:
                cloud_velocity = [0,0,0]
            cloud_velocity = np.array(cloud_velocity)*1000 #km/sec to m/sec
        else:
            cloud_velocity = None

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
                                                                           gt_velocity=cloud_velocity,
                                                                           verbose = False)
            show_mask = 1
            if show_mask:
                a = (mask_list[0].data).astype(int)
                b = ((ground_truth.get_mask(threshold=self.thr)[0].resample(dynamic_grid[0]).data)).astype(int)
                print(np.sum((a > b)))
                print(np.sum((a < b)))
                shdom.cloud_plot(a)
                shdom.cloud_plot(b)

        # Define micro-physical parameters: either optimize, keep constant at a specified value or use ground-truth
        if self.args.use_forward_lwc:
            lwc = ground_truth.get_lwc()[:num_of_mediums]#TODO use average
            for ind in range(len(lwc)):
                lwc[ind] = lwc[ind].resample(lwc_grid[ind])
        elif self.args.const_lwc:
            lwc = self.cloud_generator.get_lwc(lwc_grid)

        else:
            rr = []
            for m, r in zip(mask_list, lwc_grid):
                rr.append(self.get_monotonous_lwc(m.resample(r), 0.3))
            lwc = shdom.DynamicGridDataEstimator(rr,
                                          min_bound=1e-5,
                                          max_bound=2.0,
                                          precondition_scale_factor=self.args.lwc_scaling)
            # lwc = shdom.DynamicGridDataEstimator(self.cloud_generator.get_lwc(lwc_grid),
            #                               min_bound=1e-5,
            #                               max_bound=2.0,
            #                               precondition_scale_factor=self.args.lwc_scaling)
            lwc = lwc.dynamic_data


        if self.args.use_forward_reff:
            reff = ground_truth.get_reff()[:num_of_mediums] #TODO use average
            for ind in range(len(reff)):
                reff[ind] = reff[ind].resample(reff_grid[ind])

        elif self.args.const_reff:
            # reff = self.cloud_generator.get_reff(reff_grid)
            reff = []
            for m, r in zip(mask_list, reff_grid):
                reff.append(self.get_monotonous_reff(m.resample(r), 6.67, 3)) #cloud1
                # reff.append(self.get_monotonous_reff(m.resample(r), 7.71, 4, z0=reff_grid[0].z[18])) #cloud2

        else:
            rr=[]
            for m,r in zip(mask_list,reff_grid):
                rr.append (self.get_monotonous_reff(m.resample(r), 6.67, 3))#cloud1
                # rr.append (self.get_monotonous_reff(m.resample(r), 7.71, 4, z0=reff_grid[0].z[18]))#cloud2

            reff = shdom.DynamicGridDataEstimator(rr,
                                           min_bound=0.01,
                                           max_bound=35,
                                           precondition_scale_factor=self.args.reff_scaling)
            # reff = shdom.DynamicGridDataEstimator(self.cloud_generator.get_reff(reff_grid),
            #                                min_bound=0.01,
            #                                max_bound=35,
            #                                precondition_scale_factor=self.args.reff_scaling)
            reff = reff.dynamic_data

        if self.args.use_forward_veff:
            veff = ground_truth.get_veff()[:num_of_mediums] #TODO use average
            for ind in range(len(veff)):
                veff[ind] = veff[ind].resample(veff_grid[ind])
        elif self.args.const_veff:
            veff = self.cloud_generator.get_veff(veff_grid)
        else:
            veff = shdom.DynamicGridDataEstimator(self.cloud_generator.get_veff(veff_grid),
                                           min_bound=0.01,
                                           max_bound=0.3,
                                           precondition_scale_factor=self.args.veff_scaling)
            veff = veff.dynamic_data

        if cv_index >= 0:
            del lwc_grid[cv_index]
            del mask_list[cv_index]
            del reff[cv_index]
            del veff[cv_index]

        for lwc_i, reff_i, veff_i, mask in zip(lwc, reff, veff, mask_list):
            lwc_i.apply_mask(mask)
            reff_i.apply_mask(mask)
            veff_i.apply_mask(mask)

        # for l in lwc:
        #     l._data=(self.get_monotonous_lwc(mask_list[0], 0.3)).data
        # for r in reff:
        #     r._data = (self.get_monotonous_reff(r,6.67,3))._data
        #     a=r.data
        # r = r.resample(l.grid)
        # import matplotlib.pyplot as plt
        # shdom.cloud_plot(lwc_i.data)
        # plt.plot(reff_i.data)
        # plt.show()
        # Define a MicrophysicalScattererEstimator object
        kw_microphysical_scatterer = {"lwc": lwc, "reff": reff, "veff": veff}
        cloud_estimator = shdom.DynamicScattererEstimator(wavelength=wavelength, time_list=time_list, **kw_microphysical_scatterer)
        cloud_estimator.set_mask(mask_list)

        # Create a medium estimator object (optional Rayleigh scattering)

        air = self.air_generator.get_scatterer(cloud_estimator.wavelength)
        # sigma = {"lwc": self.args.sigma[0], 'reff': self.args.sigma[1]}
        sigma = {"lwc": self.args.sigma, 'reff': np.inf}
        medium_estimator = shdom.DynamicMediumEstimator(cloud_estimator, air, cloud_velocity,sigma=sigma, loss_type=self.args.loss_type,
                                                        stokes_weights=self.args.stokes_weights, regularization_const=self.args.reg_const)
        return medium_estimator

    def get_monotonous_reff(self, old_reff, slope, reff0, z0=None):
        mask = old_reff.data > 0
        grid = old_reff.grid
        if z0 is None:
            z0 = grid.z[mask][0]

        Z = grid.z - z0
        Z[Z < 0] = 0
        reff_data = (slope * Z ** (1. / 3.)) + reff0

        reff_data[Z == 0] = 0
        reff_data[mask == 0] = 0
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


if __name__ == "__main__":
    script = OptimizationScript(scatterer_name='cloud')
    script.main()



