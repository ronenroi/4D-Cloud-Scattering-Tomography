import numpy as np
import shdom
from scipy import io
import mayavi.mlab as mlab
def viz3D(cloud_field=None):
    if cloud_field is None:
        data = io.loadmat('../../log_name-18-Oct-2020-17:56:32_sim_eshkol_reg_weighted_extinction.mat')
        cloud_field = data['estimated_extinction'][:,:,:,8]
        x = data['x'][:, 0]
        y = data['y'][:, 0]
        z = data['z'][:, 0]
    nx, ny, nz = cloud_field.shape
    dx, dy, dz = (1, 1, 1)

    xgrid = np.linspace(0, nx - 1, nx)
    ygrid = np.linspace(0, ny - 1, ny)
    zgrid = np.linspace(0, nz - 1, nz)
    X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')
    figh = mlab.gcf()
    src = mlab.pipeline.scalar_field(X, Y, Z, cloud_field)
    src.spacing = [dx, dy, dz]
    src.update_image_data = True
    max_val = cloud_field.max()
    max_val = 100
    isosurface = mlab.pipeline.iso_surface(src, contours=[0.1 * max_val, \
                                                          0.2 * max_val, \
                                                          0.3 * max_val, \
                                                          0.4 * max_val, \
                                                          0.5 * max_val, \
                                                          0.6 * max_val, \
                                                          0.7 * max_val, \
                                                          0.8 * max_val, \
                                                          0.9 * max_val, \
                                                          ], opacity=0.9)
    mlab.pipeline.volume(isosurface, figure=figh)
    color_bar = mlab.colorbar(title="Extinction [1/km]", orientation='vertical', nb_labels=5)
    color_bar.data_range= (0,100)
    mlab.outline(figure=figh, color=(1, 1, 1))  # box around data axes
    mlab.orientation_axes(figure=figh)
    mlab.axes(figure=figh, xlabel="x (km)", ylabel="y (km)", zlabel="z (km)")
    mlab.show()
if __name__ == "__main__":
    input_directory = '../experiments/WIZ_highres_wl_[0.66]_vel_[0.,0.,0.]_img_9_projection_perspective/dynamic_medium_monochromatic/ground_truth_dynamic_medium'

    dynamic_medium, dynamic_solver, measurements = shdom.load_dynamic_forward_model(input_directory)
    script = viz3D()
