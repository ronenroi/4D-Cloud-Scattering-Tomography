# 4D Cloud Scattering Tomography
[![arXiv](https://img.shields.io/static/v1?label=ICCV2021&message=4DCloudScatteringTomography&color=blueviolet)](https://openaccess.thecvf.com/content/ICCV2021/papers/Ronen_4D_Cloud_Scattering_Tomography_ICCV_2021_paper.pdf)

## Abstract
We derive computed tomography (CT) of a time-varying volumetric scattering object, using a small number of moving cameras. We focus on passive tomography of dynamic clouds, as clouds have a major effect on the Earth's climate. State of the art scattering CT assumes a static object. Existing 4D CT methods rely on a linear image formation model and often on significant priors. In this paper, the angular and temporal sampling rates needed for a proper recovery are discussed. Spatiotemporal CT is achieved using gradient-based optimization, which accounts for the correlation time of the dynamic object content. We demonstrate this in physics-based simulations and on experimental real-world data.
![4DcloudScatteringTomography](4DcloudScatteringTomography.png)

## Description
This repository contains the official implementation of 4D Cloud Scattering Tomography, which is implemented ontop of Pyshdom3.0 package[1].
Our framework recovers 4D cloud microphysics fields, using small number of moving cameras. It relies on the natural temporal evolution of clouds for weighing gradients of different time steps.
sampling needed for a good reconstruction. 


[1]: https://github.com/aviadlevis/pyshdom.html


&nbsp;


## Installation 
Installation using using anaconda package management

Start a clean virtual environment
```
conda create -n pyshdom python=3
source activate pyshdom
```

Install required packages
```
conda install anaconda dill tensorflow tensorboard pillow joblib
```

Install pyshdom distribution with (either install or develop flag)
```
python setup.py install
```

&nbsp;

## Usage
For basic usage follow the following jupyter notebook tutorials
- notebooks/Radiance Rendering [Single Image].ipynb
- notebooks/Radiance Rendering [Multiview].ipynb
- notebooks/Radiance Rendering [Multispectral].ipynb
 
&nbsp;

## Main scripts
For generating rendering and optimization scripts see the list below. 
The scripts folder contains another readme file with examples of how to run each script.
  - scripts/generate_mie_tables.py
  - scripts/render_radiance_toa.py
  - scripts/optimize_extinction_lbfgs.py
  - scripts/optimize_microphysics_lbfgs.py
  
For info about command-line flags of each script use 
```
python script.py --help
```

&nbsp;

## Usage and Contact
If you find this package useful please let me know at aviad.levis@gmail.com, I am interested.
If you use this package in an academic publication please acknowledge the appropriate publications (see LICENSE file). 

