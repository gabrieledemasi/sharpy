# SHARPy
Sequential HAmiltonian Riemann montecarlo Python sampler for <em>cutting-edge</em> Gravitational Wave parameter estimation 


For avoiding package incompatibility, install SHARPy in a dedicated conda environment 
conda create --name sharpy_env python==3.11
conda activate sharpy_env

or if conda is not available

python y -m venv /path_to/sharpy_env
/path_to/sharpy_env/bin/activate


To run on GPU please install this version of JAX:
pip install  jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt


