# SHARPy
## Sequential HAmiltonian Riemann montecarlo Python sampler for <em>cutting-edge</em> Gravitational Wave parameter estimation 

### Installation 
For avoiding package incompatibilities, create a dedicated conda environment  
`conda create --name sharpy_env python==3.11` <br> `conda activate sharpy_env`

or if conda is not available

`python y -m venv /path_to/sharpy_env`<br>
`/path_to/sharpy_env/bin/activate`

then install SHARPy:<br>
`git clone git@github.com:gabrieledemasi/SHARPy-GW.git` <br>
`cd SHARPy-GW`<br>
`pip install .`

To run on GPU please install this version of JAX:
`pip install jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt`
