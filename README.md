# SHARPy
### Sequential HAmiltonian Riemann montecarlo Python sampler for <em>cutting-edge</em> Gravitational Wave inference
© 2025-  Gabriele Demasi, Walter Del Pozzo

### Installation 
For avoiding package incompatibilities, create a dedicated conda environment  
`conda create --name sharpy_env python==3.11` <br> `conda activate sharpy_env`

or if conda is not available

`python -m venv /path_to/sharpy_env`<br>
`source /path_to/sharpy_env/bin/activate`

then install SHARPy:<br>
`git clone git@github.com:gabrieledemasi/SHARPy-GW.git` <br>
`cd SHARPy-GW`<br>
`pip install .`

To run on GPU please install this version of JAX:
`pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`

## Examples
This [notebook](Examples/11D_gaussian_mixture.ipynb) contains a simple test case( a 11D bimodal gaussian mixture) that can be run on a laptop. The run time should be around 1 min, depending on the hardware.

## Acknowledgements
-
