prism_api
====
Prism does not work depending on the architecture of the host machine (e.g. M1 Mac). 

Therefore, we construct a Web API of prism using FastAPI.

The environment is containerized by Docker so that prism can be used regardless of the host machine environment.

Directory strunture:
- `app/` : FastAPI codes
- `prism/` : prism files (will be installed when container started)
- `Dockerfile` : Dockerfile of prism_server (base image: ubuntu)
- `install_launch_prism.sh` : shell script for install prism and launch FastAPI server (via `uvicorn`)

When this container is launched by docker compose, prism is installed on the container using ubunut as the base image.
After that, uvicorn will start prism server on port 8000.