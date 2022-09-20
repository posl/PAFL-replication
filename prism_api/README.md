prism_api
====
Prism does not work depending on the architecture of the host machine (e.g., M1 Mac). 

Therefore, we construct a Web API of prism using FastAPI.

The environment is containerized by Docker so that prism can be used regardless of the host machine environment.

Directory strunture:
- `app/` : FastAPI codes
- `prism/` : prism files (will be installed when container started)
- `Dockerfile` : Dockerfile of prism_server (base image: ubuntu)
- `install_launch_prism.sh` : shell script for install prism and launch FastAPI server (via `uvicorn`)

When this container is launched by docker compose, prism is installed on the container using ubunut as the base image.
After that, uvicorn will start prism server on port 8000.
These procedure is written in `install_launch_prism.sh`.

# How to Check the Prism API
If the container named `prism_server` is built correctly by docker compose, then you can use the Prism API via `localhost:8000`.

## Check the API document
Since it is implemented with FastAPI, you can check the API document by typing `localhost:8000/docs` into the your browser.

## Interfaces
Three interfaces below are provided:
### **`health`**
Interface to simply check if the server is alive or not.
Type `localhost:8000/health` into the your browser, or use `curl` command like below:
```bash
curl -X 'GET' \
  'http://localhost:8000/health' \
  -H 'accept: application/json'
```
If `{"health":"ok"}` returns, it is no problem for the status of the server.

### **`prism_test`**
Interface to check if Prism runs correctly.
Type `localhost:8000/prism_test` into the your browser, or use `curl` command like below:
```bash
curl -X 'GET' \
  'http://localhost:8000/prism_test' \
  -H 'accept: application/json'
```
This command executes the prism script prepared on the server side in advance, and the result is obtained as an HTTP response.

HTTP response:
```
"PRISM\n=====\n\nVersion: 4.7.dev\nDate: Tue Sep 20 21:26:04 UTC 2022\nHostname: 1bbfa9710345\nMemory limits: cudd=1g, java(heap)=1g\nCommand line: prism app/die.pm app/die.pctl\n\nParsing model file \"app/die.pm\"...\n\nType:        DTMC\nModules:     die \nVariables:   s d \n\nParsing properties file \"app/die.pctl\"...\n\n1 property:\n(1) P=? [ F s=7&d=1 ]\n\n---------------------------------------------------------------------\n\nModel checking: P=? [ F s=7&d=1 ]\n\nBuilding model...\n\nComputing reachable states...\n\nReachability (BFS): 4 iterations in 0.00 seconds (average 0.000000, setup 0.00)\n\nTime for model construction: 0.045 seconds.\n\nType:        DTMC\nStates:      13 (1 initial)\nTransitions: 20\n\nTransition matrix: 71 nodes (3 terminal), 20 minterms, vars: 6r/6c\n\nProb0: 4 iterations in 0.00 seconds (average 0.000000, setup 0.00)\n\nProb1: 3 iterations in 0.00 seconds (average 0.000000, setup 0.00)\n\nyes = 1, no = 9, maybe = 3\n\nComputing remaining probabilities...\nEngine: Hybrid\n\nBuilding hybrid MTBDD matrix... [levels=6, nodes=30] [1.4 KB]\nAdding explicit sparse matrices... [levels=6, num=1, compact] [0.0 KB]\nCreating vector for diagonals... [dist=1, compact] [0.0 KB]\nCreating vector for RHS... [dist=2, compact] [0.0 KB]\nAllocating iteration vectors... [2 x 0.1 KB]\nTOTAL: [1.7 KB]\n\nStarting iterations...\n\nJacobi: 22 iterations in 0.00 seconds (average 0.000000, setup 0.00)\n\nValue in the initial state: 0.16666650772094727\n\nTime for model checking: 0.014 seconds.\n\nResult: 0.16666650772094727 (+/- 1.1920928955078125E-6 estimated; rel err 7.1525641942636435E-6)\n"
```

### **`prism`**
The actual interface to call in PAFL.
This interface can be executed against any prism file by passing a prism script or other parameters, while `prism_test` does not receive any parameters.

Since HTTP requests are made by the python program in PAFL, users do not need to be aware of these details of usage.