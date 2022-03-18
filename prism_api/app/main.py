from fastapi import FastAPI
import subprocess

app = FastAPI()

@app.get("/health")
def health():
    return {"health": "ok"}

prism_script = "prism/prism/bin/prism"

@app.get("/prism_test")
def prism_run_test():
    pm_file = "app/die.pm"
    property_file = "app/die.pctl"
    prism_command = " ".join([prism_script, pm_file, property_file])
    output = subprocess.getoutput(prism_command)
    return output

@app.get("/prism")
def prism_run(pm_file: str, property_file: str, prop_id: str):    
    prism_command = " ".join([prism_script, pm_file, property_file, "-prop", str(prop_id)])
    output = subprocess.getoutput(prism_command)
    return output