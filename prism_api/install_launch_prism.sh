#!/bin/sh

# Script for installation of PRISM on a clean install of Ubuntu

set -e # Abort if one of the commands fail
set -x # Print commands as they are executed

apt update
apt install sudo

# Install dependencies: make/gcc/Java/git
sudo apt-get -y update
sudo apt -y install make gcc g++ default-jdk git

# Install Python (only needed for testing (prism-auto) currently)
sudo apt -y install python

# Download the latest development version from GitHub if prism dir does not exist
if [ ! -d prism ]; then
  git clone https://github.com/prismmodelchecker/prism.git
fi

# Compile PRISM and run a single test
# (should ultimately display: "Testing result: PASS")
(cd prism/prism && make && make test)

# launch prism server
uvicorn app.main:app --host 0.0.0.0 --port 8000