#!/bin/bash
#
# This script is meant to update all provided requirements-files according to the
# specified dependencies from setup.py and the dev-requirements-files for all the
# different versions.
# The production dependencies belong into the according list 'install_requires' in
# setup.py and the development dependencies into the various dev-requirements.in-files.
# For execution the script needs virtual environments, one for each of the upstream
# supported Python versions, with pip-tools installed. Those environments need to be
# placed at ../envs/PyDynamic-$PYTHONVERSION relative to the project root. The proper
# naming for the versions you find in the line starting with 'for PYVENV in ' in this
# script.
# Since pip-tools did not work as convenient for Python 3.5 and there were
# conflicts for some of the newer packageversions this case is handled separately.
# Teh script starts with navigating to the project root, if it was called from
# the subfolder ./requirements/. 
if [ -f requirements.txt ] && [ -d ../PyDynamic/ ] && [ -d ../requirements/ ]; then
    cd ..
fi

# Handle Python 3.5 via requirements.in.
# Activate according Python environment.
source ../envs/PyDynamic-py35/bin/activate && \
# Create requirements.txt from requirements.in.
pip-compile --upgrade requirements/requirements.in --output-file requirements/requirements-py35.txt && \
# Create dev-requirements...txt from dev-requirements...in.
pip-compile requirements/dev-requirements-py35.in --output-file requirements/dev-requirements-py35.txt && \
deactivate

# Handle all other Python versions via setup.py.
export PYTHONPATH=$PYTHONPATH:$(pwd)
for PYVENV in "py36" "py37" "py38"
do
    # Activate according Python environment.
    source ../envs/PyDynamic-$PYVENV/bin/activate && \
    # Create requirements.txt from requirements.in.
    pip-compile --upgrade --output-file requirements/requirements-$PYVENV.txt && \
    # Create dev-requirements...txt from dev-requirements...in.
    pip-compile requirements/dev-requirements-$PYVENV.in --output-file requirements/dev-requirements-$PYVENV.txt && \
    deactivate
done
