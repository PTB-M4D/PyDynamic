#!/bin/bash
#
# This script is meant to simplify the upgrade of the various provided requirements
# files to the latest available package versions. We assume, that the script is
# either called from the project root directly or from its subfolder 'requirements'.
# All provided requirements-files are updated according to the
# specified dependencies from setup.py and the dev-requirements-files for all the
# different versions.
# The production dependencies belong into the according list 'install_requires' in
# setup.py and the development dependencies into the dev-requirements.in-file.
# For execution the script needs virtual environments, one for each of the upstream
# supported Python versions, with pip-tools installed. Those environments need to be
# placed at ../envs/PyDynamic-PYTHONVERSION relative to the project root.
# 'PYTHONVERSION' takes the value 'py3X', where X is one of the numbers in the line
# starting with 'for PYVENV in ' in this script, prescribing the supported Python
# versions. If you want to execute this script on Windows you have to adapt all path
# separators to backslashes.
# The script starts with navigating to the project root, if it was called from
# the subfolder ./requirements/.
if [ -f requirements.txt ] && [ -d ../PyDynamic/ ] && [ -d ../requirements/ ]; then
    cd ..
fi

# Handle all Python versions via setup.py by cycling through the different Python
# environments and update the corresponding two requirements files by issuing the
# appropriate pip-tools command pip-compile from within the specific environments.
export PYTHONPATH=$PYTHONPATH:$(pwd)
for PYVENV in "7" "8" "9" "10"
do
    echo "
Compile dependencies for Python 3.$PYVENV
====================================
    "
    # Activate according Python environment.
    source ../envs/PyDynamic-py3$PYVENV/bin/activate && \
    # Upgrade pip and pip-tools.
    python -m pip install --upgrade pip pip-tools && \
    # Create requirements...txt from setup.py.
    python -m piptools compile --upgrade --output-file requirements/requirements-py3$PYVENV.txt && \
    # Create dev-requirements...txt from dev-requirements...in.
    python -m piptools compile --upgrade requirements/dev-requirements-py3$PYVENV.in --output-file requirements/dev-requirements-py3$PYVENV.txt && \
    deactivate
done
