#!/bin/bash
# This script is meant to simplify the upgrade of the various provided requirements
# files to the latest available package versions. We assume, that the script is
# either called from the project root directly or from its subfolder 'requirements' of
# the project root.

# Switch to requirements folder, if not already in it.
if [ -d requirements ]; then
    cd requirements
fi

# Cycle through the different Python environments and update the according two
# requirements files by issuing the according pip-tools command pip-compile from within
# the specific environments. Since twine is a dependency of python-semantic-release and
# current versions of twine are not compatible with Python 3.5 anymore, this gives a
# failure for the Python 3.5 environement. It should though be possible to upgrade the
# environment without the --upgrade flag for all subdependencies at least.
for PYVENV in  "py35" "py36" "py37" "py38"
do
    source ../../../envs/PyDynamic-$PYVENV/bin/activate && \
    pip-compile --upgrade requirements.in --output-file requirements-$PYVENV.txt && \
    pip-compile --upgrade dev-requirements-$PYVENV.in \
        --output-file dev-requirements-$PYVENV.txt && \
    deactivate
done
