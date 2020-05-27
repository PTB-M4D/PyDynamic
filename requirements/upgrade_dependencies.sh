#!/bin/bash -x
if [ -d requirements ]; then
    cd requirements
fi
for PYVENV in  "py35" "py36" "py37" "py38"
do
    source ../../../envs/PyDynamic-$PYVENV/bin/activate && \
    pip-compile --upgrade requirements.in --output-file requirements-$PYVENV.txt && \
    pip-compile --upgrade dev-requirements-$PYVENV.in --output-file dev-requirements-$PYVENV.txt && \
    deactivate
done
