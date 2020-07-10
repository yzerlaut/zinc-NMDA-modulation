#!/bin/bash
## CHARACTERIZING PASSIVE AND SYNAPTIC PARAMS IN MORPHOLOGICAL MODEL
# bash bash/calib-passive.sh # RUN
# python passive_props.py calib-analysis # ANALYSIS
bash bash/calib-chelated-zinc.sh # RUN
python calibration_runs.py chelated-zinc-calib-analysis # ANALYSIS
bash bash/calib-free-zinc.sh # RUN
python calibration_runs.py free-zinc-calib-analysis # ANALYSIS
## NEURON RESPONSE TO SYNAPTIC STIMULATION
bash bash/syn-input-threshold.sh
