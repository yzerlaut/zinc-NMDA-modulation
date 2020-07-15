#!/bin/bash
# bash bash/calib-passive.sh # RUN
# python passive_props.py calib-analysis # ANALYSIS
# bash bash/calib-chelated-zinc.sh # RUN
# python calibration_runs.py chelated-zinc-calib-analysis # ANALYSIS
# bash bash/calib-free-zinc.sh # RUN
bash bash/calib-free-zinc-1.sh # RUN
bash bash/calib-free-zinc-2.sh # RUN
bash bash/calib-free-zinc-3.sh # RUN
python calibration_runs.py free-zinc-calib-analysis # ANALYSIS
