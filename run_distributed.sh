#!/bin/bash

# This is an example launcher script for running a distributed job with accelerate.
#
# Before running this, you must first run `accelerate config` in your terminal
# to set up your environment for multi-GPU or FSDP (Fully Sharded Data Parallelism).
#
# Usage:
# ./run_distributed.sh your_python_script.py --your --args
#
# The arguments after the script name will be passed directly to your Python script.

# The `accelerate launch` command will read the default configuration file
# you created with `accelerate config` to determine how to distribute the job.
# It handles setting all the necessary environment variables.

accelerate launch "$@" 