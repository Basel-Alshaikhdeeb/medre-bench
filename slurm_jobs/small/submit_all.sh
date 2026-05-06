#!/bin/bash
set -e
cd "$(dirname "$0")"

sbatch biobert__ddi__seed42.sh
