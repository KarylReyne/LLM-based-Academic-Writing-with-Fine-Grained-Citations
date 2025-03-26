#!/bin/bash
export PYTHONPATH=/mnt:$PYTHONPATH
python -m model.main -cfg /mnt/config/vit.json --train --scratch $1
