#!/bin/bash

cd /workspaces/LabeLMaker/src
pip install --upgrade pip setuptools wheel\
	    && pip install -e ".[dev]"
