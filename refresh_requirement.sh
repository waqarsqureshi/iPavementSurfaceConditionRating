#!/bin/bash

# Activate the conda environment (if you have a specific environment, replace 'myenv' with its name)
# source activate myenv

# Capture the environment using pip freeze
pip freeze > temp_requirements.txt

# Process the file to remove all @ references and save to requirements.txt
sed '/@/d' temp_requirements.txt > requirements.txt

# Remove the temporary file
rm temp_requirements.txt

echo "Processed requirements saved to requirements.txt"
