#!/bin/bash

# Specify the arguments you want 
L=1
act="erf"
k=50 #how many lines until thermalisation in the run file
curr_dir=$(pwd)
file_name="${curr_dir}/data_analysis.py"
# Define the directories to search for folders
cd ..
dir_name="runs"
search_dirs=("${dir_name}/teacher_cifar10_net_${L}hl_actf_${act}/" "${dir_name}/teacher_mnist_net_${L}hl_actf_${act}/")

# Loop through the search directories
for dir in "${search_dirs[@]}"
do
  if [ -d "$dir" ]
  then
    for folder in "$dir"*/
    do
      if [ "$(ls -A $folder)" ]
      then
        cp "$file_name" "$folder"
        cd "$folder"
        python "$file_name" -k $k
        cd -
      fi
    done
  fi
done
