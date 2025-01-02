#!bin/bash
for c_init in {2,2.5,3,3.5}; do
    for num_simulations in {120,150,180}; do
        sbatch CC_script.sh $c_init $num_simulations
    done
done