#!/bin/bash

#SBATCH -t 30:00:00
#SBATCH --mem=4G

for i in {1..100}
do
    /users/sk99/myenv/bin/python3 -u acrobot.py param/lr1e-3_h3_nn300_tuf1 max $i
	/users/sk99/myenv/bin/python3 -u acrobot.py param/lr1e-3_h3_nn300_tuf1 mellow 1 $i
    /users/sk99/myenv/bin/python3 -u acrobot.py param/lr1e-3_h3_nn300_tuf1 mellow 2 $i
    /users/sk99/myenv/bin/python3 -u acrobot.py param/lr1e-3_h3_nn300_tuf1 mellow 5 $i
    /users/sk99/myenv/bin/python3 -u acrobot.py param/lr1e-3_h3_nn300_tuf1 mellow 10 $i
    /users/sk99/myenv/bin/python3 -u acrobot.py param/lr1e-3_h3_nn300_tuf100 max $i
done

