for stimseed in 10 20 30
do
    for syn_location in 0 1 2 5 8 
    do
	for seed in 10 11 12 13 14 15 16 17 18
	do
	    command='python bg_modulation.py full --seed $seed --syn_location $syn_location --duration_per_bg_level 2000 --stim_delay 300 --NSTIMs 0 2 4 6 8 10 12 14 16 18 20'
	    $command --bg_level 0 &
	    $command --bg_level 1 &
	    $command --bg_level 2 &
	    $command --bg_level 3 &
	    $command --bg_level 4
	done
    done
done
