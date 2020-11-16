for stimseed in 10 20 30
do
    # for syn_location in 0 1 2 5 8 
    for syn_location in 0 1 10 11 12 13 14 15
    do
	for seed in 0 1 2 3 4 5 6 7 8
	do
	    command='python bg_modulation.py full --duration_per_bg_level 400 --stim_delay 300 --NSTIMs 0 2 4 6 8 10 12 14 --active'
	    $command --stimseed $stimseed  --seed $seed --syn_location $syn_location --bg_level 0 &
	    $command --stimseed $stimseed  --seed $seed --syn_location $syn_location --bg_level 0.5 &
	    $command --stimseed $stimseed  --seed $seed --syn_location $syn_location --bg_level 1. &
	    $command --stimseed $stimseed  --seed $seed --syn_location $syn_location --bg_level 1.5 &
	    $command --stimseed $stimseed  --seed $seed --syn_location $syn_location --bg_level 2. &
	    $command --stimseed $stimseed  --seed $seed --syn_location $syn_location --bg_level 2.5
	done
    done
done
