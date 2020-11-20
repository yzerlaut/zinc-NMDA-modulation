command='python bg_modulation.py run --duration_per_bg_level 2000 --stim_delay 500 --NSTIMs 0 2 4 6 8 10 12 14 16 18 --Nsyn 18'
for syn_location in $(seq 0 20)
do
    for stimseed in $(seq 10 10 100)
    do
	for seed in $(seq 1 10)
	do
	    for aZn in 0 0.45
	    do
		echo [passive] Running stimseed=$stimseed, seed=$seed, syn_location=$syn_location, alphaZn=$aZn
		$command --stimseed $stimseed  --seed $seed --syn_location $syn_location --alphaZn $aZn --bg_level 0. &
		$command --stimseed $stimseed  --seed $seed --syn_location $syn_location --alphaZn $aZn --bg_level 1. &
		$command --stimseed $stimseed  --seed $seed --syn_location $syn_location --alphaZn $aZn --bg_level 2. &
		$command --stimseed $stimseed  --seed $seed --syn_location $syn_location --alphaZn $aZn --bg_level 3.
	    done
	done
    done
done
