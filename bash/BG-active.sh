for i in $(seq 0 1 10)
do
    # for alphaZn in 0.00 0.25 0.35 0.45
    for bg in 0 2 4 6 8
    do
	for alphaZn in 0.00 0.45
	do
	    for iseed in $(seq 4 1 5)
	    do
		seed=$(($i+$bg+3*$iseed))
		python bg_modulation.py run --seed $seed --bg_level $bg --syn_location $i --alphaZn $alphaZn --duration_per_bg_level 400 --stim_delay 300 --NSTIMs 4 6 8 10 12 14 16 --active &
	    done
	    iseed=6
	    seed=$(($i+$bg+3*$iseed))
	    python bg_modulation.py run --seed $seed --bg_level $bg --syn_location $i --alphaZn $alphaZn --duration_per_bg_level 400 --stim_delay 300 --NSTIMs 4 6 8 10 12 14 16 --active
	done
	# python bg_modulation.py plot --seed $seed --bg_level $bg --syn_location $i --ampa_only --duration_per_bg_level 500 --stim_delay 300 --NSTIMs 0 5 10 15 --active
    done
done
