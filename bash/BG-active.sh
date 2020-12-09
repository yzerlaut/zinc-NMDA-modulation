command='python bg_modulation.py run --duration_per_bg_level 700 --stim_delay 50 --NSTIMs 0 6 8 10 12 14 16 18 --Nsyn 18 --active'
# --- demo config ---
# for preset in L23 L4 AMPA
# do
#     $command --stimseed 10  --seed 1 --syn_location 1 --preset $preset --bg_level 1. --store_full_data &
# done
# --- batch sim ---
Nproc=4
n=1
for stimseed in $(seq 10 10 30)
do
    for syn_location in $(seq 1 5)
    do
	for seed in $(seq 1 10)
	do
	    for bg_level in 0 1 2 3 4
	    do
		# for preset in L23 L4 AMPA
		for preset in L4
		do
		    echo [active] Running preset=$preset, stimseed=$stimseed, seed=$seed, syn_location=$syn_location, bg_level=$bg_level
		    if [ $(($n % $Nproc)) -eq 0 ]
		    then
			$command --stimseed $stimseed  --seed $seed --syn_location $syn_location --preset $preset --bg_level $bg_level
		    else
			$command --stimseed $stimseed  --seed $seed --syn_location $syn_location --preset $preset --bg_level $bg_level &
		    fi
		    n=$(($n+1))
		done
	    done
	done
    done
done
