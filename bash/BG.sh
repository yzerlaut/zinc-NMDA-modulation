for i in $(seq 0 1 3)
do
    for bg in $(seq 0 2 4)
    do
	for iseed in $(seq 4 1 7)
	do
	    seed=$(($i+$bg+3*$iseed))
	    python bg_modulation.py full --seed $seed --bg_level $bg --syn_location $i --duration_per_bg_level 2000 --stim_delay 300 --NSTIMs 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 &
	done
	seed=$(($seed+1))
	python bg_modulation.py full --seed $seed --bg_level $bg --syn_location $i --duration_per_bg_level 2000 --stim_delay 300 --NSTIMs 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
    done
done
