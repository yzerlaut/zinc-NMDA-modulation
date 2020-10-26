for i in $(seq 1 1 2)
do
    # for alphaZn in 0.00 0.25 0.35 0.45
    for bg in 0 3
    do
	seed=$(($i+$bg+3))
	for alphaZn in 0.00 0.45
	do
	    python bg_modulation.py run --seed $seed --bg_level $bg --syn_location $i --alphaZn $alphaZn --active &
	    python bg_modulation.py run --seed $seed --bg_level $bg --syn_location $(($i+1)) --alphaZn $alphaZn --active &
	done
	python bg_modulation.py run --seed $seed --syn_location $i --ampa_only --active  &
	python bg_modulation.py run --seed $seed --syn_location $(($i+1)) --ampa_only --active 
    done
done
