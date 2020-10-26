for i in $(seq 0 1 10)
do
    for alphaZn in 0.00 0.25 0.35 0.45
    do
	python bg_modulation.py run --syn_location $i --alphaZn $alphaZn &
	python bg_modulation.py run --syn_location $(($i+1)) --alphaZn $alphaZn &
    done
	python bg_modulation.py run --syn_location $i --ampa_only &
	python bg_modulation.py run --syn_location $(($i+1)) --ampa_only
done
