# python bg_modulation.py run --NbgSEEDS 10 --save_presynaptic_input
for i in $(seq 0 1 4)
do
    for alphaZn in 0.00 0.25 0.35 0.45
    do
	echo 'alphaZn' $alphaZn 'syn_location' $i
	python bg_modulation.py run --syn_location $i --alphaZn $alphaZn --use_preloaded_presynact &
    done
    python bg_modulation.py run --syn_location $i --ampa_only --use_preloaded_presynact
done
