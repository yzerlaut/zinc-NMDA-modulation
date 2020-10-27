# for i in $(seq 0 2 10)
for alphaZn in 0.00 0.45
do
    # for alphaZn in 0.00 0.25 0.35 0.45
    for i in $(seq 0 1 4)
    do
	echo 'alphaZn' $alphaZn 'syn_location' $i
	python bg_modulation.py run --syn_location $i --alphaZn $alphaZn --NbgSEEDS 10 &
    done
	python bg_modulation.py run --syn_location $i --ampa_only --NbgSEEDS 10
done
