for i in $(seq 1 1 10)
do
    for alphaZn in 0.00 0.25 0.35 0.45
    do
	python bg_modulation.py run --syn_location $i --alphaZn $alphaZn &
	python bg_modulation.py run --syn_location $(($i+1)) --alphaZn $alphaZn &
    done
	python bg_modulation.py run --syn_location $i --ampa_only &
	python bg_modulation.py run --syn_location $(($i+1)) --ampa_only &
done    
# python bg_modulation.py run --bg_levels 0 2 4 6 --NbgSEEDS 10 --alphaZn 0.00 --syn_location 0 &
# python bg_modulation.py run --bg_levels 0 2 4 6 --NbgSEEDS 10 --alphaZn 0.25 --syn_location 0 &
# python bg_modulation.py run --bg_levels 0 2 4 6 --NbgSEEDS 10 --alphaZn 0.35 --syn_location 0 &
# python bg_modulation.py run --bg_levels 0 2 4 6 --NbgSEEDS 10 --alphaZn 0.45 --syn_location 0 &
# python bg_modulation.py run --bg_levels 0 2 4 6 --NbgSEEDS 10 --alphaZn 0.00 --syn_location 2 &
# python bg_modulation.py run --bg_levels 0 2 4 6 --NbgSEEDS 10 --alphaZn 0.25 --syn_location 2 &
# python bg_modulation.py run --bg_levels 0 2 4 6 --NbgSEEDS 10 --alphaZn 0.35 --syn_location 2 &
# python bg_modulation.py run --bg_levels 0 2 4 6 --NbgSEEDS 10 --alphaZn 0.45 --syn_location 2 
# python bg_modulation.py run --bg_levels 0 2 4 6 --NbgSEEDS 10 --alphaZn 0.00 --syn_location 3 &
# python bg_modulation.py run --bg_levels 0 2 4 6 --NbgSEEDS 10 --alphaZn 0.25 --syn_location 3 &
# python bg_modulation.py run --bg_levels 0 2 4 6 --NbgSEEDS 10 --alphaZn 0.35 --syn_location 3 &
# python bg_modulation.py run --bg_levels 0 2 4 6 --NbgSEEDS 10 --alphaZn 0.45 --syn_location 3 &
# python bg_modulation.py run --bg_levels 0 2 4 6 --NbgSEEDS 10 --alphaZn 0.00 --syn_location 4 &
# python bg_modulation.py run --bg_levels 0 2 4 6 --NbgSEEDS 10 --alphaZn 0.25 --syn_location 4 &
# python bg_modulation.py run --bg_levels 0 2 4 6 --NbgSEEDS 10 --alphaZn 0.35 --syn_location 4 &
# python bg_modulation.py run --bg_levels 0 2 4 6 --NbgSEEDS 10 --alphaZn 0.45 --syn_location 4 &
# python bg_modulation.py full --bg_levels 0 2 4 6 --NbgSEEDS 2 --alphaZn 0.45 &
# python bg_modulation.py full --bg_levels 0 1 2 3 4 5 6 --NbgSEEDS 3 --alphaZn 0.25 &
# python bg_modulation.py full --bg_levels 0 1 2 3 4 5 6 --NbgSEEDS 3 --alphaZn 0.35 &
# python bg_modulation.py full --bg_levels 0 1 2 3 4 5 6 --NbgSEEDS 2 --alphaZn 0.45 &
# python bg_modulation.py run --bg_levels 0 2 4 6 8 --NbgSEEDS 2 --chelated True --active --alphaZn 0.25 -sl 1 &
# python bg_modulation.py run --bg_levels 0 2 4 6 8 --NbgSEEDS 2 --chelated False --active --alphaZn 0.35 -sl 1 &
# python bg_modulation.py run --bg_levels 0 2 4 6 8 --NbgSEEDS 2 --chelated False --active --alphaZn 0.45 -sl 1 &
# python bg_modulation.py run --bg_levels 0 2 4 6 8 --NbgSEEDS 2 --chelated True --active --alphaZn 0.25 -sl 5 &
# python bg_modulation.py run --bg_levels 0 2 4 6 8 --NbgSEEDS 2 --chelated False --active --alphaZn 0.35 -sl 5 &
# python bg_modulation.py run --bg_levels 0 2 4 6 8 --NbgSEEDS 2 --chelated False --active --alphaZn 0.45 -sl 5 &
# python bg_modulation.py full --bg_levels 0 2 4 6 8 --NbgSEEDS 2 --alphaZn 0.3 &
# python bg_modulation.py full --bg_levels 0 2 4 6 8 --NbgSEEDS 2 --chelated True --alphaZn 0.3 &
# python bg_modulation.py full --bg_levels 0 2 4 6 8 --NbgSEEDS 2 --active --chelated True &
# python bg_modulation.py full --bg_levels 0 2 4 6 8 --NbgSEEDS 2 --active --alphaZn 0.3 &
# python bg_modulation.py full --bg_levels 0 2 4 6 8 --NbgSEEDS 2 --active --chelated True --alphaZn 0.3 &
