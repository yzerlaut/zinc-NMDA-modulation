# for stimseed in 40 50 60
# do
#     for syn_location in 10 11 12 13 14 15 16 17 18 19
#     do
# 	for seed in 0 1 2 3 4 5 6 7 8 9 
# 	do
# 	    command='python bg_modulation.py full --duration_per_bg_level 2000 --stim_delay 700 --NSTIMs 0 2 4 6 8 10 12 14 16 18 20'
# 	    $command --stimseed $stimseed  --seed $seed --syn_location $syn_location --bg_level 0 &
# 	    $command --stimseed $stimseed  --seed $seed --syn_location $syn_location --bg_level 1 &
# 	    $command --stimseed $stimseed  --seed $seed --syn_location $syn_location --bg_level 2 &
# 	    $command --stimseed $stimseed  --seed $seed --syn_location $syn_location --bg_level 3 &


# 	    $command --stimseed $stimseed  --seed $seed --syn_location $syn_location --bg_level 4
# 	done
#     done
# done
command='python bg_modulation.py run --duration_per_bg_level 2000 --stim_delay 700 --NSTIMs 0 2 4 6 8 10 12 14 16 18 20'
$command --stimseed 10  --seed 1 --syn_location 0 --bg_level 0 --ampa_only &
$command --stimseed 10  --seed 1 --syn_location 0 --bg_level 2 --ampa_only &
