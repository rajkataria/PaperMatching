#declare -a greedy_steps=("0" "1" "2")
declare -a greedy_steps=("2")
declare -a tpms_weights=("0.8")
declare -a subject_area_weights=("0.2")
declare -a suggestion_weights=("10.0")
declare -a experience_weights=("0.0")
declare -a configs=("low")

declare inputs_folder="scoring-inputs"

for g in "${greedy_steps[@]}"
do
	for t in "${tpms_weights[@]}"
	do
		for a in "${subject_area_weights[@]}"
		do
			for s in "${suggestion_weights[@]}"
			do
				for e in "${experience_weights[@]}"
				do
					for c in "${configs[@]}"
						do
						echo "**********************************************************************************";
						python -u calculate_scoring_matrix.py \
							-u ./$inputs_folder/Users.txt \
							-r ./$inputs_folder/reviewers.csv \
							-t ./$inputs_folder/ReviewerTpmsScores_ICCV2019.csv \
							-p ./$inputs_folder/Papers.csv \
							-s ./$inputs_folder/ReviewerSuggestions.txt \
							-c ./$inputs_folder/ReviewerConflicts.txt \
							-n 3 \
							-g $g \
							-w_t $t \
							-w_a $a \
							-w_s $s \
							-w_e $e \
							-o $c \
							--cached_folder ./output-w_t-$t-w_a-$a-w_s-$s-w_e-$e-g-$g-n-3-config-$c/ > output-w_t-$t-w_a-$a-w_s-$s-w_e-$e-g-$g-n-3-config-$c.log 2>&1 &
						echo "Running assignments with w_t=$t, w_a=$a, w_s=$s, w_e=$e, g=$g config=$c";
					done;
				done;
			done;
		done;
	done;
done
echo "**********************************************************************************";
