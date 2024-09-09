# Generation of triplets

## 1: Generation

### Random poses (joint perturbations / taking steps):
```
p humanoid_generate_posneg_v3.py -r -k 10000 --output_log --debug
```

### Sequence:
```
python humanoid_generate_seq_triplets.py --output_log
p quick_viz.py
```

### Flipping:
```
python humanoid_generate_flipping_triplets.py --num_triplets 10000 --output_log
```

## 2: Manual test set: get 200 to choose from manually.

1. Use `humanoid_generate_select_subset.py` to select k (e.g. k = 50) samples from the generated data.
2. Manually inspect

### For flipping:

In: `FOLDER = f"{OUTPUT_ROOT}/v3_flipping/manual_test"`

4 from existing targets:
```
DICT_KEYS=("arms_bracket_right_final_only" "left_arm_extend_wave_higher_final_only" "arms_up_then_down_left" "arms_up_then_down_right")

for key in "${DICT_KEYS[@]}"; do
    python humanoid_generate_flipping_triplets.py --seq_name "$key" --num_triplets 1 --output_log
done
```

46 from generated samples.


## 3. Train/val/test splits:

Randomly select
- 5k from train
- 2k for val
- 2k for test
among the remaining ones.

```
python humanoid_generate_split_data.py
```

## 4. Combine the splits into one json file for every split:

```
python humanoid_combine_splits.py
```