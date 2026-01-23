# python3 dataset_rewrite_literary.py --dataset_path ./datasets/counterfact_filtered.jsonl --split train --output_dir ./datasets --limit 200





python3 ./preprocessing/dataset_rewrite_geo.py --dataset_path ./datasets/counterfact_spatial_filtered.jsonl --generate all --output_dir ./processed_data --limit 200


# python3 ./preprocessing/dataset_rewrite_geo.py --dataset_path ./datasets/counterfact_spatial_filtered.jsonl --split train --generate rewrite --rewrite_categories 4_discrimination --output_dir ./processed_data --limit 5