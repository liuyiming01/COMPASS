export CUDA_VISIBLE_DEVICES=1,5

python paper_classification.py
python table_classification.py
# python e2e_data.py

echo "All evaluations completed successfully."