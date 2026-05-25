SHELL := /bin/bash
CONDA_ACTIVATE := source $$(conda info --base)/etc/profile.d/conda.sh && conda activate insar

.PHONY: setup run score validate upload notebook clean all

setup:
	bash src/00_setup_ec2.sh

run:
	$(CONDA_ACTIVATE) && bash run_pipeline.sh

score:
	$(CONDA_ACTIVATE) && python src/04_score_rank.py

validate:
	$(CONDA_ACTIVATE) && python src/07_validate.py

upload:
	$(CONDA_ACTIVATE) && python src/06_upload_s3.py

notebook:
	$(CONDA_ACTIVATE) && jupyter nbconvert --to notebook --execute \
	  --ExecutePreprocessor.timeout=600 \
	  --output notebooks/village_growth_analysis.ipynb \
	  notebooks/village_growth_analysis.ipynb

clean:
	rm -f /data/satellite/kritter/processed/village_scored.csv
	rm -f /data/satellite/kritter/output/*.html
	rm -f /data/satellite/kritter/output/*.csv
	@echo "Cleaned output and scored CSVs (raw rasters preserved)"

all: run validate upload notebook
