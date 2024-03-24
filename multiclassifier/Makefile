help:
	@echo "INFO: make <tab> for targets"
.PHONY: help

jupyterlab:
	jupyter-lab
.PHONY: jupyterlab

jupyterlab-server:
	jupyter-lab --no-browser --port=8080 --ip=0.0.0.0
.PHONY: jupyterlab-server

clean:
	@echo "INFO: Use ultraclean instead"
.PHONY: clean

restore:
	git checkout images/*.png
	git checkout model/qkeras_model.h5
.PHONY: restore

ultraclean:
	rm -rf hls4ml*_prj
	rm -f hls4ml*_prj.tar.gz
	rm -rf __pycache__
.PHONY: ultraclean
