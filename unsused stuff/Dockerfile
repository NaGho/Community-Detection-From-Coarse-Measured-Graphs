FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --upgrade pip && \
	pip install --no-cache-dir -r requirements.txt && \
	pip install h5py && \
	pip install dask[dataframe] && \
	pip install rpy2 && \
	pip install pytest && \
	pip install snap && \
	pip install numpy && \
	pip install matplotlib && \
	pip install seaborn && \
	pip install pandas && \
	pip install networkx && \
	pip install cdlib && \
	pip install datetime && \
	pip install tzlocal && \
	pip install scipy
	
	
COPY . .

	
CMD ["python", "./coarsendSBM_communityRecovery_fromGraph.py"]