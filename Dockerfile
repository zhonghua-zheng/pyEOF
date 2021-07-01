FROM condaforge/miniforge3

RUN conda install -y \
        numpy \
        pandas \
        scipy \
        xarray \
        scikit-learn \
        rpy2 \
        jupyter \
        matplotlib \
        pooch \
        eofs \
    && pip --no-cache-dir install \ 
        pyEOF \
    && conda clean --all 

CMD ["jupyter", "notebook", "--port=8888", "--no-browser",  "--ip=0.0.0.0", "--notebook-dir=/home", "--allow-root"]

# docker build -t zzheng93/pyeof .
# docker run -it --rm -p 8888:8888 -v $PWD:/home zzheng93/pyeof
# docker push zzheng93/pyeof