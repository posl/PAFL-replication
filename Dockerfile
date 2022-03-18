FROM continuumio/anaconda3:master
RUN conda install nltk -y && \
    conda install requests -y && \
    conda install gensim -y && \
    conda install pytorch torchvision -c pytorch -y
WORKDIR /src
ADD bash_setting /root/.bashrc
