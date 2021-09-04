FROM tensorflow/tensorflow:2.6.0 AS trainer

ARG workdir=/opt/olhwcr

WORKDIR $workdir

ARG train_pot_dir=train_pot
ARG val_pot_dir=val_pot

COPY training/* src/

RUN apt-get update \
\
    && apt-get install -y unzip \
\
    && pip install opencv-python-headless Pillow \
\
    && rm -rf /var/lib/apt/lists/* \
\
    && curl -L -O http://www.nlpr.ia.ac.cn/databases/download/feature_data/OLHWDB1.1trn_pot.zip \
    && curl -L -O http://www.nlpr.ia.ac.cn/databases/download/feature_data/OLHWDB1.1tst_pot.zip \
\
    && unzip OLHWDB1.1trn_pot.zip -d $train_pot_dir \
    && unzip OLHWDB1.1tst_pot.zip -d $val_pot_dir \
\
    && python3 src/model.py -d $workdir -t $train_pot_dir -v $val_pot_dir -E 15000


FROM scratch

COPY --from=trainer /opt/olhwcr/ckpts /ckpts/
COPY --from=trainer /opt/olhwcr/tb_logs /tb_logs/