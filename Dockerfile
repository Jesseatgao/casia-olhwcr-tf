FROM tensorflow/tensorflow:2.6.0 AS trainer

ENV WORKDIR /opt/olhwcr

WORKDIR $WORKDIR

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
    && curl -L -O https://github.com/Jesseatgao/tmp_dataset/raw/master/train_pot_dir.zip \
    && curl -L -O https://github.com/Jesseatgao/tmp_dataset/raw/master/val_pot_dir.zip \
\
    && unzip train_pot_dir.zip -d $train_pot_dir \
    && unzip val_pot_dir.zip -d $val_pot_dir \
\
    && python3 src/model.py -d $WORKDIR -t $train_pot_dir -v $val_pot_dir -E 1


FROM scratch

COPY --from=trainer /opt/olhwcr/ckpts /ckpts/
COPY --from=trainer /opt/olhwcr/tb_logs /tb_logs/