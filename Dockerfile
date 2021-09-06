ARG CHECKPOINTS_IMG=cgdoc/casia-olhwcr-tf:latest

FROM ${CHECKPOINTS_IMG} AS backups

FROM tensorflow/tensorflow:2.6.0 AS trainer

ARG workdir=/opt/olhwcr

WORKDIR $workdir

ARG train_pot_dir=train_pot
ARG val_pot_dir=val_pot
ARG checkpoint_dir=ckpts
ARG tensorboard_dir=tb_logs
ARG backup_dir=backup_n_restore

# FIXME: experiment
ARG epochs=5

COPY training/* src/

COPY --from=backups /$checkpoint_dir $checkpoint_dir/
COPY --from=backups /$tensorboard_dir $tensorboard_dir/
COPY --from=backups /$backup_dir $backup_dir/


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
    && python3 src/model.py -d $workdir -t $train_pot_dir -v $val_pot_dir -C $checkpoint_dir -T $tensorboard_dir -R $backup_dir -E $epochs


FROM scratch

COPY --from=trainer /opt/olhwcr/ckpts /ckpts/
COPY --from=trainer /opt/olhwcr/tb_logs /tb_logs/
COPY --from=trainer /opt/olhwcr/backup_n_restore /backup_n_restore/