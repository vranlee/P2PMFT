cd src

CUDA_VISIBLE_DEVICES='0' python3 track.py cmot \
--val_mft24 True \
--data_dir /data1/LWR/vranlee/DATASETS/MFT \
--load_model ../exp/bst.pth \
--arch sbt/lbT

cd ..