import os
import sys
import glob
import shutil
from tqdm import tqdm

SOURCE_FOLDER = "/media/joe46973/HD-Vogelsang"
DEST_FOLDER = "/home/joe46973/Masterarbeit/data/MODISSA/sequences"
FOLDERS_TO_COPY = ["o1", "o2", "view"]

SEQUENCES = dict(Sequence_TEST = [34,58], Sequence_0=[34, 633], Sequence_3=[715, 814], Sequence_4=[1200, 1400], Sequence_5=[1900, 2100],
                 Sequence_6=[2200, 2600], Sequence_7=[3850, 4000], Sequence_8=[6950,7250], Sequence_9=[8595, 6730],
                 Sequence_10=[8900, 9050], Sequence_11=[9830, 10030], Sequence_12=[10480, 10680], Sequence_13=[12950, 13250])
SEQUENCES = dict(Sequence_11=[9830, 10030])

for seq, start_end in tqdm(SEQUENCES.items(), total=len(SEQUENCES)):
    for folder in tqdm(FOLDERS_TO_COPY,total=len(FOLDERS_TO_COPY), leave=False):
        os.makedirs(os.path.join(DEST_FOLDER,seq, folder), exist_ok=True)
        FILE_TYPE= "png" if folder=="view" else "pcd"
        for i in tqdm(range(start_end[0], start_end[1]+1), leave=False):
            file_name = f'{i:07d}.' + FILE_TYPE
            file = os.path.join(SOURCE_FOLDER, folder, file_name)
            dest = os.path.join(DEST_FOLDER, seq, folder, "PCD" if FILE_TYPE=="pcd" else ".")
            shutil.copy(file,dest)

