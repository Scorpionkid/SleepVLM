# scripts/preprocess_hmc.py
import os
import glob
import argparse
import numpy as np
import pandas as pd

from sleepvlm.data.preprocess import load_psg_signals
from sleepvlm.data.renderer import render_psg_from_dict

HMC_CHANNEL_CONFIG = {
    'F4':   ('EEG F4-M1',),
    'C4':   ('EEG C4-M1',),
    'O2':   ('EEG O2-M1',),
    'LOC':  ('EOG E1-M2',),
    'ROC':  ('EOG E2-M2',),
    'Chin': ('EMG chin',),
}

STAGE_TEXT_MAP = {
    "Sleep stage W":  0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage R":  4,
}

EPOCH_DURATION = 30


def load_hmc_stages(txt_path):
    df = pd.read_csv(txt_path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df["Annotation"] = df["Annotation"].str.strip()

    rows = df[df["Annotation"].isin(STAGE_TEXT_MAP.keys())].copy()
    if rows.empty:
        return np.array([], dtype=np.int32)

    rows["epoch_idx"] = (rows["Recording onset"].astype(float) / EPOCH_DURATION).round().astype(int)

    n_epochs = int(rows["epoch_idx"].max()) + 1
    stages = np.full(n_epochs, -1, dtype=np.int32)
    for _, r in rows.iterrows():
        stages[int(r["epoch_idx"])] = STAGE_TEXT_MAP[r["Annotation"]]
    return stages


def build_channel_dict(edf_path, eeg_only=False):
    _, sig_dict = load_psg_signals(edf_path, HMC_CHANNEL_CONFIG)
    available = [ch for ch in HMC_CHANNEL_CONFIG if ch in sig_dict]
    filtered = {ch: sig_dict[ch] for ch in available}

    if eeg_only:
        # 保留 6 通道的图像版式（band 数量/位置不变），
        # 只把 EOG/EMG 通道置零，模拟"只有EEG"的部署场景
        eeg_present = [ch for ch in ("F4", "C4", "O2") if ch in filtered]
        if eeg_present:
            fs = filtered[eeg_present[0]]["sample_rate"]
            length = len(filtered[eeg_present[0]]["data"])
            for ch in ("LOC", "ROC", "Chin"):
                if ch in filtered:
                    filtered[ch] = {"sample_rate": fs, "data": np.zeros(length)}

    return filtered


def process_hmc_subject(subject_id, edf_path, txt_path, output_dir, eeg_only=False):
    filtered = build_channel_dict(edf_path, eeg_only=eeg_only)
    if not filtered:
        print(f"[Skip] {subject_id}: no valid channels resolved")
        return False

    stages = load_hmc_stages(txt_path)
    if stages.size == 0:
        print(f"[Skip] {subject_id}: no stage annotations found")
        return False

    rendered = render_psg_from_dict(filtered, stages, output_dir, subject_id)
    print(f"{subject_id}: rendered {len(rendered)} epochs -> {output_dir}")
    return len(rendered) > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/media/HDD4/personal_files/EEG_workspace/sleep/HMC/data/datas")
    parser.add_argument("--output_dir", default="/media/HDD4/personal_files/EEG_workspace/sleep/HMC/data/images")
    parser.add_argument("--limit", type=int, default=None, help="只处理前N个subject，测试用")
    parser.add_argument("--subjects", type=str, default=None, help="逗号分隔的subject id列表，如 SN001,SN002")
    parser.add_argument("--eeg_only", action="store_true", help="只用EEG通道，EOG/EMG置零")
    args = parser.parse_args()

    edf_files = sorted(glob.glob(os.path.join(args.data_dir, "SN*.edf")))
    edf_files = [f for f in edf_files if "sleepscoring" not in f]

    if args.subjects:
        wanted = set(args.subjects.split(","))
        edf_files = [f for f in edf_files
                     if os.path.basename(f).replace(".edf", "") in wanted]
    elif args.limit:
        edf_files = edf_files[: args.limit]

    for edf_path in edf_files:
        subject_id = os.path.basename(edf_path).replace(".edf", "")
        txt_path = os.path.join(args.data_dir, f"{subject_id}_sleepscoring.txt")
        if not os.path.isfile(txt_path):
            print(f"[Skip] {subject_id}: no scoring txt found")
            continue
        process_hmc_subject(subject_id, edf_path, txt_path, args.output_dir, eeg_only=args.eeg_only)

