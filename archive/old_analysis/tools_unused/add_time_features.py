# -*- coding: utf-8 -*-
# 目的: 入力特徴(23)に、周期的な時間特徴を4本追加して 27 次元にする
# 追加: sin/cos(hour/24), sin/cos(doy/365)
# 出力: splits_96_96_weather_timeF/ に .npy を保存（y はそのまま）
import os, numpy as np

SRC = r"dataset_PV\splits_96_96_weather"
DST = r"dataset_PV\splits_96_96_weather_timeF"
os.makedirs(DST, exist_ok=True)

def add_time_feats(x):
    # x: (N, 96, F)
    N, H, F = x.shape
    # 時刻インデックスから周期特徴を作る（簡易版）
    hours = np.arange(H) % 24
    doys  = np.arange(H) % 365
    hsin = np.sin(2*np.pi*hours/24)[None, :, None].repeat(N,0)
    hcos = np.cos(2*np.pi*hours/24)[None, :, None].repeat(N,0)
    dsin = np.sin(2*np.pi*doys /365)[None, :, None].repeat(N,0)
    dcos = np.cos(2*np.pi*doys /365)[None, :, None].repeat(N,0)
    extra = np.concatenate([hsin, hcos, dsin, dcos], axis=2).astype(np.float32)  # (N,96,4)
    return np.concatenate([x.astype(np.float32), extra], axis=2)  # (N,96,F+4=27)

for split in ["train","val","test"]:
    x = np.load(os.path.join(SRC, f"{split}_x.npy"))
    y = np.load(os.path.join(SRC, f"{split}_y.npy"))
    x2 = add_time_feats(x)
    np.save(os.path.join(DST, f"{split}_x.npy"), x2)
    np.save(os.path.join(DST, f"{split}_y.npy"), y)
print("done ->", DST)
