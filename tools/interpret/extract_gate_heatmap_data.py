# -*- coding: utf-8 -*-
"""
MMK_Mix のゲーティング重み（MoKLayer の softmax score）を抽出して、
(1) npy 保存
(2) ヒートマップ png 保存
を行うスクリプト。

特徴:
- MMK_Mix.py を改造しない（forward hookで score を横取り）
- compare_models_kw.py と同じ読み込み経路を使用（PeakRunner, get_model, NPYDataInterface）
- 学習時と合わせて GHI 強調（var_x[:,:,2] *= 1.3）を適用

出力:
- out_dir/
  - gate_layer0.npy, gate_layer1.npy, ...  (shape: [num_samples, N, n_expert] or [num_samples, n_expert])
  - gate_heatmap_layer0.png, ...
  - meta.txt  (expert名、shape、集約方法など)

実行例（CMDワンライナー）:
  python tools\\extract_gate_heatmap_data.py --model_dir save_mmk_optimized --config config\\tsukuba_conf\\MMK_Mix_PV_96.py --out_dir results_gate --aggregate var_mean
"""

import os
import sys
import glob
import argparse
import importlib.util

import numpy as np
import torch
import matplotlib.pyplot as plt

# -------------------------
# パス設定（tools配下から親ディレクトリを import 可能にする）
# -------------------------
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# compare_models_kw.py と同じ依存（traingraph 側に PeakRunner, get_model がある前提）
from traingraph import PeakRunner, get_model  # noqa


def find_checkpoint(model_dir: str) -> str:
    """
    model_dir 配下から ckpt を見つける。
    - best を優先（あれば）
    - なければ最新更新の ckpt を使う
    """
    search_pattern = os.path.join(parent_dir, model_dir, "**", "*.ckpt")
    ckpts = glob.glob(search_pattern, recursive=True)
    if not ckpts:
        raise FileNotFoundError(f"[ERROR] {model_dir} 配下に .ckpt が見つかりません: {search_pattern}")

    best = [p for p in ckpts if "best" in os.path.basename(p)]
    target = max(best, key=os.path.getmtime) if best else max(ckpts, key=os.path.getmtime)
    return target


def load_config(config_path: str) -> dict:
    """config/*.py の exp_conf を読み込む"""
    abs_path = os.path.join(parent_dir, config_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"[ERROR] config が見つかりません: {abs_path}")

    spec = importlib.util.spec_from_file_location("config_module", abs_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)  # type: ignore
    if not hasattr(config_module, "exp_conf"):
        raise AttributeError(f"[ERROR] {config_path} に exp_conf がありません")
    return config_module.exp_conf


def get_expert_names(exp_conf: dict):
    """
    exp_conf["layer_hp"] から expert 名を推定する。
    layer_hp は [(type_name, hp_dict), ...] を想定。
    形式が違っても落ちにくいように保険を入れる。
    """
    names = []
    layer_hp = exp_conf.get("layer_hp", None)
    if isinstance(layer_hp, (list, tuple)) and len(layer_hp) > 0:
        for item in layer_hp:
            # item が ("WavKAN", {...}) のような形式を想定
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                names.append(str(item[0]))
            else:
                names.append(str(item))
    if not names:
        # 最悪の場合：expert数が分かってから "expert0, expert1..." にする
        names = None
    return names


def plot_heatmap(gate_2d: np.ndarray, expert_names, save_path: str, title: str):
    """
    gate_2d: shape [T, E]（時間/サンプル × expert）
    """
    plt.figure(figsize=(14, 4))
    # [E, T] にして expert を縦軸にする
    img = plt.imshow(gate_2d.T, aspect="auto", interpolation="nearest")
    plt.colorbar(img, label="Gate weight [-]")

    if expert_names is not None and len(expert_names) == gate_2d.shape[1]:
        plt.yticks(np.arange(len(expert_names)), expert_names)
    else:
        plt.yticks(np.arange(gate_2d.shape[1]), [f"expert{i}" for i in range(gate_2d.shape[1])])

    plt.xlabel("Sample index (test) [-]")
    plt.ylabel("Expert [-]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="save_mmk_optimized", help="ckpt が入っているディレクトリ（親からの相対）")
    ap.add_argument("--config", default=r"config\tsukuba_conf\MMK_Mix_PV_96.py", help="exp_conf を含む config")
    ap.add_argument("--out_dir", default="results_gate", help="出力先フォルダ")
    ap.add_argument("--aggregate", choices=["none", "var_keep", "var_mean"], default="var_mean",
                    help="gate の集約方法: none=(B*N,E), var_keep=(B,N,E), var_mean=(B,E)")
    ap.add_argument("--max_batches", type=int, default=-1, help="デバッグ用: 先頭から何バッチだけ回すか。-1で全て")
    ap.add_argument("--device", default="cpu", help="cpu か cuda（使えるなら）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # -------------------------
    # ckpt / config / dataloader
    # -------------------------
    ckpt_path = find_checkpoint(args.model_dir)
    exp_conf = load_config(args.config)

    from easytsf.runner.data_runner import NPYDataInterface  # 過去スクリプトと同じ
    data_module = NPYDataInterface(**exp_conf)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    base_model = get_model(exp_conf["model_name"], exp_conf)
    model = PeakRunner.load_from_checkpoint(ckpt_path, model=base_model, config=exp_conf, strict=False).eval()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model.to(device)

    expert_names = get_expert_names(exp_conf)

    # -------------------------
    # forward hook で gate(score) を回収
    # -------------------------
    hooks = []
    # layerごとに [batchごとのgate] を貯める
    # collected[layer_idx] = list of np.ndarray
    collected = []

    # MoKLayer を探して hook を付ける（MMK_Mix.py を改造しない作戦）
    mok_layers = []
    for m in model.modules():
        # クラス名で判定（import パス差異に強くする）
        if m.__class__.__name__ == "MoKLayer":
            mok_layers.append(m)

    if len(mok_layers) == 0:
        raise RuntimeError("[ERROR] MoKLayer が見つかりませんでした。MMK_Mix モデルを読み込めていない可能性があります。")

    for i in range(len(mok_layers)):
        collected.append([])

    def make_hook(layer_idx: int):
        def _hook(module, inputs, outputs):
            # outputs は (y, score) を期待
            if not (isinstance(outputs, (tuple, list)) and len(outputs) >= 2):
                return
            score = outputs[1]  # (B*N, E)
            if torch.is_tensor(score):
                collected[layer_idx].append(score.detach().cpu().numpy())
        return _hook

    for li, layer in enumerate(mok_layers):
        hooks.append(layer.register_forward_hook(make_hook(li)))

    # -------------------------
    # 推論（test）を回して gate を貯める
    # -------------------------
    sample_count = 0
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            # batch: var_x, marker_x, var_y, marker_y（既存ツールと同じ想定）
            var_x, marker_x, var_y, marker_y = [_.float().to(device) for _ in batch]

            # 学習時と一致させる：GHI 強調（既存ツールと同じ）
            var_x_input = var_x.clone()
            # 0: PV, 1: temp, 2: GHI, ... の前提（あなたの既存コードに合わせる）
            var_x_input[:, :, 2] *= 1.3

            _ = model(var_x_input, marker_x)

            sample_count += var_x.shape[0]  # B を加算

            if args.max_batches > 0 and (bi + 1) >= args.max_batches:
                break

    # hook解除（念のため）
    for h in hooks:
        h.remove()

    # -------------------------
    # gate を整形して保存 + ヒートマップ生成
    # -------------------------
    meta_lines = []
    meta_lines.append(f"ckpt_path={ckpt_path}")
    meta_lines.append(f"config={args.config}")
    meta_lines.append(f"aggregate={args.aggregate}")
    meta_lines.append(f"num_samples(B total)={sample_count}")
    meta_lines.append(f"num_layers_detected={len(mok_layers)}")
    meta_lines.append(f"expert_names={expert_names}")

    for li in range(len(collected)):
        if len(collected[li]) == 0:
            meta_lines.append(f"[WARN] layer{li}: gate が1件も取れていません")
            continue

        g = np.concatenate(collected[li], axis=0)  # (sum(B*N), E)
        E = g.shape[1]

        # B と N に分解するために、各バッチごとの B を足し上げた sample_count は分かるが、
        # N（特徴量数）は exp_conf["var_num"] を使う（存在しない場合は推定を試す）
        N = exp_conf.get("var_num", None)
        if N is None:
            # var_num が無い場合：g.shape[0] / sample_count が N になっているはず
            N = int(round(g.shape[0] / max(sample_count, 1)))

        # 形状チェック
        if sample_count * N != g.shape[0]:
            meta_lines.append(f"[WARN] layer{li}: reshape できませんでした。gate.shape={g.shape}, B={sample_count}, N={N}")
            # そのまま保存
            save_npy = os.path.join(args.out_dir, f"gate_layer{li}_raw.npy")
            np.save(save_npy, g)
            continue

        g_bne = g.reshape(sample_count, N, E)  # (B_total, N, E)

        if args.aggregate == "none":
            # (B_total*N, E) に戻して保存
            g_save = g_bne.reshape(sample_count * N, E)
        elif args.aggregate == "var_keep":
            # (B_total, N, E)
            g_save = g_bne
        else:
            # var_mean: (B_total, E)
            g_save = g_bne.mean(axis=1)

        save_npy = os.path.join(args.out_dir, f"gate_layer{li}.npy")
        np.save(save_npy, g_save)

        meta_lines.append(f"layer{li}: saved {save_npy}, shape={g_save.shape}")

        # ヒートマップ（時間×expert）用に [T,E] にそろえる
        if g_save.ndim == 2:
            gate_2d = g_save  # (B_total, E)
        elif g_save.ndim == 3:
            # var_keep の場合は変数平均して表示（図は見やすさ重視）
            gate_2d = g_save.mean(axis=1)  # (B_total, E)
        else:
            continue

        # expert_names が無い場合は E から自動命名
        if expert_names is None or len(expert_names) != E:
            exp_names_for_plot = [f"expert{i}" for i in range(E)]
        else:
            exp_names_for_plot = expert_names

        save_png = os.path.join(args.out_dir, f"gate_heatmap_layer{li}.png")
        plot_heatmap(
            gate_2d=gate_2d,
            expert_names=exp_names_for_plot,
            save_path=save_png,
            title=f"MMK_Mix gating heatmap (layer {li})"
        )
        meta_lines.append(f"layer{li}: saved {save_png}")

    meta_path = os.path.join(args.out_dir, "meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines))

    print("[SUCCESS] gate の抽出とヒートマップ保存が完了しました。")
    print(f"  out_dir: {os.path.abspath(args.out_dir)}")
    print(f"  meta   : {os.path.abspath(meta_path)}")


if __name__ == "__main__":
    main()
