# tools/ray_tune_MMK.py
# MMK 用の簡易「Ray Tune 風」グリッドサーチスクリプト
# - base_config（例: config/tsukuba_conf/MMK_PVProcessed_96for96.py）を読み込み
# - 学習率などのハイパラを差し替えた trial 用 config を自動生成
# - train.py をサブプロセスで実行し、標準出力から test/mae, test/mse を取得
# - summary.json に全試行結果を書き出し、Best Trial を表示

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path


# ---------- 共通ユーティリティ ----------

def load_exp_conf(py_path: str) -> dict:
    """
    base の config .py から exp_conf を dict として読み込む。
    """
    spec = importlib.util.spec_from_file_location("user_conf", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "exp_conf"):
        raise RuntimeError(f"{py_path} に exp_conf がありません。")

    # dict にしておく（後で書き換えやすいように）
    return dict(mod.exp_conf)


def write_conf(exp_conf: dict, out_path: str) -> None:
    """
    exp_conf を含む config .py を out_path に書き出す。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# auto-generated for MMK hyperparam search\n")
        f.write("exp_conf = ")
        # repr でそのまま Python リテラルとして書き出す
        f.write(repr(exp_conf))
        f.write("\n")


def run_train_and_capture_metrics(config_path: str, seed: int = 2025):
    """
    train.py を 1 回実行し、標準出力から test/mae, test/mse をパースする。

    戻り値:
        (mae, mse) もしくは (None, None) （失敗時）
    """
    cmd = [
        sys.executable,
        "train.py",
        "-c", config_path,
        "-s", "save",
        "-d", ".",
        "--devices", "1",
        "--seed", str(seed),
    ]

    print(f"\n[INFO] 実行コマンド: {' '.join(cmd)}")

    # stdout / stderr をまとめて取得
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print("------ train.py output (truncated) ------")
    # ログが長くなりすぎないように末尾だけ少し表示
    tail = "\n".join(proc.stdout.splitlines()[-40:])
    print(tail)
    print("------ end of output ------")

    if proc.returncode != 0:
        print(f"[WARN] train.py がエラー終了しました (returncode={proc.returncode})")
        return None, None

    # Lightning の test 結果から MAE/MSE を正規表現で抜き出す
    # 例：
    # test/mae             0.687812089920044
    # test/mse             0.6598495841026306
    mae = None
    mse = None

    for line in proc.stdout.splitlines():
        m_mae = re.search(r"test/mae\s+([0-9.eE+\-]+)", line)
        if m_mae:
            try:
                mae = float(m_mae.group(1))
            except ValueError:
                pass

        m_mse = re.search(r"test/mse\s+([0-9.eE+\-]+)", line)
        if m_mse:
            try:
                mse = float(m_mse.group(1))
            except ValueError:
                pass

    if mae is None or mse is None:
        print("[WARN] ログから test/mae, test/mse を取得できませんでした。")
        return None, None

    print(f"[INFO] 取得した指標: test/mae = {mae}, test/mse = {mse}")
    return mae, mse


# ---------- メイン処理 ----------

def main():
    parser = argparse.ArgumentParser(description="MMK 用ハイパーパラメータグリッドサーチスクリプト")
    parser.add_argument(
        "--base-config",
        default="config/tsukuba_conf/MMK_PVProcessed_96for96.py",
        help="ベースとなる設定ファイル (.py)",
    )
    parser.add_argument(
        "--save-dir",
        default="ray_results_MMK",
        help="各 trial の設定ファイルや summary.json を保存するディレクトリ",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="train.py に渡す乱数シード",
    )
    args = parser.parse_args()

    base_config_path = args.base_config
    save_root = Path(args.save_dir)
    seed = args.seed

    print(f"[INFO] base config = {base_config_path}")
    print(f"[INFO] save dir    = {save_root}")
    print(f"[INFO] seed        = {seed}")

    # ベース設定を読み込み
    base_conf = load_exp_conf(base_config_path)

    # ---------- ここでチューニングしたいハイパラのグリッドを定義 ----------
    lr_list = [1e-3, 5e-4, 1e-4]          # 学習率
    d_model_list = [256]                  # MMK 側で無視されても OK（exp_conf に入るだけ）
    e_layers_list = [2]                   # 同上
    dropout_list = [0.1]                  # 同上

    print("\n[INFO] 探索するハイパーパラメータの組み合わせ:")
    for lr in lr_list:
        for dm in d_model_list:
            for el in e_layers_list:
                for do in dropout_list:
                    print(f"  lr={lr:g}, d_model={dm}, e_layers={el}, dropout={do}")

    save_root.mkdir(parents=True, exist_ok=True)

    all_results = []

    # ---------- 全組み合わせループ ----------
    for lr in lr_list:
        for dm in d_model_list:
            for el in e_layers_list:
                for do in dropout_list:
                    trial_name = f"lr{lr:g}_dm{dm}_el{el}_do{do}"
                    trial_dir = save_root / trial_name
                    trial_dir.mkdir(parents=True, exist_ok=True)

                    print("\n==================================================")
                    print(f"[TRIAL] {trial_name}")
                    print("==================================================")

                    # ベース設定をコピーして、この trial 用に上書き
                    trial_conf = dict(base_conf)

                    # ★★★ 必須パラメータの補完 ①: var_num ★★★
                    if "var_num" not in trial_conf:
                        print("[WARN] base_conf に 'var_num' が無かったので、暫定的に var_num=1 を設定します。")
                        trial_conf["var_num"] = 1
                    else:
                        trial_conf["var_num"] = base_conf["var_num"]

                    # ★★★ 必須パラメータの補完 ②: optimizer_weight_decay ★★★
                    if "optimizer_weight_decay" not in trial_conf:
                        if "weight_decay" in base_conf:
                            wd = base_conf["weight_decay"]
                            print(f"[INFO] base_conf の 'weight_decay'={wd} を optimizer_weight_decay として使用します。")
                            trial_conf["optimizer_weight_decay"] = wd
                        else:
                            print("[WARN] base_conf に 'optimizer_weight_decay' も 'weight_decay' も無かったので、optimizer_weight_decay=0.0 を設定します。")
                            trial_conf["optimizer_weight_decay"] = 0.0
                    else:
                        trial_conf["optimizer_weight_decay"] = base_conf["optimizer_weight_decay"]

                    # ★★★ 必須パラメータの補完 ③: lr_step_size ★★★
                    if "lr_step_size" not in trial_conf:
                        if "lr_step_size" in base_conf:
                            ss = base_conf["lr_step_size"]
                            print(f"[INFO] base_conf の 'lr_step_size'={ss} を使用します。")
                            trial_conf["lr_step_size"] = ss
                        else:
                            # 適当なデフォルト（必要なら変えて OK）
                            print("[WARN] base_conf に 'lr_step_size' が無かったので、lr_step_size=10 を設定します。")
                            trial_conf["lr_step_size"] = 10

                    # ★★★ 必須パラメータの補完 ④: lr_gamma ★★★
                    if "lr_gamma" not in trial_conf:
                        if "lr_gamma" in base_conf:
                            gm = base_conf["lr_gamma"]
                            print(f"[INFO] base_conf の 'lr_gamma'={gm} を使用します。")
                            trial_conf["lr_gamma"] = gm
                        else:
                            print("[WARN] base_conf に 'lr_gamma' が無かったので、lr_gamma=0.5 を設定します。")
                            trial_conf["lr_gamma"] = 0.5

                    # ここでハイパラを上書き（MMK の __init__ で使われないキーは無視されるだけ）
                    trial_conf["lr"] = float(lr)
                    trial_conf["d_model"] = int(dm)
                    trial_conf["e_layers"] = int(el)
                    trial_conf["dropout"] = float(do)

                    # ログ/保存用に trial 名を埋めておくと後で見やすい
                    trial_conf["trial_name"] = trial_name

                    # trial 用 config ファイルを書き出し
                    trial_conf_path = trial_dir / "exp_conf_trial.py"
                    write_conf(trial_conf, trial_conf_path)

                    # train.py を実行して MAE/MSE を取得
                    mae, mse = run_train_and_capture_metrics(str(trial_conf_path), seed=seed)

                    result = {
                        "trial_name": trial_name,
                        "config_path": str(trial_conf_path),
                        "lr": lr,
                        "d_model": dm,
                        "e_layers": el,
                        "dropout": do,
                        "mae": mae,
                        "mse": mse,
                    }
                    all_results.append(result)

                    print("\n== TRIAL RESULT ==")
                    print(json.dumps(result, ensure_ascii=False, indent=2))

    # ---------- 全試行結果を summary.json に保存 ----------
    summary_path = save_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n=== 全試行の結果を {summary_path} に保存しました ===")

    # ---------- MAE ベースでベスト trial を表示 ----------
    valid_results = [r for r in all_results if r["mae"] is not None]
    if valid_results:
        best = min(valid_results, key=lambda r: r["mae"])
        print("=== Best Trial (by MAE) ===")
        print(json.dumps(best, ensure_ascii=False, indent=2))
    else:
        print("[WARN] 有効な結果が 1 つもありませんでした（すべて学習失敗または MAE/MSE パース失敗）。")


if __name__ == "__main__":
    main()
