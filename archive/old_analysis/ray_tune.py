import argparse
import importlib.util
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple


def load_exp_conf(config_path: str) -> Dict[str, Any]:
    """設定ファイル(.py)をimportして、exp_conf辞書を取り出す"""
    spec = importlib.util.spec_from_file_location("cfg", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config module from {config_path}")
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)  # type: ignore[arg-type]
    if not hasattr(cfg, "exp_conf"):
        raise AttributeError(f"{config_path} に exp_conf が定義されていません")
    exp_conf = getattr(cfg, "exp_conf")
    if not isinstance(exp_conf, dict):
        raise TypeError("exp_conf は dict である必要があります")
    return exp_conf


def save_temp_conf(base_conf: Dict[str, Any], out_path: str) -> None:
    """exp_conf辞書を、そのまま Python の設定ファイルとして書き出す"""
    lines = [
        "# 自動生成された一時設定ファイル",
        "exp_conf = " + repr(base_conf),
        "",
    ]
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def run_train(config_path: str, save_root: str, seed: int = 2025) -> Tuple[float, float]:
    """
    train.py をサブプロセスで実行し、最後に print された test/mae, test/mse を拾って返す想定。
    （train.py が test/mae, test/mse を標準出力に出している前提）
    """
    cmd = [
        sys.executable,
        "train.py",
        "-c",
        config_path,
        "-s",
        save_root,
        "-d",
        ".",
        "--devices",
        "1",
        "--seed",
        str(seed),
    ]
    print("== Run:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"train.py failed with code {proc.returncode}")

    # stdout から test/mae, test/mse をパース（lightning の出力を前提にかなり雑に拾う）
    mae = None
    mse = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("test/mae"):
            # 例: "test/mae             0.6878"
            try:
                mae = float(line.split()[-1])
            except Exception:
                pass
        if line.startswith("test/mse"):
            try:
                mse = float(line.split()[-1])
            except Exception:
                pass
    if mae is None or mse is None:
        raise RuntimeError("stdout から test/mae, test/mse を取得できませんでした")
    return mae, mse


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple grid search for hyperparameters.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="ベースとなる設定ファイル(.py)のパス",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default="ray_results",
        help="結果(一時configやログJSON)を保存するルートディレクトリ",
    )
    parser.add_argument("--seed", type=int, default=2025, help="train.py に渡す乱数シード")

    # 探索したいハイパーパラメータの候補リスト
    parser.add_argument(
        "--lr_list",
        type=float,
        nargs="+",
        default=[1e-4, 5e-4, 1e-3],
        help="学習率の候補リスト",
    )
    parser.add_argument(
        "--d_model_list",
        type=int,
        nargs="+",
        default=[256],
        help="d_model の候補リスト（iTransformer など）",
    )
    parser.add_argument(
        "--e_layers_list",
        type=int,
        nargs="+",
        default=[2],
        help="Encoder 層数 e_layers の候補リスト",
    )
    parser.add_argument(
        "--dropout_list",
        type=float,
        nargs="+",
        default=[0.1],
        help="dropout の候補リスト",
    )

    args = parser.parse_args()

    base_conf = load_exp_conf(args.config)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    # すべての組み合わせでグリッドサーチ
    for lr, d_model, e_layers, dropout in itertools.product(
        args.lr_list, args.d_model_list, args.e_layers_list, args.dropout_list
    ):
        # exp_conf をコピーして、この組み合わせ用に上書き
        trial_conf = dict(base_conf)
        trial_conf["lr"] = lr
        if "d_model" in trial_conf:
            trial_conf["d_model"] = d_model
        if "e_layers" in trial_conf:
            trial_conf["e_layers"] = e_layers
        if "dropout" in trial_conf:
            trial_conf["dropout"] = dropout

        # 一意な名前を作成
        trial_name = f"lr{lr:g}_dm{d_model}_el{e_layers}_do{dropout:g}"
        trial_dir = out_root / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_conf_path = trial_dir / "exp_conf_trial.py"

        # 一時設定ファイルを書き出し
        save_temp_conf(trial_conf, str(trial_conf_path))

        # 学習実行
        mae, mse = run_train(str(trial_conf_path), save_root=str(trial_dir), seed=args.seed)

        # 結果を保存
        trial_result = {
            "trial_name": trial_name,
            "config_path": str(trial_conf_path),
            "lr": lr,
            "d_model": d_model,
            "e_layers": e_layers,
            "dropout": dropout,
            "mae": mae,
            "mse": mse,
        }
        results.append(trial_result)
        print("== TRIAL RESULT:", json.dumps(trial_result, ensure_ascii=False, indent=2))

    # 全体の結果を JSON にまとめて保存
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"=== 全試行の結果を {summary_path} に保存しました ===")

    # ベスト試行を表示（ここでは MAE 最小）
    best = min(results, key=lambda r: r["mae"])
    print("=== Best Trial (by MAE) ===")
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
