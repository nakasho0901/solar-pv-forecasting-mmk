import argparse
import json
import os
from typing import Dict, List, Tuple

import torch


# ============================================================
# Utility: find TaylorKAN coefficients keys in a Lightning ckpt
# ============================================================
def find_taylor_coeff_keys(state_dict: Dict[str, torch.Tensor]) -> List[Tuple[str, str]]:
    """
    Find (coeff_key, bias_key) pairs from a state_dict.

    We expect keys like:
      - "...transform.coeffs" : shape (out_dim, in_dim, order)
      - "...transform.bias"   : shape (1, out_dim) or (out_dim,)

    Returns:
      List of tuples: [(coeff_key, bias_key_or_empty), ...]
    """
    coeff_keys = [k for k in state_dict.keys() if k.endswith("transform.coeffs")]
    pairs = []
    for ck in sorted(coeff_keys):
        bk = ck.replace("transform.coeffs", "transform.bias")
        if bk in state_dict:
            pairs.append((ck, bk))
        else:
            pairs.append((ck, ""))  # bias might be missing
    return pairs


# ============================================================
# LaTeX formatting
# ============================================================
def fmt_float(x: float, digits: int = 6) -> str:
    """
    Format float for LaTeX.
    - Use scientific notation when needed.
    """
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-4:
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


def build_full_latex_for_output(
    coeffs: torch.Tensor,
    bias: torch.Tensor,
    out_idx: int,
    var_symbol: str = "x",
    y_symbol: str = r"\hat{y}",
    digits: int = 6,
) -> str:
    """
    Build a FULL LaTeX polynomial for a single output dimension.

    coeffs shape: (out_dim, in_dim, order)
    bias shape  : (out_dim,) or (1, out_dim)

    Polynomial:
      y_o = b_o + sum_{j=1..in_dim} sum_{m=0..order-1} a_{o,j,m} * x_j^m

    Note:
      x_j is a token component (unitless internal representation).
    """
    # Ensure bias is 1D (out_dim,)
    if bias is None:
        b0 = 0.0
    else:
        if bias.ndim == 2 and bias.shape[0] == 1:
            b0 = float(bias[0, out_idx].item())
        else:
            b0 = float(bias[out_idx].item())

    out_dim, in_dim, order = coeffs.shape
    assert 0 <= out_idx < out_dim, f"out_idx={out_idx} is out of range [0,{out_dim-1}]"

    lines = []
    lines.append(r"\[")
    lines.append(f"{y_symbol}_{{{out_idx}}} = {fmt_float(b0, digits)}")

    # Add terms: group by input dimension j
    # We include m=0 term as well, but note it can be merged into bias.
    for j in range(in_dim):
        # Collect polynomial for this x_j
        parts = []
        for m in range(order):
            a = float(coeffs[out_idx, j, m].item())
            if a == 0.0:
                continue
            coef = fmt_float(a, digits)

            # term representation
            if m == 0:
                term = f"{coef}"
            elif m == 1:
                term = f"{coef}{var_symbol}_{{{j}}}"
            else:
                term = f"{coef}{var_symbol}_{{{j}}}^{{{m}}}"
            parts.append(term)

        if parts:
            # Add as "+ ( ... )" for each j
            poly_j = " + ".join(parts)
            lines.append(rf"+ \left( {poly_j} \right)")

    lines.append(r"\]")
    return "\n".join(lines)


def build_topk_latex_for_output(
    coeffs: torch.Tensor,
    bias: torch.Tensor,
    out_idx: int,
    topk: int = 50,
    var_symbol: str = "x",
    y_symbol: str = r"\hat{y}",
    digits: int = 6,
) -> str:
    """
    Build a TOP-K LaTeX polynomial for a single output dimension, selecting terms
    by |coefficient| magnitude (simple and robust).

    This is useful because the full polynomial can be huge:
      in_dim=64, order=4 => up to 256 terms (+ constant)
    """
    # Bias
    if bias is None:
        b0 = 0.0
    else:
        if bias.ndim == 2 and bias.shape[0] == 1:
            b0 = float(bias[0, out_idx].item())
        else:
            b0 = float(bias[out_idx].item())

    out_dim, in_dim, order = coeffs.shape
    assert 0 <= out_idx < out_dim

    # Collect (abs(a), j, m, a)
    terms = []
    for j in range(in_dim):
        for m in range(order):
            a = float(coeffs[out_idx, j, m].item())
            if a == 0.0:
                continue
            terms.append((abs(a), j, m, a))

    terms.sort(key=lambda t: t[0], reverse=True)
    terms = terms[:max(0, topk)]

    lines = []
    lines.append(r"\[")
    lines.append(f"{y_symbol}_{{{out_idx}}} = {fmt_float(b0, digits)}")

    for _, j, m, a in terms:
        coef = fmt_float(a, digits)
        if m == 0:
            term = f"{coef}"
        elif m == 1:
            term = f"{coef}{var_symbol}_{{{j}}}"
        else:
            term = f"{coef}{var_symbol}_{{{j}}}^{{{m}}}"
        lines.append(f"+ {term}")

    lines.append(r"\]")
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Extract TaylorKAN polynomial (LaTeX) from a Lightning checkpoint (last.ckpt)."
    )
    parser.add_argument("--ckpt", required=True, help="Path to .ckpt (Lightning checkpoint).")
    parser.add_argument("--outdir", default="results_formula", help="Output directory.")
    parser.add_argument(
        "--pair_index",
        type=int,
        default=0,
        help="Which TaylorKAN coeff/bias pair to use (if multiple are found).",
    )
    parser.add_argument(
        "--out_idx",
        type=int,
        default=0,
        help="Which output dimension o to export (0..out_dim-1). For pv head you may want 0.",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "topk"],
        default="topk",
        help="full: all terms; topk: only largest |coef| terms.",
    )
    parser.add_argument("--topk", type=int, default=60, help="Number of terms for topk mode.")
    parser.add_argument(
        "--var_symbol",
        default="x",
        help="Variable symbol for token components. x_j means token dimension j (unitless).",
    )
    parser.add_argument("--y_symbol", default=r"\hat{y}", help="Output symbol in LaTeX.")
    parser.add_argument("--digits", type=int, default=6, help="Digits for float formatting.")
    parser.add_argument(
        "--meta_json",
        default="",
        help="Optional meta.json (not required). If provided, we will store it alongside output for traceability.",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "state_dict" not in ckpt:
        raise ValueError("This ckpt does not contain 'state_dict'. Are you sure it's a Lightning checkpoint?")
    sd = ckpt["state_dict"]

    # Find TaylorKAN keys
    pairs = find_taylor_coeff_keys(sd)
    if not pairs:
        # Give a helpful hint: list keys that contain 'taylor' or 'coeffs'
        hints = [k for k in sd.keys() if ("coeff" in k.lower() or "taylor" in k.lower())]
        hint_path = os.path.join(args.outdir, "debug_keys_hint.txt")
        with open(hint_path, "w", encoding="utf-8") as f:
            f.write("No keys ending with 'transform.coeffs' were found.\n")
            f.write("Here are some keys containing 'coeff' or 'taylor':\n")
            for k in sorted(hints)[:500]:
                f.write(k + "\n")
        raise RuntimeError(
            "TaylorKAN coeff keys not found. Check results_formula/debug_keys_hint.txt for hints."
        )

    # Save key list
    keylist_path = os.path.join(args.outdir, "found_taylor_pairs.json")
    with open(keylist_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"coeff_key": ck, "bias_key": bk} for ck, bk in pairs],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Select pair
    if not (0 <= args.pair_index < len(pairs)):
        raise ValueError(f"--pair_index must be in [0, {len(pairs)-1}] but got {args.pair_index}")

    coeff_key, bias_key = pairs[args.pair_index]
    coeffs = sd[coeff_key].detach().cpu()
    bias = sd[bias_key].detach().cpu() if bias_key else None

    # Validate shape
    if coeffs.ndim != 3:
        raise ValueError(f"Expected coeffs to be 3D (out_dim,in_dim,order) but got shape={tuple(coeffs.shape)}")

    out_dim, in_dim, order = coeffs.shape

    # Build LaTeX
    if args.mode == "full":
        latex = build_full_latex_for_output(
            coeffs=coeffs,
            bias=bias,
            out_idx=args.out_idx,
            var_symbol=args.var_symbol,
            y_symbol=args.y_symbol,
            digits=args.digits,
        )
    else:
        latex = build_topk_latex_for_output(
            coeffs=coeffs,
            bias=bias,
            out_idx=args.out_idx,
            topk=args.topk,
            var_symbol=args.var_symbol,
            y_symbol=args.y_symbol,
            digits=args.digits,
        )

    # Write output
    out_tex = os.path.join(args.outdir, f"taylor_formula_pair{args.pair_index}_out{args.out_idx}_{args.mode}.tex")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("% TaylorKAN polynomial extracted from checkpoint\n")
        f.write(f"% ckpt: {args.ckpt}\n")
        f.write(f"% coeff_key: {coeff_key}\n")
        f.write(f"% bias_key : {bias_key}\n")
        f.write(f"% coeffs shape: (out_dim={out_dim}, in_dim={in_dim}, order={order})\n")
        f.write("% NOTE: x_j is token component (unitless internal representation), not raw feature.\n\n")
        f.write(latex)
        f.write("\n")

    # Optionally copy meta.json for traceability
    if args.meta_json:
        try:
            with open(args.meta_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta_out = os.path.join(args.outdir, "meta_used.json")
            with open(meta_out, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to copy meta_json: {e}")

    # Print summary
    print("[OK] Found TaylorKAN pairs:", len(pairs))
    print("[OK] Using pair_index:", args.pair_index)
    print("[OK] coeff_key:", coeff_key)
    print("[OK] bias_key :", bias_key if bias_key else "(none)")
    print("[OK] coeffs shape:", tuple(coeffs.shape))
    print("[OK] Saved LaTeX:", out_tex)
    print("[HINT] Use found_taylor_pairs.json to switch pair_index if you want another layer/block.")


if __name__ == "__main__":
    main()
