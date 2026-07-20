"""Executor principal do pipeline de filtragem e avaliacao.

Fluxo:
1) Aplica filtros em um conjunto de TIFFs.
2) Avalia metricas de fidelidade (RMSE, SSIM, USDSAI) contra o gold standard.
3) Executa segmentacao por Level Set para metricas funcionais (TP/FP/AC/HD/HM).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence
import numpy as np

from compare import ComparisonResult, run_comparison
from kuan import AVAILABLE_FILTERS, run_batch
from octave import generate_speckle_batch
from segmentation import SegmentationResult, run_segmentation

VALID_FILTERS = AVAILABLE_FILTERS


@dataclass(frozen=True)
class PipelineRun:
    filter_name: str
    filtered_outputs: list[Path]
    comparison_results: list[ComparisonResult]
    segmentation_results: list[SegmentationResult]
    compare_csv: Path
    segmentation_output_dir: Path


def _count_csv_rows(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def _count_segmentation_masks(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for candidate in path.glob("mask_*.tif") if candidate.is_file())


def cumulative_counts(base_output_root: Path, filter_name: str) -> tuple[int, int, int]:
    run_dirs = sorted(
        candidate
        for candidate in base_output_root.glob("run_*")
        if candidate.is_dir()
    )
    run_count = 0
    comparison_total = 0
    segmentation_total = 0

    for run_dir in run_dirs:
        compare_csv = run_dir / "compare" / f"{filter_name}_metrics.csv"
        compare_count = _count_csv_rows(compare_csv)
        if compare_count == 0:
            continue
        run_count += 1
        comparison_total += compare_count
        segmentation_total += _count_segmentation_masks(run_dir / "segmentation" / filter_name)

    return run_count, comparison_total, segmentation_total


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def _iter_tiff_candidates(path: Path, recursive: bool = True) -> list[Path]:
    if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
        return [path]
    if not path.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    return sorted(
        candidate
        for candidate in path.glob(pattern)
        if candidate.is_file() and candidate.suffix.lower() in {".tif", ".tiff"}
    )


def resolve_gold_path(gold_path: Path, recursive: bool = True) -> Path:
    if gold_path.is_file():
        return gold_path

    candidates = _iter_tiff_candidates(gold_path, recursive=recursive)
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum arquivo TIFF de gold standard encontrado em: {gold_path}"
        )

    # Seleção determinística para diretórios com múltiplos TIFFs.
    priority_names = {
        "cistosgoldstd.tif",
        "cistosgoldstd.tiff",
        "gold.tif",
        "gold.tiff",
        "gold_standard.tif",
        "gold_standard.tiff",
    }
    named_hits = [c for c in candidates if c.name.lower() in priority_names]
    if len(named_hits) == 1:
        return named_hits[0]
    if len(named_hits) > 1:
        options = "\n".join(f" - {candidate}" for candidate in named_hits[:10])
        suffix = "\n - ..." if len(named_hits) > 10 else ""
        raise ValueError(
            "Mais de um TIFF com nome de gold standard encontrado. "
            "Informe o arquivo exato para evitar ambiguidades:\n"
            f"{options}{suffix}"
        )

    keyword_hits = [
        c
        for c in candidates
        if any(token in c.name.lower() for token in ("gold", "gstd", "goldstd"))
    ]
    if len(keyword_hits) == 1:
        return keyword_hits[0]
    if len(keyword_hits) > 1:
        options = "\n".join(f" - {candidate}" for candidate in keyword_hits[:10])
        suffix = "\n - ..." if len(keyword_hits) > 10 else ""
        raise ValueError(
            "Mais de um TIFF candidato a gold standard encontrado por nome. "
            "Informe o arquivo exato para evitar ambiguidades:\n"
            f"{options}{suffix}"
        )

    if len(candidates) > 1:
        options = "\n".join(f" - {candidate}" for candidate in candidates[:10])
        suffix = "\n - ..." if len(candidates) > 10 else ""
        raise ValueError(
            "Mais de um TIFF encontrado para o gold standard. "
            "Informe o arquivo exato para evitar ambiguidades:\n"
            f"{options}{suffix}"
        )
    return candidates[0]


def parse_filter_list(raw_filters: Sequence[str]) -> list[str]:
    normalized = [item.strip().lower() for raw in raw_filters for item in raw.split(",") if item.strip()]
    if not normalized:
        return list(VALID_FILTERS)
    if any(item in {"all", "*", "todos"} for item in normalized):
        return list(VALID_FILTERS)

    parsed: list[str] = []
    for item in normalized:
        if item not in VALID_FILTERS:
            raise ValueError(f"Filtro invalido: {item}. Opcoes: {', '.join(VALID_FILTERS)}")
        if item not in parsed:
            parsed.append(item)
    if not parsed:
        raise ValueError("Nenhum filtro valido foi informado.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa filtragem + comparacao + segmentacao de forma automatizada.")
    parser.add_argument("gold", type=Path, help="Caminho do Gold Standard (.tif)")
    parser.add_argument("input", type=Path, help="Arquivo .tif ou diretorio com TIFFs para filtrar")
    parser.add_argument(
        "legacy_filters",
        nargs="*",
        help="(Legado) Filtros posicionais apos gold e input, ex: mean ou kuan lee.",
    )
    parser.add_argument("--filters", nargs="+", default=["all"], help="Lista de filtros (ex: kuan lee), CSV (ex: kuan,lee) ou 'all' para todos")
    parser.add_argument("--window-size", type=int, default=21, help="Tamanho da janela local para filtros")
    parser.add_argument("--noise-cv2", type=float, default=None, help="CV^2 do ruido. Se omitido, estima automaticamente para Kuan/Lee")
    parser.add_argument("--iter", type=int, default=60, help="Numero de iteracoes da segmentacao")
    parser.add_argument("--win-size", type=int, default=7, help="Janela do SSIM para comparacao")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True, help="Busca TIFFs recursivamente")
    parser.add_argument("--run-id", type=str, default=None, help="Identificador da execução para evitar sobrescrita")
    parser.add_argument("--with-octave", action=argparse.BooleanOptionalAction, default=True, help="Gera speckle com octave.py antes da filtragem")
    parser.add_argument("--octave-frequencies", nargs="+", type=float, default=[3.5], help="Frequências em MHz para gerar speckle")
    parser.add_argument("--octave-dr-db", type=float, default=45.0, help="Faixa dinâmica (dB) para geração de speckle")
    parser.add_argument("--octave-seed", type=int, help="Semente da geração de speckle")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/pipeline"), help="Diretorio raiz das saidas")
    args = parser.parse_args()
    # Compatibilidade: se o usuario informar filtros posicionais, eles prevalecem.
    if args.legacy_filters:
        args.filters = args.legacy_filters
    return args


def run_pipeline(args: argparse.Namespace) -> list[PipelineRun]:
    if not args.gold.exists():
        raise FileNotFoundError(f"Gold Standard nao encontrado: {args.gold}")
    if not args.input.exists():
        raise FileNotFoundError(f"Entrada nao encontrada: {args.input}")

    filters = parse_filter_list(args.filters)
    gold_file = resolve_gold_path(args.gold, recursive=args.recursive)
    print(f"Gold Standard resolvido: {gold_file}")
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_root = args.output_root / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    filter_input_path = args.input
    if args.with_octave:
        speckle_output_dir = output_root / "speckle"
        generated = generate_speckle_batch(
            gold_path=gold_file,
            output_dir=speckle_output_dir,
            frequencies_mhz=args.octave_frequencies,
            dr_db=args.octave_dr_db,
            seed=args.octave_seed,
            run_id=run_id,
            stem_prefix=gold_file.stem,
        )
        if not generated:
            raise RuntimeError("A geração de speckle não produziu arquivos.")
        filter_input_path = speckle_output_dir
        print(f"Speckle gerado em: {speckle_output_dir} ({len(generated)} arquivo(s))")

    all_runs: list[PipelineRun] = []

    for filter_name in filters:
        print("\n" + "=" * 100)
        print(f"PIPELINE - FILTRO: {filter_name.upper()}")
        print("=" * 100)

        filter_output_root = output_root / "filters"
        filtered_outputs = run_batch(
            input_path=filter_input_path,
            filter_name=filter_name,
            window_size=args.window_size,
            noise_cv2=args.noise_cv2,
            output_path=None,
            output_dir=filter_output_root,
            recursive=args.recursive,
        )

        filtered_dir = filter_output_root / filter_name
        compare_csv = output_root / "compare" / f"{filter_name}_metrics.csv"
        comparison_results = run_comparison(
            gold_path=gold_file,
            target_path=filtered_dir,
            recursive=args.recursive,
            win_size=args.win_size,
            output_csv=compare_csv,
        )

        segmentation_output_dir = output_root / "segmentation" / filter_name
        segmentation_results = run_segmentation(
            gold_path=gold_file,
            directory_path=filtered_dir,
            num_iter=args.iter,
            output_dir=segmentation_output_dir,
            recursive=args.recursive,
        )

        all_runs.append(
            PipelineRun(
                filter_name=filter_name,
                filtered_outputs=filtered_outputs,
                comparison_results=comparison_results,
                segmentation_results=segmentation_results,
                compare_csv=compare_csv,
                segmentation_output_dir=segmentation_output_dir,
            )
        )

    return all_runs


def print_summary(runs: Sequence[PipelineRun], output_root: Path) -> None:
    print("\n" + "=" * 100)
    print("RESUMO FINAL DO PIPELINE")
    print("=" * 100)
    for run in runs:
        print(
            f"Filtro {run.filter_name:<6} | "
            f"filtradas: {len(run.filtered_outputs):<4} | "
            f"comparacao: {len(run.comparison_results):<4} | "
            f"segmentacao: {len(run.segmentation_results):<4}"
        )
        print(f"  CSV comparacao : {run.compare_csv}")
        print(f"  Segmentation   : {run.segmentation_output_dir}")

        if run.comparison_results:
            rmse_rel = np.asarray([result.rmse_rel for result in run.comparison_results], dtype=np.float64)
            ssim_value = np.asarray([result.ssim_value for result in run.comparison_results], dtype=np.float64)
            usdsai = np.asarray([result.usdsai_value for result in run.comparison_results], dtype=np.float64)
            print(
                "  Compare mean±std: "
                f"RMSE%={rmse_rel.mean():.4f}±{rmse_rel.std(ddof=0):.4f}, "
                f"SSIM={ssim_value.mean():.4f}±{ssim_value.std(ddof=0):.4f}, "
                f"USDSAI={usdsai.mean():.4f}±{usdsai.std(ddof=0):.4f}"
            )

        if run.segmentation_results:
            tp = np.asarray([result.tp_pct for result in run.segmentation_results], dtype=np.float64)
            fp = np.asarray([result.fp_pct for result in run.segmentation_results], dtype=np.float64)
            ac = np.asarray([result.ac_pct for result in run.segmentation_results], dtype=np.float64)
            hd = np.asarray([result.hd for result in run.segmentation_results], dtype=np.float64)
            hm = np.asarray([result.hm for result in run.segmentation_results], dtype=np.float64)
            print(
                "  Segm mean±std   : "
                f"TP%={tp.mean():.4f}±{tp.std(ddof=0):.4f}, "
                f"FP%={fp.mean():.4f}±{fp.std(ddof=0):.4f}, "
                f"AC%={ac.mean():.4f}±{ac.std(ddof=0):.4f}, "
                f"HD={hd.mean():.4f}±{hd.std(ddof=0):.4f}, "
                f"HM={hm.mean():.4f}±{hm.std(ddof=0):.4f}"
            )
    print("-" * 100)
    print(f"Saidas em: {output_root}")


def write_summary_csv(runs: Sequence[PipelineRun], output_root: Path) -> Path:
    summary_csv = output_root / "summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "filter",
            "filtered_count",
            "comparison_count",
            "segmentation_count",
            "run_count_total",
            "comparison_total_count",
            "segmentation_total_count",
            "rmse_rel_mean",
            "rmse_rel_std",
            "ssim_mean",
            "ssim_std",
            "usdsai_mean",
            "usdsai_std",
            "tp_pct_mean",
            "tp_pct_std",
            "fp_pct_mean",
            "fp_pct_std",
            "ac_pct_mean",
            "ac_pct_std",
            "hd_mean",
            "hd_std",
            "hm_mean",
            "hm_std",
            "compare_csv",
            "segmentation_dir",
        ])

        for run in runs:
            run_total, comparison_total, segmentation_total = cumulative_counts(output_root, run.filter_name)
            rmse_rel_m, rmse_rel_s = _mean_std([result.rmse_rel for result in run.comparison_results])
            ssim_m, ssim_s = _mean_std([result.ssim_value for result in run.comparison_results])
            usdsai_m, usdsai_s = _mean_std([result.usdsai_value for result in run.comparison_results])

            tp_m, tp_s = _mean_std([result.tp_pct for result in run.segmentation_results])
            fp_m, fp_s = _mean_std([result.fp_pct for result in run.segmentation_results])
            ac_m, ac_s = _mean_std([result.ac_pct for result in run.segmentation_results])
            hd_m, hd_s = _mean_std([result.hd for result in run.segmentation_results])
            hm_m, hm_s = _mean_std([result.hm for result in run.segmentation_results])

            writer.writerow([
                run.filter_name,
                len(run.filtered_outputs),
                len(run.comparison_results),
                len(run.segmentation_results),
                run_total,
                comparison_total,
                segmentation_total,
                f"{rmse_rel_m:.6f}",
                f"{rmse_rel_s:.6f}",
                f"{ssim_m:.6f}",
                f"{ssim_s:.6f}",
                f"{usdsai_m:.6f}",
                f"{usdsai_s:.6f}",
                f"{tp_m:.6f}",
                f"{tp_s:.6f}",
                f"{fp_m:.6f}",
                f"{fp_s:.6f}",
                f"{ac_m:.6f}",
                f"{ac_s:.6f}",
                f"{hd_m:.6f}",
                f"{hd_s:.6f}",
                f"{hm_m:.6f}",
                f"{hm_s:.6f}",
                str(run.compare_csv),
                str(run.segmentation_output_dir),
            ])

    return summary_csv


def main() -> None:
    args = parse_args()
    runs = run_pipeline(args)
    print_summary(runs, args.output_root)
    summary_csv = write_summary_csv(runs, args.output_root)
    print(f"Resumo CSV salvo em: {summary_csv}")


if __name__ == "__main__":
    main()
