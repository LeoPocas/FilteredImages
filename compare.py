"""
Compara um gold standard com um diretório de imagens filtradas.
Suporta validação direta do próprio Gold Standard e a métrica USDSAI.
"""

from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim

try:
    from tifffile import imread as tif_read
except Exception:
    tif_read = None


@dataclass(frozen=True)
class ComparisonResult:
    path: Path
    group: str
    mean_val: float          # Média na região homogênea (mu)
    std_val: float           # Desvio padrão na região homogênea (sigma)
    rmse_abs: float          # RMSE absoluto
    rmse_rel: float          # RMSE relativo (%)
    ssim_value: float        # SSIM x 100
    mean_shift: float
    usdsai_value: float      # Métrica USDSAI adicionada


def load_tiff(path: Path) -> np.ndarray:
    if tif_read is not None:
        return np.asarray(tif_read(str(path)))
    with Image.open(path) as image:
        return np.asarray(image)


def iter_tiff_paths(directory: Path, recursive: bool) -> Iterable[Path]:
    if directory.is_file():
        yield directory
        return
    pattern = "**/*" if recursive else "*"
    for path in sorted(directory.glob(pattern)):
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
            yield path


def infer_group(root: Path, path: Path) -> str:
    if root.is_file():
        return "Self-Evaluation"
    try:
        relative = path.relative_to(root)
    except ValueError:
        return path.parent.name
    if len(relative.parts) <= 1:
        return root.name
    return relative.parts[0]


def get_homogeneous_region(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    # Retângulo vertical central no topo (entre as colunas de cistos)
    ymin, ymax = int(h * 0.05), int(h * 0.25)
    xmin, xmax = int(w * 0.45), int(w * 0.55)
    return img[ymin:ymax, xmin:xmax]


def compute_usdsai(gold_std: np.ndarray, filtered: np.ndarray) -> float:
    """
    Calcula o USDSAI usando Desvio Padrão e uma ROI espessa.
    Simula as caixas de seleção manuais utilizadas em artigos científicos.
    """
    filtered_f = filtered.astype(np.float64)
    
    # 1. Encontra a localização da borda (1 pixel)
    mask_bool = (gold_std < 110) | (gold_std > 145)
    kernel_thin = np.ones((3, 3), dtype=np.uint8)
    edge_thin = cv2.morphologyEx(mask_bool.astype(np.uint8), cv2.MORPH_GRADIENT, kernel_thin) > 0
    
    # 2. Dilata a borda para transformá-la em uma "faixa" grossa (~11 pixels)
    # Isso simula o retângulo de ROI do artigo, englobando tecido e cisto vizinho
    kernel_thick = np.ones((11, 11), dtype=np.uint8)
    mask_edges_thick = cv2.dilate(edge_thin.astype(np.uint8), kernel_thick, iterations=1) > 0
    
    if np.sum(mask_edges_thick) == 0:
        return 1.0

    # 3. Extrai pixels da região homogênea e da "faixa" de borda
    region_homog = get_homogeneous_region(filtered_f)
    pixels_edge = filtered_f[mask_edges_thick]
    
    # 4. Usa o Desvio Padrão (linear) ao invés da Variância (quadrática)
    std_homog = np.std(region_homog)
    std_edge = np.std(pixels_edge)
    
    # Proteção contra divisão por zero em autoavaliação perfeita
    if std_homog < 1e-4:
        if np.array_equal(gold_std, filtered):
            return 1.0
        return float(std_edge / 1e-4)
        
    return float(std_edge / std_homog)


def relative_rmse(gold_std: np.ndarray, filtered: np.ndarray) -> float:
    gold_f = gold_std.astype(np.float64)
    filtered_f = filtered.astype(np.float64)
    
    numerator = np.sum(np.square(filtered_f - gold_f))
    denominator = np.sum(np.square(gold_f))
    if denominator == 0:
        return 0.0
    return float(np.sqrt(numerator / denominator) * 100.0)


def compute_ssim(gold_std: np.ndarray, filtered: np.ndarray, win_size: int) -> float:
    data_range = float(np.max(gold_std) - np.min(gold_std))
    if data_range == 0:
        data_range = 255.0
    return float(ssim(gold_std, filtered, data_range=data_range, win_size=win_size)) * 100.0


def compare_images(gold_std: np.ndarray, filtered: np.ndarray, win_size: int) -> tuple:
    if gold_std.shape != filtered.shape:
        filtered = cv2.resize(filtered, (gold_std.shape[1], gold_std.shape[0]))

    region = get_homogeneous_region(filtered)
    mu = float(np.mean(region))
    sigma = float(np.std(region))

    rmse_abs = float(np.sqrt(mse(gold_std, filtered)))
    rmse_rel = relative_rmse(gold_std, filtered)
    ssim_value = compute_ssim(gold_std, filtered, win_size=win_size)
    mean_shift = float(gold_std.mean() - filtered.mean())
    usdsai_value = compute_usdsai(gold_std, filtered)

    return mu, sigma, rmse_abs, rmse_rel, ssim_value, mean_shift, usdsai_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara um gold standard com imagens filtradas.")
    parser.add_argument("gold", type=Path, help="Caminho do gold standard (.tif)")
    parser.add_argument("directory", type=Path, help="Diretório com as imagens ou o próprio Gold Standard para teste")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True, help="Busca recursiva")
    parser.add_argument("--win-size", type=int, default=7, help="Janela do SSIM")
    parser.add_argument("--output-csv", type=Path, default=None, help="Salva resultados em CSV")
    return parser.parse_args()


def summarize(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def write_csv(path: Path, results: Sequence[ComparisonResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "path",
            "group",
            "mean_val",
            "std_val",
            "rmse_abs",
            "rmse_rel",
            "ssim_value",
            "mean_shift",
            "usdsai_value",
        ])
        for result in results:
            writer.writerow([
                str(result.path),
                result.group,
                f"{result.mean_val:.6f}",
                f"{result.std_val:.6f}",
                f"{result.rmse_abs:.6f}",
                f"{result.rmse_rel:.6f}",
                f"{result.ssim_value:.6f}",
                f"{result.mean_shift:.6f}",
                f"{result.usdsai_value:.6f}",
            ])


def run_comparison(
    gold_path: Path,
    target_path: Path,
    recursive: bool = True,
    win_size: int = 7,
    output_csv: Path | None = None,
) -> list[ComparisonResult]:
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold standard não encontrado: {gold_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Alvo de comparação não encontrado: {target_path}")

    gold_std = load_tiff(gold_path)
    image_paths = list(iter_tiff_paths(target_path, recursive=recursive))

    if not image_paths:
        raise FileNotFoundError("Nenhuma imagem TIFF encontrada.")

    results: list[ComparisonResult] = []
    for image_path in image_paths:
        filtered = load_tiff(image_path)
        mu, sigma, rmse_abs, rmse_rel, ssim_val, mean_shift, usdsai_val = compare_images(
            gold_std,
            filtered,
            win_size=win_size,
        )
        results.append(
            ComparisonResult(
                path=image_path,
                group=infer_group(target_path, image_path),
                mean_val=mu,
                std_val=sigma,
                rmse_abs=rmse_abs,
                rmse_rel=rmse_rel,
                ssim_value=ssim_val,
                mean_shift=mean_shift,
                usdsai_value=usdsai_val,
            )
        )

    if output_csv is not None:
        write_csv(output_csv, results)

    return results


def main() -> None:
    args = parse_args()
    results = run_comparison(
        gold_path=args.gold,
        target_path=args.directory,
        recursive=args.recursive,
        win_size=args.win_size,
        output_csv=args.output_csv,
    )

    print("\n" + "="*95)
    print(f" RELATÓRIO DE COMPARAÇÃO DE MÉTRICAS (Padrão Cardoso et al. 2012)")
    print("="*95)
    print(f"Gold Standard Referência : {args.gold.name}")
    print(f"Total de Arquivos        : {len(results)}")
    print("-"*95)
    print(f"{'Grupo / Arquivo':<35} | {'Média (mu)':<10} | {'Desvio (sig)':<12} | {'RMSE %':<10} | {'SSIM x 100':<10} | {'USDSAI':<8}")
    print("-"*95)
    
    for r in results:
        name = r.path.name if len(r.path.name) <= 34 else r.path.name[:21] + "..."
        print(f"{name:<35} | {r.mean_val:<10.4f} | {r.std_val:<12.4f} | {r.rmse_rel:<10.4f}% | {r.ssim_value:<10.4f} | {r.usdsai_value:<8.4f}")
    
    print("="*95)
    
    # Resumos Estatísticos
    mu_m, mu_s = summarize([r.mean_val for r in results])
    sig_m, sig_s = summarize([r.std_val for r in results])
    rel_m, rel_s = summarize([r.rmse_rel for r in results])
    ssim_m, ssim_s = summarize([r.ssim_value for r in results])
    usd_m, usd_s = summarize([r.usdsai_value for r in results])
    
    print(f"MÉDIA GLOBAL DO CONJUNTO:")
    print(f" -> Média (mu)      : {mu_m:.2f} ± {mu_s:.2f}  (Esperado p/ Lee: 57.89 ± 2.16)")
    print(f" -> Desvio (sigma)  : {sig_m:.2f} ± {sig_s:.2f}  (Esperado p/ Lee: 2.79  ± 0.91)")
    print(f" -> RMSE Rel (Eq 36): {rel_m:.2f}% ± {rel_s:.2f}% (Esperado p/ Lee: 13.46% ± 1.55%)")
    print(f" -> SSIM (%)        : {ssim_m:.2f}% ± {ssim_s:.2f}% (Esperado p/ Lee: 89.00% ± 2.28%)")
    print(f" -> USDSAI          : {usd_m:.4f} ± {usd_s:.4f}")
    print("="*95 + "\n")

    if args.output_csv is not None:
        print(f"CSV salvo em: {args.output_csv}")


if __name__ == "__main__":
    main()