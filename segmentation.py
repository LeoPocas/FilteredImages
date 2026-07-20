"""
Realiza a segmentação por Level Set (Chan-Vese) nas imagens filtradas,
calcula as métricas funcionais (TP%, FP%, AC%, HD e HM) e salva as máscaras.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import cv2
from PIL import Image
from skimage.segmentation import morphological_chan_vese
from scipy.spatial.distance import directed_hausdorff

try:
    from tifffile import imread as tif_read, imwrite as tif_write
except Exception:
    tif_read = None
    tif_write = None


@dataclass(frozen=True)
class SegmentationResult:
    path: Path
    tp_pct: float
    fp_pct: float
    ac_pct: float
    hd: float
    hm: float


def load_tiff(path: Path) -> np.ndarray:
    if tif_read is not None:
        return np.asarray(tif_read(str(path)))
    with Image.open(path) as image:
        return np.asarray(image)


def save_tiff(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tif_write is not None:
        tif_write(str(path), array)
        return
    image = Image.fromarray(array)
    image.save(path, format="TIFF")


def iter_tiff_paths(directory: Path, recursive: bool = True) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for path in sorted(directory.glob(pattern)):
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
            yield path


def create_local_init_mask(shape: tuple[int, int]) -> np.ndarray:
    """
    Cria uma máscara inicial contendo pequenos discos circulares
    nas regiões aproximadas onde estão os cistos (conforme a imagem cistosGoldStd).
    """
    h, w = shape
    init_mask_uint8 = np.zeros((h, w), dtype=np.uint8)
    
    centros = [
        # Coluna da esquerda (hiperecóicos)
        (int(h * 0.14), int(w * 0.38)),
        (int(h * 0.37), int(w * 0.38)),
        (int(h * 0.52), int(w * 0.38)),
        (int(h * 0.68), int(w * 0.38)),
        (int(h * 0.82), int(w * 0.38)),
        # Coluna da direita (anecóicos)
        (int(h * 0.14), int(w * 0.75)),
        (int(h * 0.37), int(w * 0.75)),
        (int(h * 0.52), int(w * 0.75)),
        (int(h * 0.68), int(w * 0.75)),
        (int(h * 0.82), int(w * 0.75)),
    ]
    
    for cy, cx in centros:
        cv2.circle(init_mask_uint8, (cx, cy), 15, 1, -1)
        
    return init_mask_uint8 > 0


def compute_hausdorff_metrics(mask_gt: np.ndarray, mask_seg: np.ndarray) -> tuple[float, float]:
    """
    Calcula a Distância de Hausdorff (HD) e a Distância de Hausdorff Média (HM)
    entre as BORDAS (contornos) de duas máscaras binárias.
    """
    # 1. Extraímos estritamente as bordas (contornos de 1 pixel de espessura)
    # usando um gradiente morfológico (Dilação - Erosão)
    kernel = np.ones((3, 3), dtype=np.uint8)
    edge_gt = cv2.morphologyEx(mask_gt.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    edge_seg = cv2.morphologyEx(mask_seg.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    
    # Extrai as coordenadas (y, x) apenas dos pixels de borda
    pts_gt = np.argwhere(edge_gt)
    pts_seg = np.argwhere(edge_seg)
    
    # Casos de máscaras vazias
    if len(pts_gt) == 0 or len(pts_seg) == 0:
        return 0.0, 0.0

    # 2. Distância de Hausdorff Direcional Simétrica (HD)
    d1 = directed_hausdorff(pts_gt, pts_seg)[0]
    d2 = directed_hausdorff(pts_seg, pts_gt)[0]
    hd = max(d1, d2)
    
    # 3. Distância de Hausdorff Média (HM) sem subamostragem aleatória perigosa
    from scipy.spatial.distance import cdist
    try:
        # Como estamos usando apenas as bordas, o número de pontos é pequeno
        # e podemos calcular a matriz de distância completa com total segurança
        dist_matrix = cdist(pts_gt, pts_seg, metric='euclidean')
        
        # Distância média do GT para o segmento e do segmento para o GT
        mean_gt_to_seg = np.mean(np.min(dist_matrix, axis=1))
        mean_seg_to_gt = np.mean(np.min(dist_matrix, axis=0))
        hm = (mean_gt_to_seg + mean_seg_to_gt) / 2.0
    except Exception:
        hm = hd / 2.0
        
    return float(hd), float(hm)


def run_level_set_segmentation(
    img_filtered: np.ndarray, 
    img_gold_std: np.ndarray, 
    num_iter: int = 60
) -> tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    """
    Aplica o Level Set (Chan-Vese) e calcula TP%, FP%, AC%, HD e HM.
    """
    # 1. Definir a Máscara de Referência Real (Ground Truth)
    ref_mask = (img_gold_std < 110) | (img_gold_std > 145)
    
    # 2. Mapeamento de 3 Fases para 2 Fases (Distância até o fundo de 128)
    img_prep = np.abs(img_filtered.astype(np.float64) - 128.0)
    
    # 3. Inicialização Localizada
    init_mask = create_local_init_mask(img_filtered.shape)
    
    # 4. Evolução da Curva (Morphological Chan-Vese)
    segmented_mask = morphological_chan_vese(
        img_prep, 
        num_iter=num_iter, 
        init_level_set=init_mask, 
        smoothing=0, 
        lambda1=1.0, 
        lambda2=1.0
    )
    
    # 5. Cálculo das Métricas de Área (TP%, FP%, AC%)
    gs_pixels = np.sum(ref_mask)
    overlap = np.sum(segmented_mask & ref_mask)
    excess = np.sum(segmented_mask & ~ref_mask)
    
    tp_pct = (overlap / gs_pixels) * 100.0 if gs_pixels > 0 else 0.0
    fp_pct = (excess / gs_pixels) * 100.0 if gs_pixels > 0 else 0.0
    ac_pct = (tp_pct + (100.0 - fp_pct)) / 2.0
    
    # 6. Cálculo das Métricas de Borda (HD e HM)
    hd, hm = compute_hausdorff_metrics(ref_mask, segmented_mask)
    
    return tp_pct, fp_pct, ac_pct, hd, hm, ref_mask, segmented_mask


def generate_overlay(img_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(
        mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img_color, contours, -1, (0, 0, 255), 1)
    return img_color


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validação funcional por segmentação Level Set com HD/HM.")
    parser.add_argument("gold", type=Path, help="Caminho do Gold Standard (.tif)")
    parser.add_argument("directory", type=Path, help="Diretório contendo os arquivos filtrados")
    parser.add_argument("--iter", type=int, default=60, help="Número de iterações do Level Set (padrão: 60)")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True, help="Busca TIFFs recursivamente no diretório de entrada.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/segmentation"), help="Diretório para salvar as máscaras")
    return parser.parse_args()


def run_segmentation(
    gold_path: Path,
    directory_path: Path,
    num_iter: int = 60,
    output_dir: Path = Path("outputs/segmentation"),
    recursive: bool = True,
) -> list[SegmentationResult]:
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold Standard não encontrado: {gold_path}")
    if not directory_path.exists() or not directory_path.is_dir():
        raise FileNotFoundError(f"Diretório não encontrado: {directory_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    gold_std = load_tiff(gold_path)
    image_paths = list(iter_tiff_paths(directory_path, recursive=recursive))

    if not image_paths:
        return []

    gt_saved = False
    results: list[SegmentationResult] = []

    for img_path in image_paths:
        filtered = load_tiff(img_path)

        if filtered.shape != gold_std.shape:
            filtered = cv2.resize(filtered, (gold_std.shape[1], gold_std.shape[0]))

        tp, fp, ac, hd, hm, ref_mask, seg_mask = run_level_set_segmentation(filtered, gold_std, num_iter=num_iter)

        if not gt_saved:
            gt_mask_uint8 = (ref_mask.astype(np.float64) * 255).astype(np.uint8)
            save_tiff(output_dir / "ground_truth_mask.tif", gt_mask_uint8)
            gt_saved = True

        seg_mask_uint8 = (seg_mask.astype(np.float64) * 255).astype(np.uint8)
        save_tiff(output_dir / f"mask_{img_path.stem}.tif", seg_mask_uint8)

        overlay_img = generate_overlay(filtered, seg_mask_uint8)
        cv2.imwrite(str(output_dir / f"overlay_{img_path.stem}.tif"), overlay_img)

        results.append(
            SegmentationResult(
                path=img_path,
                tp_pct=tp,
                fp_pct=fp,
                ac_pct=ac,
                hd=hd,
                hm=hm,
            )
        )

    return results


def main() -> None:
    args = parse_args()
    results = run_segmentation(
        gold_path=args.gold,
        directory_path=args.directory,
        num_iter=args.iter,
        output_dir=args.output_dir,
        recursive=args.recursive,
    )

    if not results:
        print("Nenhuma imagem filtrada encontrada.")
        return

    print("\n" + "="*110)
    print(" RELATÓRIO DE VALIDAÇÃO FUNCIONAL POR SEGMENTAÇÃO (Métricas Completas / Tabela 5)")
    print("="*110)
    print(f"Gold Standard Referência : {args.gold.name}")
    print(f"Iterações do Level Set   : {args.iter}")
    print(f"Pasta de salvamento      : {args.output_dir}/")
    print("-"*110)
    print(f"{'Arquivo Filtrado':<35} | {'TP%':<9} | {'FP%':<9} | {'AC%':<9} | {'HD (px)':<10} | {'HM (px)':<10}")
    print("-"*110)

    for result in results:
        name = result.path.name if len(result.path.name) <= 34 else result.path.name[:31] + "..."
        print(f"{name:<35} | {result.tp_pct:<7.2f}% | {result.fp_pct:<7.2f}% | {result.ac_pct:<7.2f}% | {result.hd:<8.2f} | {result.hm:<8.2f}")

    print("="*110 + "\n")


if __name__ == "__main__":
    main()