"""Aplicacao do filtro de Kuan para imagens TIFF.

Uso:
	python kuan.py caminho/da_imagem.tif

Opcionalmente, informe o caminho de saida:
	python kuan.py entrada.tif saida.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

try:
	from tifffile import imread as tif_read, imwrite as tif_write
except Exception:  # pragma: no cover - fallback for environments without tifffile
	tif_read = None
	tif_write = None

try:
	from PIL import Image
except Exception as exc:  # pragma: no cover - Pillow is expected to be available
	raise RuntimeError("Pillow is required to read and write TIFF images.") from exc


FilterFunc = Callable[[np.ndarray, int, Optional[float]], np.ndarray]
AVAILABLE_FILTERS = ("kuan", "lee", "mean", "median")


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


def iter_tiff_paths(path: Path, recursive: bool = True) -> Iterable[Path]:
	if path.is_file():
		yield path
		return

	if path.is_dir():
		candidates = path.glob("**/*") if recursive else path.iterdir()
		for candidate in sorted(candidates):
			if candidate.is_file() and candidate.suffix.lower() in {".tif", ".tiff"}:
				yield candidate
		return

	raise FileNotFoundError(f"Arquivo ou diretorio nao encontrado: {path}")


def box_filter(array: np.ndarray, window_size: int) -> np.ndarray:
	if window_size < 1 or window_size % 2 == 0:
		raise ValueError("window_size must be an odd positive integer.")

	pad = window_size // 2
	padded = np.pad(array, pad_width=pad, mode="reflect")
	integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)

	top_left = integral[:-window_size, :-window_size]
	top_right = integral[:-window_size, window_size:]
	bottom_left = integral[window_size:, :-window_size]
	bottom_right = integral[window_size:, window_size:]

	return (bottom_right - top_right - bottom_left + top_left) / float(window_size * window_size)


def median_filter_2d(image: np.ndarray, window_size: int) -> np.ndarray:
	if window_size < 1 or window_size % 2 == 0:
		raise ValueError("window_size must be an odd positive integer.")

	pad = window_size // 2
	padded = np.pad(image.astype(np.float64, copy=False), pad_width=pad, mode="reflect")
	windows = np.lib.stride_tricks.sliding_window_view(padded, (window_size, window_size))
	return np.median(windows, axis=(-2, -1))


def estimate_noise_cv2(local_mean: np.ndarray, local_var: np.ndarray) -> float:
    # 1. Calcula o CV^2 local evitando divisão por zero
    mean_sq = np.square(local_mean)
    cv2 = local_var / (mean_sq + 1e-12)
    
    # 2. Encontramos o valor máximo da média local para criar limiares
    max_mean = np.max(local_mean)
    
    # 3. Criamos uma máscara para selecionar APENAS o tecido de fundo (cinza)
    #    - Ignoramos pixels abaixo de 15% do máximo (cistos pretos/regiões anecóicas)
    #    - Ignoramos pixels acima de 85% do máximo (cistos muito brilhantes/hiperecóicos)
    tissue_mask = (local_mean > 0.15 * max_mean) & (local_mean < 0.85 * max_mean)
    
    # Filtramos os valores de CV^2 válidos dentro da região de tecido
    valid = cv2[tissue_mask & np.isfinite(cv2) & (cv2 > 1e-6)]
    
    if valid.size == 0:
        # Fallback seguro para imagens médicas log-comprimidas de 8-bit
        return 0.05 
        
    # 4. Em vez de pegar os menores valores (lowest), extraímos a MEDIANA do tecido.
    #    A mediana representa estatisticamente o comportamento médio do speckle na região homogênea.
    estimated_cv2 = float(np.median(valid))
    
    print(f"-> Região de tecido isolada: {valid.size} pixels analisados.")
    print(f"-> Estimativa automática de CV^2 do ruído: {estimated_cv2:.6f}")
    
    return estimated_cv2


def kuan_filter_2d(image: np.ndarray, window_size: int = 5, noise_cv2: Optional[float] = None) -> np.ndarray:
	image = image.astype(np.float64, copy=False)

	local_mean = box_filter(image, window_size)
	local_mean_sq = box_filter(np.square(image), window_size)
	local_var = np.maximum(local_mean_sq - np.square(local_mean), 0.0)

	if noise_cv2 is None:
		noise_cv2 = estimate_noise_cv2(local_mean, local_var)

	denominator = np.maximum(local_var / (np.square(local_mean) + 1e-12), 1e-12)
	weight = (1.0 - (noise_cv2 / denominator)) / (1.0 + noise_cv2)
	weight = np.clip(weight, 0.0, 1.0)

	return local_mean + weight * (image - local_mean)


def lee_filter_2d(image: np.ndarray, window_size: int = 5, noise_cv2: Optional[float] = None) -> np.ndarray:
	image = image.astype(np.float64, copy=False)

	local_mean = box_filter(image, window_size)
	local_mean_sq = box_filter(np.square(image), window_size)
	local_var = np.maximum(local_mean_sq - np.square(local_mean), 0.0)
	local_cv2 = local_var / (np.square(local_mean) + 1e-12)

	if noise_cv2 is None:
		noise_cv2 = estimate_noise_cv2(local_mean, local_var)

	weight = 1.0 - (noise_cv2 / np.maximum(local_cv2, 1e-12))
	weight = np.clip(weight, 0.0, 1.0)
	return local_mean + weight * (image - local_mean)


def mean_filter_2d(image: np.ndarray, window_size: int = 5, noise_cv2: Optional[float] = None) -> np.ndarray:
	_ = noise_cv2
	return box_filter(image.astype(np.float64, copy=False), window_size)


def median_filter_image_2d(image: np.ndarray, window_size: int = 5, noise_cv2: Optional[float] = None) -> np.ndarray:
	_ = noise_cv2
	return median_filter_2d(image, window_size)


def apply_per_channel(image: np.ndarray, filter_2d: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
	if image.ndim == 2:
		return filter_2d(image)

	if image.ndim == 3:
		channels = [filter_2d(image[..., index]) for index in range(image.shape[-1])]
		return np.stack(channels, axis=-1)

	raise ValueError("Only 2D grayscale images or 3D channel-last images are supported.")


def cast_like(reference: np.ndarray, filtered: np.ndarray) -> np.ndarray:
	if np.issubdtype(reference.dtype, np.integer):
		info = np.iinfo(reference.dtype)
		return np.clip(np.rint(filtered), info.min, info.max).astype(reference.dtype)

	return filtered.astype(reference.dtype, copy=False)


def default_output_path(input_path: Path, filter_name: str, output_dir: Path) -> Path:
	return output_dir / filter_name / f"{input_path.stem}_{filter_name}{input_path.suffix}"


def is_tiff_file_path(path: Path) -> bool:
	return path.suffix.lower() in {".tif", ".tiff"}


def resolve_output_path(input_path: Path, filter_name: str, output_path: Optional[Path], output_dir: Path) -> Path:
	if output_path is None:
		return default_output_path(input_path, filter_name, output_dir)

	if input_path.is_file() and not is_tiff_file_path(output_path):
		return default_output_path(input_path, filter_name, output_path)

	return output_path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Aplica filtros locais a uma imagem TIFF.")
	parser.add_argument("input", type=Path, help="Caminho do arquivo .tif de entrada ou de um diretorio com TIFFs")
	parser.add_argument("output", nargs="?", type=Path, help="(Legado) Caminho da imagem ou diretorio de saida.")
	parser.add_argument("legacy_filter", nargs="?", choices=AVAILABLE_FILTERS, help="(Legado) Filtro a ser aplicado.")
	parser.add_argument("--filter", dest="filter_name", choices=AVAILABLE_FILTERS, default=None, help="Filtro a ser aplicado (preferencial).")
	parser.add_argument("--window-size", type=int, default=21, help="Tamanho da janela local (impar, padrao: 21)")
	parser.add_argument("--noise-cv2", type=float, default=None, help="Coeficiente de variacao ao quadrado do ruido. Se omitido, sera estimado automaticamente para Kuan/Lee.")
	parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True, help="Busca TIFFs recursivamente quando a entrada for um diretorio.")
	parser.add_argument("--output-dir", type=Path, default=Path("outputs/filters"), help="Diretorio base para saidas organizadas.")
	parser.add_argument("--compare-lee", action="store_true", help="Gera tambem uma saida com o filtro de Lee para comparacao.")
	args = parser.parse_args()
	args.filter = args.filter_name or args.legacy_filter or "kuan"
	return args


def build_filter_function(filter_name: str, window_size: int, noise_cv2: Optional[float]) -> Callable[[np.ndarray], np.ndarray]:
	if filter_name == "kuan":
		return lambda image: apply_per_channel(image, lambda channel: kuan_filter_2d(channel, window_size=window_size, noise_cv2=noise_cv2))

	if filter_name == "lee":
		return lambda image: apply_per_channel(image, lambda channel: lee_filter_2d(channel, window_size=window_size, noise_cv2=noise_cv2))

	if filter_name == "mean":
		return lambda image: apply_per_channel(image, lambda channel: mean_filter_2d(channel, window_size=window_size))

	if filter_name == "median":
		return lambda image: apply_per_channel(image, lambda channel: median_filter_image_2d(channel, window_size=window_size))

	raise ValueError(f"Filtro desconhecido: {filter_name}")


def run_filter(input_path: Path, filter_name: str, window_size: int, noise_cv2: Optional[float], output_path: Optional[Path], output_dir: Path) -> Path:
	image = load_tiff(input_path)
	filter_function = build_filter_function(filter_name, window_size, noise_cv2)
	filtered = filter_function(image)
	output = cast_like(image, filtered)

	final_output = resolve_output_path(input_path, filter_name, output_path, output_dir)
	save_tiff(final_output, output)
	return final_output


def run_batch(
	input_path: Path,
	filter_name: str,
	window_size: int,
	noise_cv2: Optional[float],
	output_path: Optional[Path],
	output_dir: Path,
	recursive: bool = True,
) -> list[Path]:
	if input_path.is_file():
		return [run_filter(input_path, filter_name, window_size, noise_cv2, output_path, output_dir)]

	base_output_dir = output_path if output_path is not None and not is_tiff_file_path(output_path) else output_dir
	outputs: list[Path] = []
	for image_path in iter_tiff_paths(input_path, recursive=recursive):
		outputs.append(run_filter(image_path, filter_name, window_size, noise_cv2, None, base_output_dir))
	return outputs


def main() -> None:
	args = parse_args()

	input_path = args.input
	if not input_path.exists():
		raise FileNotFoundError(f"Arquivo ou diretorio nao encontrado: {input_path}")

	run_batch(
		input_path=input_path,
		filter_name=args.filter,
		window_size=args.window_size,
		noise_cv2=args.noise_cv2,
		output_path=args.output,
		output_dir=args.output_dir,
		recursive=args.recursive,
	)

	if args.compare_lee and args.filter == "kuan":
		run_batch(
			input_path=input_path,
			filter_name="lee",
			window_size=args.window_size,
			noise_cv2=args.noise_cv2,
			output_path=None,
			output_dir=args.output_dir,
			recursive=args.recursive,
		)


if __name__ == "__main__":
	main()
