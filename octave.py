from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from scipy.signal import hilbert

try:
    from tifffile import imread as tif_read, imwrite as tif_write
except Exception:  # pragma: no cover - fallback for environments without tifffile
    tif_read = None
    tif_write = None

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover - Pillow is expected to be available
    raise RuntimeError("Pillow is required to read and write TIFF images.") from exc

# =====================================================================

class UltrasoundSpeckleSimulator:
    def __init__(self, seed=42):
        # Define a semente aleatória para garantir a reprodutibilidade dos scatterers
        np.random.seed(seed)
        
    def generate_speckle(self, img_gold_std, frequency_mhz=3.5, dr_db=45.0):
        """
        Simula o ruído de speckle físico baseado na modelagem do transdutor.
        
        Parâmetros:
        - img_gold_std: imagem original em escala de cinza (numpy array uint8)
        - frequency_mhz: 3.5 (Sim1 original) ou 7.0 (Sim2 ou variação de frequência)
        - dr_db: Faixa dinâmica de compressão em decibéis (tipicamente entre 40 e 50 dB)
        """
        # 1. Normalização da anatomia de entrada [0.0, 1.0]
        img_ref = img_gold_std.astype(np.float64) / 255.0
        
        # 2. Geração do Mapa de Espalhadores (Scatterers) Coerentes
        # Espalhadores distribuídos uniformemente com amplitude modulada pela anatomia
        scatterers = (np.random.rand(*img_ref.shape) - 0.5) * img_ref
        
        # 3. Modelagem da Resposta Acústica do Transdutor (PSF)
        # Ajustamos as resoluções axiais e laterais com base na frequência física
        if frequency_mhz == 3.5:
            sigma_x = 4.2      # Resolução lateral (feixe mais largo horizontalmente)
            sigma_y = 1.5      # Resolução axial (comprimento do pulso)
            f_central = 0.35   # Frequência central de oscilação na grade discreta
            kernel_size = 15   # Tamanho da grade da PSF (-15 a 15)
        elif frequency_mhz == 7.0:
            sigma_x = 2.1      # Frequência maior = melhor resolução lateral (grãos menores)
            sigma_y = 0.75     # Melhor resolução axial (pulso mais curto)
            f_central = 0.70   # Frequência de oscilação dobrada
            kernel_size = 8    # Kernel menor para representar maior foco espacial
        else:
            raise ValueError("Frequências suportadas nesta calibração: 3.5 MHz ou 7.0 MHz.")

        # Criação da grade espacial para a PSF
        x = np.arange(-kernel_size, kernel_size + 1)
        y = np.arange(-kernel_size, kernel_size + 1)
        X, Y = np.meshgrid(x, y)
        
        # Pulso acústico de RF (Gabor wavelet)
        psf_rf = np.exp(-(X**2 / (2 * sigma_x**2) + Y**2 / (2 * sigma_y**2))) * np.cos(2 * np.pi * f_central * Y)
        psf_rf /= np.sum(np.abs(psf_rf)) # Normaliza para preservar energia
        
        # 4. Convolução 2D (Interferência de fase no meio acústico)
        sinal_rf = cv2.filter2D(scatterers, -1, psf_rf, borderType=cv2.BORDER_REFLECT)
        
        # 5. Detecção de Envelope (Transformada de Hilbert axialmente)
        sinal_analitico = hilbert(sinal_rf, axis=0)
        envelope = np.abs(sinal_analitico)
        
        # 6. Compressão Logarítmica (B-Mode processing)
        img_log = 20 * np.log10(envelope + 1e-4) # Constante pequena para evitar log(0)
        img_log -= np.max(img_log)               # Normaliza o maior pico em 0 dB
        
        # Mapeamento linear de [-DR, 0] dB para [0, 255] em ponto flutuante
        img_ruido_raw = 255 * (img_log + dr_db) / dr_db
        img_ruido_raw[img_log < -dr_db] = 0      # Clipa valores abaixo da faixa dinâmica
        
# =====================================================================
        # PASSO 2: CALIBRAÇÃO VIA CORREÇÃO GAMMA ANCORADA (CORRIGIDO)
        # =====================================================================
        # 1. Normalização Min-Max para restaurar o contraste total [0, 1]
        img_min = np.min(img_ruido_raw)
        img_max = np.max(img_ruido_raw)
        img_normalized = (img_ruido_raw - img_min) / (img_max - img_min + 1e-12)
        
        # 2. Isolamos a máscara do tecido de fundo (fundo cinza do Gold Standard original)
        mascara_tecido = (img_gold_std > 110) & (img_gold_std < 145)
        if np.sum(mascara_tecido) == 0:
            mascara_tecido = np.ones_like(img_gold_std, dtype=bool)
            
        # 3. Medimos onde a média do tecido foi parar na escala temporária [0, 1]
        # Devido à compressão logarítmica, ela costuma subir para perto de 0.62 (muito clara)
        media_tecido_norm = np.mean(img_normalized[mascara_tecido])
        
        # 4. Calculamos o Gamma necessário para puxar essa média de volta para exatamente 0.5 (128/255)
        # Equação: media_tecido_norm ^ gamma = 0.5  =>  gamma = ln(0.5) / ln(media_tecido_norm)
        gamma = np.log(0.5) / np.log(media_tecido_norm + 1e-12)
        gamma = np.clip(gamma, 0.5, 3.0) # Limite de segurança para evitar distorções extremas
        
        # 5. Aplicamos a curva Gamma e convertemos de volta para 8-bits [0, 255]
        img_calibrada = np.power(img_normalized, gamma) * 255.0
        
        # Retorna a imagem final clipada no intervalo uint8 [0, 255]
        return np.clip(img_calibrada, 0, 255).astype(np.uint8)


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


def normalize_freq_token(frequency_mhz: float) -> str:
    return f"{frequency_mhz:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def generate_speckle_batch(
    gold_path: Path,
    output_dir: Path,
    frequencies_mhz: Sequence[float] = (3.5, 7.0),
    dr_db: float = 45.0,
    seed: int = np.random.SeedSequence().entropy,
    run_id: str = "run",
    stem_prefix: str = "speckle_fieldii",
) -> list[Path]:
    if not gold_path.exists() or not gold_path.is_file():
        raise FileNotFoundError(f"Gold standard não encontrado: {gold_path}")

    img_gold = load_tiff(gold_path)
    simulator = UltrasoundSpeckleSimulator(seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for index, frequency_mhz in enumerate(frequencies_mhz, start=1):
        speckled = simulator.generate_speckle(
            img_gold,
            frequency_mhz=frequency_mhz,
            dr_db=dr_db,
        )
        freq_token = normalize_freq_token(frequency_mhz)
        filename = f"{stem_prefix}_{run_id}_f{freq_token}MHz_n{index:02d}.tif"
        output_path = output_dir / filename
        save_tiff(output_path, speckled)
        outputs.append(output_path)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera imagens com speckle físico no estilo Field II.")
    parser.add_argument("gold", type=Path, help="Caminho do Gold Standard (.tif)")
    parser.add_argument("--frequencies", nargs="+", type=float, default=[3.5, 7.0], help="Frequências em MHz (suportado: 3.5 e 7.0)")
    parser.add_argument("--dr-db", type=float, default=45.0, help="Faixa dinâmica (dB)")
    parser.add_argument("--seed", type=int, default=42, help="Semente para reprodutibilidade")
    parser.add_argument("--run-id", type=str, default="manual", help="Identificador da execução para nomear arquivos")
    parser.add_argument("--stem-prefix", type=str, default="speckle_fieldii", help="Prefixo dos nomes de arquivo")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/matlab"), help="Diretório de saída")
    parser.add_argument("--show", action="store_true", help="Mostra comparação visual com matplotlib")
    return parser.parse_args()

# =====================================================================
# EXEMPLO DE EXECUÇÃO PRÁTICA COM O SEU GOLD STANDARD
# =====================================================================
if __name__ == "__main__":
    args = parse_args()
    outputs = generate_speckle_batch(
        gold_path=args.gold,
        output_dir=args.output_dir,
        frequencies_mhz=args.frequencies,
        dr_db=args.dr_db,
        seed=args.seed,
        run_id=args.run_id,
        stem_prefix=args.stem_prefix,
    )

    print("Arquivos gerados:")
    for output in outputs:
        print(f" - {output}")

    if args.show:
        import matplotlib.pyplot as plt

        img_gold = load_tiff(args.gold)
        cols = len(outputs) + 1
        plt.figure(figsize=(5 * cols, 5))

        plt.subplot(1, cols, 1)
        plt.imshow(img_gold, cmap="gray")
        plt.title("Gold Standard")
        plt.axis("off")

        for index, output in enumerate(outputs, start=2):
            plt.subplot(1, cols, index)
            plt.imshow(load_tiff(output), cmap="gray")
            plt.title(output.stem)
            plt.axis("off")

        plt.tight_layout()
        plt.show()