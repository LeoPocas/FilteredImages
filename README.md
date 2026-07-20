# Filters

Pipeline em Python para gerar speckle, aplicar filtros locais e avaliar os resultados contra um gold standard em imagens TIFF.

O projeto combina três etapas principais:

1. Geração de imagens com ruído speckle no estilo Field II.
2. Filtragem com Kuan, Lee, média ou mediana.
3. Avaliação quantitativa com métricas de fidelidade e segmentação.

## Scripts principais

### `octave.py`
Gera imagens speckle a partir de um gold standard em TIFF.

- Simula o speckle físico com base em scatterers, PSF, envelope e compressão logarítmica.
- Expõe `generate_speckle_batch(...)` para uso programático pelo pipeline.
- Gera arquivos com nomes únicos usando `run_id`, frequência e índice.

Exemplo:
```bash
python octave.py input/cistosGoldStd.tif --frequencies 3.5 7.0 --output-dir outputs/matlab --run-id teste01
```

### `kuan.py`
Aplica filtros locais a imagens TIFF.

Filtros disponíveis:
- `kuan`
- `lee`
- `mean`
- `median`

O script aceita uma imagem única ou uma pasta com TIFFs e organiza as saídas em subpastas por filtro.

Exemplo:
```bash
python kuan.py input/cistosGoldStd.tif --filter kuan --output-dir outputs/filters
```

### `compare.py`
Compara um gold standard com imagens filtradas.

Métricas calculadas:
- média na região homogênea
- desvio padrão na região homogênea
- RMSE absoluto
- RMSE relativo
- SSIM
- mean shift
- USDSAI

Também pode exportar CSV com `--output-csv`.

Exemplo:
```bash
python compare.py input/cistosGoldStd.tif outputs/filters/kuan --output-csv outputs/pipeline/compare/kuan_metrics.csv
```

### `pipeline.py`
Executa o fluxo completo.

Etapas:
1. Resolve o gold standard.
2. Opcionalmente gera speckle com `octave.py`.
3. Aplica todos os filtros disponíveis em `kuan.py`.
4. Executa comparação com `compare.py`.
5. Executa segmentação com `segmentation.py`.
6. Gera `summary.csv` consolidado.

Por padrão, o pipeline cria uma execução isolada em `outputs/pipeline/<run_id>/` para evitar sobrescrita.

Exemplo:
```bash
python pipeline.py input/cistosGoldStd.tif outputs/matlab --with-octave 
```

## Estrutura do projeto

```text
/home/wsl/Filters
├── compare.py
├── kuan.py
├── octave.py
├── pipeline.py
├── segmentation.py
├── input/
├── outputs/
├── Field_II_ver_3_30_linux/
└── README.md
```

### Diretórios importantes

- `input/`: entrada do projeto, incluindo gold standards e imagens base.
- `outputs/`: saídas geradas pelo pipeline e pelos scripts auxiliares.
- `Field_II_ver_3_30_linux/`: distribuição do Field II original e arquivos auxiliares relacionados.

## Saídas geradas

O pipeline organiza os resultados por execução:

```text
outputs/pipeline/<run_id>/
├── speckle/
├── filters/
│   ├── kuan/
│   ├── lee/
│   ├── mean/
│   └── median/
├── compare/
├── segmentation/
└── summary.csv
```

Em geral:

- `speckle/`: TIFFs sintéticos gerados a partir do gold standard.
- `filters/<filtro>/`: imagens filtradas.
- `compare/`: CSVs com métricas de comparação por filtro.
- `segmentation/<filtro>/`: máscaras e overlays da segmentação.
- `summary.csv`: resumo consolidado da execução.

## Dependências

Pacotes usados com frequência:

- `numpy`
- `opencv-python`
- `pillow`
- `scipy`
- `scikit-image`
- `tifffile`
- `matplotlib` apenas para visualização opcional em `octave.py`

## Observações

- O pipeline usa nomes únicos por execução para evitar sobrescrita entre rodadas.
- O parâmetro de ruído `--noise-cv2` pode ser estimado automaticamente pelos filtros Kuan/Lee quando omitido.
- O fluxo com `--with-octave` gera imagens sintéticas antes da filtragem.

## Exemplo de fluxo completo

```bash
python pipeline.py input/cistosGoldStd.tif outputs/matlab --with-octave --noise-cv2 0.9 --run-id experimento_01
```

Esse comando:

- gera speckle sintético no estilo Field II;
- filtra as imagens com todos os filtros configurados;
- compara os resultados com o gold standard;
- executa segmentação;
- salva um resumo consolidado em `outputs/pipeline/experimento_01/summary.csv`.
