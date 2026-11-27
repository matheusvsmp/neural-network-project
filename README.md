UNIVERSIDADE DE FORTALEZA (UNIFOR) — EDUCAÇÃO A DISTÂNCIA  
Disciplina: Redes Neurais Artificiais  
Atividade final — 2026

# Classificação de resíduos com TrashNet: MLP × CNN

Este repositório reúne o código, os registros de treinamento e os resultados da atividade final da disciplina de Redes Neurais Artificiais (EAD Unifor). O trabalho trata da classificação de imagens de resíduos em seis classes (`glass`, `paper`, `cardboard`, `plastic`, `metal`, `trash`), a partir do conjunto de dados TrashNet (THUNG, 2016; YANG; THUNG, 2016), na versão `dataset-resized.zip`, com 2.527 imagens de 512 × 384 pixels.

O estudo compara **perceptrons multicamadas (MLP) e redes neurais convolucionais (CNN)**, examina o efeito dos **pesos de classe** sobre a classe minoritária `trash` e verifica se o uso de **imagens de maior resolução** justifica o aumento do custo de treinamento. Inclui, ainda, o objetivo extra com YOLO (classificação por transferência de aprendizagem).

**Integrantes**:

- David Vidal Cavalcante — 2523587
- Laisa Costa Chagas — 2523558
- Lara de Oliveira Nascimento — 2523569
- Maria Mikaelle Andrade da Silva — 2523531
- Matheus Victor Soares Moura de Paula — 2523532
- Pedro Carvalho Lendengues Neto — 2527158


---

## 1. Síntese dos resultados

Todos os modelos utilizam a mesma divisão dos dados (`split.json`, proporção 70/15/15, semente 42) e foram avaliados **uma única vez** sobre o mesmo conjunto de teste (379 imagens). O modelo submetido à análise de erros foi escolhido pelo maior F1 macro obtido na **validação**.

| Experimento | Modelo | Entrada (largura × altura) | Pesos de classe | Acurácia (teste) | F1 macro (teste) | F1 macro (validação) | Tempo total (s) | Pico de GPU (MB) |
|---|---|---|---|---|---|---|---|---|
| `mlp_128_sem` | MLP | 128 × 96 | não | 0,6016 | 0,5644 | 0,5050 | 118,4 | 309,4 |
| `mlp_128_com` | MLP | 128 × 96 | sim | 0,5884 | 0,5610 | 0,5024 | 62,4 | 437,1 |
| `cnn_128_sem` | CNN | 128 × 96 | não | 0,7335 | 0,6914 | 0,6461 | 114,5 | 717,5 |
| `cnn_128_com` | CNN | 128 × 96 | sim | 0,7124 | 0,6892 | 0,6676 | 86,8 | 717,5 |
| `cnn_256_sem` | CNN | 256 × 192 | não | 0,7309 | 0,6725 | 0,6505 | 278,2 | 2.337,2 |
| `cnn_512_sem` | CNN | 512 × 384 | não | 0,7203 | 0,6533 | 0,6484 | 1.140,2 | 6.833,4 |

- **Modelo selecionado pela validação:** `cnn_128_com` (109 erros em 379 imagens de teste).
- As três CNNs sem pesos de classe possuem **a mesma arquitetura** (111.430 parâmetros) e foram treinadas na mesma GPU (Tesla T4), com tamanho de lote 32.
- As tabelas completas (custo, hardware e precisão, revocação e F1 por classe) encontram-se em `resultados/tabelas.md`.
- **Extra YOLO** (`yolo11n-cls`, pesos pré-treinados, entrada de 224 × 224): acurácia de 0,9156 e F1 macro de 0,9072 no teste (`resultados/yolo_cls_teste.json`).

Diferenças entre modelos de uma a poucas imagens, em um total de 379, não são conclusivas: o treinamento possui componente aleatória e cada configuração foi treinada uma única vez (semente 42).

---

## 2. Estrutura do repositório

```
.
├── treino_eda_dados_colab.ipynb  notebook da análise exploratória dos dados (EDA)
├── 01_eda_split.py           inspeção do dataset, split 70/15/15 (semente 42), pesos de classe, figuras
├── 02_mlp.py                 treino dos experimentos mlp_128_sem e mlp_128_com
├── 03_cnn.py                 treino dos experimentos cnn_128_sem, cnn_128_com, cnn_256_sem, cnn_512_sem
├── 04_yolo.py                extra: YOLO classify
├── 05_resultados.py          avalia os modelos salvos no teste; gera tabelas, previsões e figuras
├── 05_resultados.ipynb       notebook de resultados (métricas, figuras e análise)
├── common.py                 código compartilhado: semente, classes, pipeline de dados, callbacks, custo
├── split.json                divisão oficial (treino/validação/teste) e pesos de classe
├── saida_eda.txt             saída da análise exploratória (contagens, duplicatas, pesos)
├── requirements.txt          dependências
├── treino_mlp_colab.ipynb    execução do 02_mlp.py no Colab
├── treino_cnn_colab.ipynb    execução do 03_cnn.py no Colab
├── treino_yolo_colab.ipynb   execução do 04_yolo.py no Colab
├── historicos/               histórico por época e metadados de cada experimento (.json e .csv)
├── figuras/
│   ├── eda/                  distribuição, exemplos, mesma imagem nas 3 resoluções, duplicatas
│   └── curvas/               curvas de perda e acurácia de cada experimento
├── resultados/
│   ├── tabelas.md, tabela_metricas_custo.csv, tabela_por_classe.csv
│   ├── melhor_modelo.json, erros_melhor_modelo.csv
│   ├── previsoes/            previsão por imagem de teste de cada experimento
│   ├── figuras/              matrizes de confusão, curvas, efeito da resolução, erros
│   └── yolo_cls_teste.json   resultado do extra YOLO no teste
└── modelos/                  (não versionada) modelos .keras, obtidos no Drive (seção 4)
```

---

## 3. Instalação e dependências

As versões registradas nos históricos de treinamento são **Python 3.13.16** e **TensorFlow 2.21.0** (treinamento na GPU Tesla T4 do Google Colab); o extra YOLO utilizou **ultralytics 8.4.218** e **PyTorch 2.14.1**. A avaliação dos modelos foi verificada também com TensorFlow 2.21.0 em CPU.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

Recomenda-se não criar o ambiente virtual em pastas sincronizadas (por exemplo, OneDrive) e utilizar uma versão do Python compatível com o TensorFlow instalado. Para o extra YOLO, instalar também: `pip install ultralytics`.

### Dataset
As imagens **não** estão incluídas no repositório. É necessário obter o arquivo `dataset-resized.zip` do TrashNet (`https://github.com/garythung/trashnet`, pasta `data/`) e extraí-lo na raiz do projeto, de modo que existam as pastas `dataset-resized/cardboard`, `glass`, `metal`, `paper`, `plastic` e `trash`:

```bash
git clone https://github.com/garythung/trashnet.git
unzip -q trashnet/data/dataset-resized.zip -d .
```

---

## 4. Modelos treinados (Google Drive)

Os arquivos de modelo são volumosos e, por isso, não são versionados no GitHub. Devem ser obtidos no Google Drive e copiados para a pasta `modelos/`, na raiz do projeto, preservando-se os nomes dos arquivos.

| Arquivo | Experimento | Link |
|---|---|---|
| `mlp_128_sem.keras` | MLP, 128 × 96, sem pesos | [pasta no Drive](https://drive.google.com/drive/folders/1Ixg4ksha1TBMhNSVH9EqWhkXxVfJIEuR?usp=drive_link) |
| `mlp_128_com.keras` | MLP, 128 × 96, com pesos | [pasta no Drive](https://drive.google.com/drive/folders/1Ixg4ksha1TBMhNSVH9EqWhkXxVfJIEuR?usp=drive_link) |
| `cnn_128_sem.keras` | CNN, 128 × 96, sem pesos | [pasta no Drive](https://drive.google.com/drive/folders/1Ixg4ksha1TBMhNSVH9EqWhkXxVfJIEuR?usp=drive_link) |
| `cnn_128_com.keras` | CNN, 128 × 96, com pesos | [pasta no Drive](https://drive.google.com/drive/folders/1Ixg4ksha1TBMhNSVH9EqWhkXxVfJIEuR?usp=drive_link) |
| `cnn_256_sem.keras` | CNN, 256 × 192, sem pesos | [pasta no Drive](https://drive.google.com/drive/folders/1Ixg4ksha1TBMhNSVH9EqWhkXxVfJIEuR?usp=drive_link) |
| `cnn_512_sem.keras` | CNN, 512 × 384, sem pesos | [pasta no Drive](https://drive.google.com/drive/folders/1Ixg4ksha1TBMhNSVH9EqWhkXxVfJIEuR?usp=drive_link) |
| `yolo_cls.pt` | Extra YOLO (`yolo11n-cls`) | [pasta no Drive](https://drive.google.com/drive/folders/1Ixg4ksha1TBMhNSVH9EqWhkXxVfJIEuR?usp=drive_link) |

Todos os arquivos estão reunidos em uma única pasta, de acesso público para leitura: https://drive.google.com/drive/folders/1Ixg4ksha1TBMhNSVH9EqWhkXxVfJIEuR?usp=drive_link

---

## 5. Instruções de execução

### 5.1 Avaliação sem novo treinamento (reprodução dos resultados)
Pré-requisitos: dependências instaladas, pasta `dataset-resized/` na raiz e modelos `.keras` em `modelos/` (seção 4).

```bash
python 05_resultados.py
```

O script avalia os seis modelos no conjunto de teste e grava todas as saídas em `resultados/`. Em seguida, o notebook `05_resultados.ipynb` pode ser executado célula a célula (a primeira célula executa o mesmo script). Os arquivos de `historicos/` já se encontram no repositório; assim, as curvas de aprendizagem e o custo de treinamento não dependem de novo treino.

### 5.2 Treinar os experimentos
Os comandos devem ser executados na raiz do projeto. Cada um grava o modelo em `modelos/`, o histórico em `historicos/` e as curvas em `figuras/curvas/`. Experimentos já existentes são ignorados; a opção `--forcar` obriga um novo treinamento.

```bash
python 02_mlp.py                                   # experimentos mlp_128_sem e mlp_128_com
python 03_cnn.py                                   # experimentos cnn_128_sem e cnn_128_com
python 03_cnn.py --exp cnn_256_sem cnn_512_sem     # experimentos da CNN em 256 × 192 e 512 × 384
python 03_cnn.py --exp cnn_512_sem --batch_size 16 # apenas em caso de falta de memória (registrar e justificar)
```

Para que a comparação de custo seja válida, todos os experimentos devem ser treinados na **mesma GPU**. O hardware, a versão do TensorFlow, o tamanho de lote e os picos de uso de RAM e de GPU são registrados no campo `meta` de cada arquivo `historicos/<experimento>.json`.

### 5.3 Análise exploratória e divisão dos dados
```bash
python 01_eda_split.py --data_dir dataset-resized
```
O script verifica a integridade das imagens, os tamanhos, os canais e as duplicatas, e gera o arquivo `split.json` e as figuras de `figuras/eda/`. O resultado é determinístico (semente 42), não sendo necessário repetir a execução. A análise também está documentada, com saídas e figuras, no notebook `treino_eda_dados_colab.ipynb`.

### 5.4 Extra YOLO
```bash
python 04_yolo.py            # treina o yolo11n-cls com pesos pré-treinados
python 04_yolo.py --teste    # avalia o modelo salvo no teste (somente ao final)
```

---

## 6. Protocolo experimental

- **Divisão:** estratificada 70/15/15, semente 42, em `split.json` (1.769 treino, 379 validação, 379 teste). Todos os experimentos utilizam a mesma divisão. Os três pares de imagens duplicadas com rótulos conflitantes foram mantidos integralmente no treino, a fim de evitar vazamento de dados (`figuras/eda/duplicatas.png`).
- **Pré-processamento:** conversão para RGB, redimensionamento a partir de 512 × 384 e normalização por divisão dos pixels por 255. No Keras, a entrada é altura × largura × canais: (96, 128, 3), (192, 256, 3) e (384, 512, 3).
- **Aumento de dados:** aplicado apenas ao treino (flip horizontal e vertical, brilho e contraste).
- **Pesos de classe:** `balanced`, calculados somente com o treino (`glass` 0,84; `paper` 0,7104; `cardboard` 1,0455; `plastic` 0,8723; `metal` 1,0273; `trash` 3,0712). Os experimentos "com pesos" repetem a configuração e a semente dos experimentos "sem pesos".
- **Treinamento:** otimizador Adam (KINGMA; BA, 2015) com taxa inicial de 3 × 10⁻⁴ no MLP e 10⁻³ na CNN, entropia cruzada categórica esparsa, lote 32, até 100 épocas, `EarlyStopping` em `val_loss` com paciência 10 restaurando os melhores pesos, `ReduceLROnPlateau` (fator 0,5, paciência 4). A semente e a época do modelo salvo estão em cada histórico.
- **MLP:** pixels achatados em vetor de 36.864 valores; duas camadas ocultas (256 e 128 neurônios), cada uma com normalização em lote, ReLU e *dropout* (0,5 e 0,3), regularização L2 de 10⁻⁴ e saída *softmax* com seis neurônios; 9.472.646 parâmetros.
- **CNN:** 3 blocos Conv 3 × 3 (32, 64, 128 filtros) + BatchNormalization + ReLU + MaxPooling, *global average pooling*, Dropout 0,3, Dense 128, Dropout 0,3, saída com 6 neurônios; 111.430 parâmetros nas três resoluções.
- **Seleção e teste:** a validação orienta a escolha de configurações e o critério de parada; o teste é utilizado somente ao final. Nas matrizes de confusão, as classes seguem a ordem `glass, paper, cardboard, plastic, metal, trash`. As métricas usam `zero_division=0`.
- **Custo:** tempo medido com `time.time()` em cada época de `model.fit` (treino + validação), excluindo download e preparação prévia; pico de RAM e de GPU registrados no histórico.

---

## 7. Limitações do estudo

- Cada configuração foi treinada uma única vez; os resultados podem variar entre execuções.
- A classe `trash` possui apenas 21 imagens no teste; cada erro altera sua revocação (recall) em 4,8 pontos percentuais.
- O conjunto de dados apresenta fundo uniforme e objetos isolados, o que limita a generalização para fotografias de situações reais.
- Há três pares de imagens idênticas com rótulos distintos (`glass115 = metal91`, `glass176 = plastic152`, `glass389 = plastic332`).
- O YOLO utiliza entrada de 224 × 224 e pré-treinamento no ImageNet; por isso, a comparação com a CNN não isola uma única variável.

---

## 8. Uso de inteligência artificial

Ferramentas de inteligência artificial foram utilizadas como apoio à geração e à revisão de código e à organização do material. A equipe revisou o código, executou os treinamentos e conduziu as decisões metodológicas, a análise e a interpretação dos resultados.

## 9. Referências

ABADI, Martín *et al*. **TensorFlow**: large-scale machine learning on heterogeneous systems. [*S. l.*]: TensorFlow, 2015. Software disponível em: https://www.tensorflow.org. Acesso em: 26 nov. 2026.

DENG, Jia *et al*. ImageNet: a large-scale hierarchical image database. *In*: IEEE CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION, 2009, Miami. **Proceedings** [...]. Piscataway: IEEE, 2009. p. 248-255.

IOFFE, Sergey; SZEGEDY, Christian. Batch normalization: accelerating deep network training by reducing internal covariate shift. *In*: INTERNATIONAL CONFERENCE ON MACHINE LEARNING, 32., 2015, Lille. **Proceedings** [...]. [*S. l.*]: PMLR, 2015. p. 448-456.

JOCHER, Glenn; QIU, Jing; CHAURASIA, Ayush. **Ultralytics YOLO**. [*S. l.*]: Ultralytics, 2023. Software disponível em: https://github.com/ultralytics/ultralytics. Acesso em: 26 nov. 2026.

KINGMA, Diederik P.; BA, Jimmy. Adam: a method for stochastic optimization. *In*: INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS, 3., 2015, San Diego. **Anais** [...]. San Diego: ICLR, 2015. Disponível em: https://arxiv.org/abs/1412.6980. Acesso em: 26 nov. 2026.

PEDREGOSA, Fabian *et al*. Scikit-learn: machine learning in Python. **Journal of Machine Learning Research**, [*s. l.*], v. 12, p. 2825-2830, 2011.

SRIVASTAVA, Nitish *et al*. Dropout: a simple way to prevent neural networks from overfitting. **Journal of Machine Learning Research**, [*s. l.*], v. 15, n. 1, p. 1929-1958, 2014.

THUNG, Gary. **TrashNet**: dataset of images of trash. [*S. l.*]: GitHub, 2016. Disponível em: https://github.com/garythung/trashnet. Acesso em: 26 nov. 2026.

YANG, Mindy; THUNG, Gary. **Classification of trash for recyclability status**. Stanford: Stanford University, 2016. Relatório de projeto da disciplina CS229.
