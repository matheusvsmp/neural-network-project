"""
05_resultados.py - Resultados e analise de erros (integrante 5)

Avalia os 6 modelos salvos NO MESMO CONJUNTO DE TESTE (uma unica vez, sem ajustes) e gera:
  - metricas de teste (acuracia, F1 macro) + custo de treino dos 6 experimentos
  - precisao, recall, F1 e suporte por classe
  - matrizes de confusao (contagem + % por linha), classes na ordem glass, paper, cardboard, plastic, metal, trash
  - curvas de aprendizagem (perda e acuracia, treino x validacao) dos 6 experimentos
  - efeito da resolucao na CNN sem pesos (desempenho e custo)
  - analise de erros do modelo com MAIOR F1 macro de VALIDACAO (ate 6 imagens)

Usa o common.py (mesmo split.json e mesmo pipeline). Rode na RAIZ do projeto, onde estao
modelos/, historicos/, split.json e dataset-resized/.

  python 05_resultados.py
Saidas: resultados/ (tabelas .csv/.md, previsoes, figuras)
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from common import (CLASSES, DIR_HISTORICOS, DIR_MODELOS, EXPERIMENTOS, RESOLUCOES, caminhos_e_rotulos,
                    carregar_historico, carregar_split, criar_dataset)

SAIDA = Path("resultados")
IDX = list(range(len(CLASSES)))


def desenhar_matriz(ax, cm, titulo):
    pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in IDX:
        for j in IDX:
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j] * 100:.0f}%", ha="center", va="center", fontsize=7,
                    color="white" if pct[i, j] > 0.5 else "black")
    ax.set_xticks(IDX); ax.set_xticklabels(CLASSES, rotation=45, fontsize=7)
    ax.set_yticks(IDX); ax.set_yticklabels(CLASSES, fontsize=7)
    ax.set_xlabel("Classe prevista"); ax.set_ylabel("Classe real"); ax.set_title(titulo, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="split.json")
    ap.add_argument("--data_dir", default="dataset-resized")
    ap.add_argument("--batch", type=int, default=16, help="so afeta a velocidade da predicao")
    a = ap.parse_args()

    import tensorflow as tf
    split = carregar_split(a.split)
    (SAIDA / "figuras").mkdir(parents=True, exist_ok=True)
    (SAIDA / "previsoes").mkdir(parents=True, exist_ok=True)
    caminhos_teste, y_teste = caminhos_e_rotulos(split, "test", a.data_dir)

    linhas, por_classe, cms, hist, previsoes = [], [], {}, {}, {}
    for nome, cfg in EXPERIMENTOS.items():
        arq = DIR_MODELOS / f"{nome}.keras"
        if not arq.exists() or not (DIR_HISTORICOS / f"{nome}.json").exists():
            print(f"AVISO: {nome} nao encontrado (modelo ou historico) - pulando"); continue
        print(f"Avaliando {nome} no teste ...")
        modelo = tf.keras.models.load_model(arq, compile=False)
        ds = criar_dataset(split, "test", cfg["res"], a.batch, a.data_dir, cache=False)
        prob = modelo.predict(ds, verbose=0)
        pred = prob.argmax(1)
        reg = carregar_historico(nome)
        meta, hist[nome] = reg["meta"], reg["historico"]
        altura, largura = RESOLUCOES[cfg["res"]]
        gpu = meta["hardware"].get("gpu")

        linhas.append({
            "experimento": nome, "modelo": cfg["modelo"].upper(), "resolucao": f"{largura}x{altura}",
            "pesos_de_classe": "Sim" if cfg["pesos"] else "Nao",
            "acuracia_teste": accuracy_score(y_teste, pred),
            "f1_macro_teste": f1_score(y_teste, pred, average="macro", labels=IDX, zero_division=0),
            "f1_macro_val": meta["val_f1_macro"], "acuracia_val": meta["val_accuracy"],
            "parametros": meta["parametros_totais"], "tamanho_modelo_MB": meta["tamanho_modelo_salvo_mb"],
            "epocas": meta["epocas_rodadas"], "epoca_modelo_salvo": meta["epoca_do_modelo_salvo"],
            "tempo_total_s": meta["tempo_total_treino_s"], "tempo_medio_epoca_s": meta["tempo_medio_por_epoca_s"],
            "batch": meta["batch_size"], "hardware": (", ".join(gpu) if gpu else "CPU") + f" ({meta['hardware'].get('cpu')})",
            "pico_ram_MB": meta.get("memoria_ram_pico_mb"), "pico_gpu_MB": meta.get("memoria_gpu_pico_mb"),
            "justificativa_lote": meta.get("justificativa_lote")})

        rel = classification_report(y_teste, pred, labels=IDX, target_names=CLASSES, zero_division=0, output_dict=True)
        for c in CLASSES:
            por_classe.append({"experimento": nome, "classe": c, "precisao": rel[c]["precision"],
                               "recall": rel[c]["recall"], "f1": rel[c]["f1-score"], "suporte": int(rel[c]["support"])})
        cms[nome] = confusion_matrix(y_teste, pred, labels=IDX)
        df = pd.DataFrame({"arquivo": caminhos_teste, "real": [CLASSES[i] for i in y_teste],
                           "previsto": [CLASSES[i] for i in pred], "prob_prevista": prob.max(1)})
        for i, c in enumerate(CLASSES):
            df[f"p_{c}"] = prob[:, i]
        df.to_csv(SAIDA / "previsoes" / f"{nome}_teste.csv", index=False)
        previsoes[nome] = (pred, prob)

    if not linhas:
        raise SystemExit("Nenhum modelo encontrado em modelos/ + historicos/.")
    tab = pd.DataFrame(linhas)
    tab.to_csv(SAIDA / "tabela_metricas_custo.csv", index=False)
    pc = pd.DataFrame(por_classe)
    pc.to_csv(SAIDA / "tabela_por_classe.csv", index=False)
    with open(SAIDA / "tabelas.md", "w", encoding="utf-8") as f:
        f.write("## Metricas e custo (teste)\n\n" + tab.round(4).to_markdown(index=False) + "\n\n")
        f.write("## Por classe (teste)\n\n" + pc.round(4).to_markdown(index=False) + "\n")
    print("\n", tab[["experimento", "acuracia_teste", "f1_macro_teste", "f1_macro_val", "tempo_total_s"]].round(4).to_string(index=False))

    # matrizes de confusao
    fig, axs = plt.subplots(2, 3, figsize=(14, 9)); axs = axs.ravel()
    for ax, (nome, cm) in zip(axs, cms.items()):
        desenhar_matriz(ax, cm, nome)
    for ax in axs[len(cms):]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(SAIDA / "figuras" / "matrizes_confusao.png", dpi=200); plt.close()
    for nome, cm in cms.items():
        f2, ax = plt.subplots(figsize=(5.5, 5)); desenhar_matriz(ax, cm, nome); plt.tight_layout()
        f2.savefig(SAIDA / "figuras" / f"matriz_{nome}.png", dpi=200); plt.close(f2)

    # curvas de aprendizagem
    fig, axs = plt.subplots(len(hist), 2, figsize=(10, 2.8 * len(hist)), squeeze=False)
    for r, (nome, h) in enumerate(hist.items()):
        ep = range(1, len(h["loss"]) + 1)
        for c, (k, rot) in enumerate([("loss", "perda"), ("accuracy", "acuracia")]):
            axs[r, c].plot(ep, h[k], label="treino"); axs[r, c].plot(ep, h[f"val_{k}"], label="validacao")
            axs[r, c].set_title(f"{nome} - {rot}", fontsize=9); axs[r, c].set_xlabel("epoca"); axs[r, c].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(SAIDA / "figuras" / "curvas_aprendizagem.png", dpi=200); plt.close()

    # efeito da resolucao (CNN sem pesos)
    res = tab[tab.experimento.isin(["cnn_128_sem", "cnn_256_sem", "cnn_512_sem"])]
    if len(res) > 1:
        x = np.arange(len(res)); rot = res.resolucao.tolist()
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        axs[0].bar(x - 0.2, res.acuracia_teste, 0.4, label="acuracia"); axs[0].bar(x + 0.2, res.f1_macro_teste, 0.4, label="F1 macro")
        axs[0].set_xticks(x); axs[0].set_xticklabels(rot); axs[0].set_title("Desempenho no teste por resolucao"); axs[0].legend()
        axs[1].bar(x - 0.2, res.tempo_medio_epoca_s, 0.4, label="tempo medio/epoca (s)")
        axs[1].bar(x + 0.2, res.tempo_total_s, 0.4, label="tempo total (s)")
        axs[1].set_xticks(x); axs[1].set_xticklabels(rot); axs[1].set_title("Custo de treinamento por resolucao"); axs[1].legend()
        plt.tight_layout(); plt.savefig(SAIDA / "figuras" / "efeito_resolucao.png", dpi=200); plt.close()

    # analise de erros do melhor modelo (F1 macro de VALIDACAO)
    melhor = tab.sort_values("f1_macro_val", ascending=False).iloc[0]["experimento"]
    pred, prob = previsoes[melhor]
    erros = np.where(pred != y_teste)[0]
    print(f"\nMelhor modelo (F1 macro na validacao): {melhor} | erros no teste: {len(erros)} de {len(y_teste)}")
    sel = erros[:6]
    if len(sel):
        fig, axs = plt.subplots(2, 3, figsize=(12, 7)); axs = axs.ravel()
        for ax, i in zip(axs, sel):
            ax.imshow(Image.open(caminhos_teste[i]).convert("RGB")); ax.axis("off")
            ax.set_title(f"real: {CLASSES[y_teste[i]]}\nprevisto: {CLASSES[pred[i]]} (p={prob[i, pred[i]]:.2f})", fontsize=9)
        for ax in axs[len(sel):]:
            ax.axis("off")
        plt.suptitle(f"Erros de {melhor} no teste"); plt.tight_layout()
        plt.savefig(SAIDA / "figuras" / "erros_melhor_modelo.png", dpi=200); plt.close()
    else:
        print("O melhor modelo nao errou nenhuma imagem de teste: registre esse resultado no artigo.")
    pd.DataFrame([{"arquivo": caminhos_teste[i], "real": CLASSES[y_teste[i]], "previsto": CLASSES[pred[i]],
                   "prob_prevista": float(prob[i, pred[i]])} for i in erros]).to_csv(SAIDA / "erros_melhor_modelo.csv", index=False)
    (SAIDA / "melhor_modelo.json").write_text(json.dumps(
        {"melhor_modelo_por_f1_macro_val": melhor, "n_erros_teste": int(len(erros)), "n_teste": int(len(y_teste))},
        indent=1, ensure_ascii=False), encoding="utf-8")
    print("OK! Tudo salvo em", SAIDA)


if __name__ == "__main__":
    main()
