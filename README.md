# F1 Neural Predictor

[![CI](https://github.com/evfernandes-dev/f1-neural-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/evfernandes-dev/f1-neural-predictor/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![Licença](https://img.shields.io/badge/licen%C3%A7a-MIT-green)](LICENSE)

Painel em [Streamlit](https://streamlit.io/) que estima a probabilidade de cada
piloto vencer o próximo Grande Prêmio de Fórmula 1. Os dados vêm da
[OpenF1](https://openf1.org/) e, se ela falhar, da [Jolpica](https://jolpi.ca/).
A previsão usa uma rede neural simples feita com scikit-learn.

> Projeto educacional. As probabilidades não são recomendações de apostas.

![Painel do F1 Neural Predictor](docs/screenshot.png)

## O que o painel mostra

- Probabilidade de vitória dos pilotos ativos.
- Métrica principal: acerto do vencedor por corrida (top-1).
- Comparação com um baseline que escolhe o piloto com melhor média de pontos.
- Forma recente de cada piloto e download da tabela em CSV.

## Como rodar

Pré-requisitos: Python 3.12 ou mais recente.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

No Linux ou macOS, ative o ambiente com `source .venv/bin/activate`.

Abra `http://localhost:8501`, escolha o período na barra lateral e clique em
**Treinar e prever**. A primeira execução baixa os resultados e pode demorar
alguns minutos.

## Testes

```powershell
pip install -r requirements-dev.txt
pytest
```

O GitHub Actions roda o lint (ruff) e os testes a cada push.

## Como funciona

1. **Coleta:** resultados das corridas concluídas desde 2023.
2. **Forma recente:** média das últimas corridas de cada piloto (posição,
   vitórias, pódios, top 5, abandonos e pontos).
3. **Treino:** uma rede neural pequena (uma camada oculta com 8 neurônios)
   aprende quais pilotos venceram. As variáveis usam apenas corridas anteriores
   para não vazar o resultado.
4. **Previsão:** o modelo gera uma probabilidade por piloto e normaliza para
   que a soma seja 100%.

O treino e o teste são separados por corrida: as últimas 20% das provas ficam
para teste, sem misturar pilotos da mesma prova.

## Estrutura

```text
f1-neural-predictor/
├── app.py                    # painel Streamlit
├── predictor.py              # coleta, features, treino e previsão
├── tests/                    # testes automatizados
├── docs/screenshot.png       # captura do painel
├── .streamlit/config.toml    # tema do painel
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml            # configuração de pytest e ruff
└── .github/workflows/ci.yml  # lint + testes
```

## Limitações

- A OpenF1 disponibiliza dados gratuitos principalmente a partir de 2023.
- O modelo não usa classificação, circuito, clima nem atualizações dos carros.
- As probabilidades são uma estimativa educacional, não uma previsão precisa.

## Próximos passos

- Incluir resultado da classificação e características do circuito.
- Salvar o modelo treinado para carregar mais rápido.
- Publicar uma demonstração online no Streamlit Community Cloud.

## Tecnologias

Python · pandas · NumPy · scikit-learn · Streamlit · Plotly · OpenF1
