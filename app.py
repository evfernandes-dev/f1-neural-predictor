from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from predictor import OpenF1Error, run_analysis

st.set_page_config(page_title="F1 Neural Predictor", page_icon="🏎️", layout="wide")
st.html("""
<style>
.block-container{max-width:1180px;padding-top:2.5rem}
.hero{padding:32px;border-radius:22px;
background:linear-gradient(125deg,rgba(30,34,44,.96),rgba(42,7,12,.96));
border:1px solid #3a3f4a;box-shadow:0 18px 55px rgba(0,0,0,.30);margin-bottom:18px}
.eyebrow{font-size:.78rem;letter-spacing:.16em;text-transform:uppercase;color:#ff5266;font-weight:700}
.hero h1{font-size:2.65rem;margin:.35rem 0}.hero p{color:#b8bdc8;margin:0;max-width:680px}
.red{color:#ff3048}
</style>
<div class="hero"><div class="eyebrow">OpenF1 • Machine Learning</div>
<h1>F1 <span class="red">Neural Predictor</span></h1>
<p>Transforme resultados históricos em uma estimativa clara de vitória para a próxima corrida.</p></div>
""")


@st.cache_data(show_spinner=False, ttl=3600)
def cached_analysis(start_year: int, end_year: int, recent_races: int):
    return run_analysis(start_year, end_year, recent_races)

with st.sidebar:
    st.header("Configuração")
    start = st.number_input("Treinar desde", 2023, 2035, 2023)
    end = st.number_input("Treinar até", 2023, 2035, min(datetime.now().year, 2035))
    recent = st.slider(
        "Corridas consideradas na forma recente",
        3, 12, 6,
        help="Altera a janela estatística e treina novamente o modelo.",
    )
    run = st.button("Treinar e prever", type="primary", width="stretch")

st.caption("O primeiro processamento pode demorar porque os resultados são baixados da OpenF1.")

if run:
    if start > end:
        st.error("O ano inicial precisa ser menor ou igual ao ano final.")
        st.stop()
    with st.spinner("Limpando dados, treinando a rede neural e calculando probabilidades..."):
        try:
            result, metrics, dataset = cached_analysis(int(start), int(end), recent)
        except OpenF1Error as exc:
            st.error(str(exc))
            st.stop()

    leader = result.iloc[0]
    a, b, c, d = st.columns([1.9, 0.9, 1, 0.9])
    a.metric("Favorito", leader["piloto"])
    b.metric("Probabilidade", f'{leader["probabilidade"]:.1f}%')
    c.metric("Equipe", leader["equipe"])
    d.metric("Registros limpos", len(dataset))

    st.subheader("Probabilidades de vitória")
    chart = px.bar(result.head(10), x="probabilidade", y="piloto", orientation="h",
                   color="probabilidade", color_continuous_scale=["#5b111a", "#ff203b"],
                   labels={"probabilidade":"Probabilidade (%)", "piloto":"Piloto"})
    chart.update_layout(template="plotly_dark", paper_bgcolor="#090b10",
                        plot_bgcolor="#090b10", coloraxis_showscale=False,
                        yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(chart)

    with st.expander("Avaliação do modelo", expanded=False):
        cols = st.columns(4)
        cols[0].metric("Vencedor por corrida", f'{metrics["top1"]:.0%}')
        cols[1].metric("Baseline", f'{metrics["baseline"]:.0%}')
        cols[2].metric("Acurácia (linhas)", f'{metrics["acuracia"]:.1%}')
        cols[3].metric("AUC", "N/D" if metrics["auc"] is None else f'{metrics["auc"]:.2f}')
        st.caption(
            f'Precisão {metrics["precisao"]:.1%} · Recall {metrics["recall"]:.1%} · '
            f'F1 {metrics["f1"]:.1%} · Treino: {metrics["treino"]} linhas · '
            f'Teste: {metrics["teste"]} linhas em {metrics["corridas_teste"]} corridas.'
        )
        matrix = pd.DataFrame(metrics["matriz"], index=["Real: não", "Real: vitória"],
                              columns=["Previsto: não", "Previsto: vitória"])
        st.markdown("**Matriz de confusão**")
        st.dataframe(matrix, hide_index=False)
        st.caption(
            "O top-1 por corrida é a métrica principal: acertar o vencedor é o que "
            "importa. A acurácia por linha pode enganar, pois quase todos os pilotos "
            "não vencem uma determinada corrida."
        )

    st.dataframe(result.rename(columns={"piloto":"Piloto", "equipe":"Equipe",
        "media_posicao_5":"Posição média", "vitorias_5":"Taxa de vitórias",
        "podios_5":"Taxa de pódios", "top5_5":"Taxa de top 5",
        "abandono_5":"Taxa de abandono", "media_pontos_5":"Média de pontos",
        "probabilidade":"Probabilidade (%)"}), hide_index=True)
    st.download_button("Baixar CSV", result.to_csv(index=False).encode("utf-8"),
                       "probabilidades_f1.csv", "text/csv")
    st.info(
        f"A soma das probabilidades é 100%. A rede usou uma janela de "
        f"{recent} corridas para comparar a forma recente dos pilotos ativos."
    )
    st.warning("Modelo educacional: circuito, clima, atualizações e acidentes não estão incluídos.")
else:
    st.write("Escolha o período na lateral e clique em **Treinar e prever**.")
