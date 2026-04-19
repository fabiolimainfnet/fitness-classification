import streamlit as st
import mlflow.sklearn
import pandas as pd

# 1. Configuração da Página
st.set_page_config(
    page_title="Sistema Fitness - Evolução", 
    page_icon="💪", 
    layout="centered"
)

st.title("💪 Diagnóstico de Condicionamento Físico")
st.markdown("---")

# 2. Função para Carregar o Modelo do MLflow
@st.cache_resource
def load_production_model():
    experiment = mlflow.get_experiment_by_name("Fitness_System_Evolution")
    
    # Busca a run específica do LDA (Exp_05)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'Exp_05_LDA_Perceptron'",
        order_by=["metrics.accuracy DESC"]
    )
    
    if runs.empty:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.accuracy DESC"]
        )
    
    model_uri = f"runs:/{runs.iloc[0].run_id}/model_lda"
    return mlflow.sklearn.load_model(model_uri)

try:
    model = load_production_model()
    st.sidebar.success("✅ Modelo Carregado com Sucesso")
except Exception as e:
    st.sidebar.error(f"❌ Erro de Conexão: {e}")

# 3. Formulário de Entrada (Interface em Português)
st.subheader("📋 Ficha de Avaliação do Aluno")

with st.form("fitness_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Idade", 18, 100, 36)
        gender_input = st.selectbox("Gênero", ["Masculino", "Feminino"])
        height_cm = st.number_input("Altura (cm)", 100, 250, 175)
        weight_kg = st.number_input("Peso (kg)", 30.0, 200.0, 80.0)
        heart_rate = st.number_input("Frequência Cardíaca (BPM)", 40, 220, 70)

    with col2:
        blood_pressure = st.number_input("Pressão Arterial (Sistólica)", 80, 200, 120)
        sleep_hours = st.number_input("Horas de Sono por Noite", 0.0, 24.0, 8.0)
        nutrition_quality = st.slider("Qualidade da Alimentação (1-10)", 1, 10, 5)
        activity_index = st.slider("Nível de Atividade Física (1-10)", 1.0, 10.0, 5.0)
        smokes_input = st.selectbox("Fumante?", ["Não", "Sim"])

    submit = st.form_submit_button("Gerar Diagnóstico")

# 4. Lógica de Inferência com Mapeamento de Idioma
if submit:
    # MAPEAMENTO: Traduzimos a interface em PT para o formato que o modelo conhece (M/F e yes/no)
    gender_map = "M" if gender_input == "Masculino" else "F"
    smokes_map = "yes" if smokes_input == "Sim" else "no"

    data = {
        'age': [age],
        'height_cm': [height_cm],
        'weight_kg': [weight_kg],
        'heart_rate': [heart_rate],
        'blood_pressure': [blood_pressure],
        'sleep_hours': [sleep_hours],
        'nutrition_quality': [nutrition_quality],
        'activity_index': [activity_index],
        'smokes': [smokes_map],
        'gender': [gender_map]
    }
    
    input_df = pd.DataFrame(data)
    
    try:
        prediction = model.predict(input_df)
        
        st.markdown("---")
        if prediction[0] == 1:
            st.balloons()
            st.success("### 🏆 Resultado: CONDICIONADO")
            st.write("O aluno apresenta excelentes indicadores de saúde e performance.")
        else:
            st.warning("### ⚠️ Resultado: NÃO CONDICIONADO")
            st.write("Recomenda-se ajuste na rotina de treinos e hábitos diários.")
            
    except Exception as e:
        st.error(f"Erro técnico na inferência: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Engenharia de Machine Learning**")
st.sidebar.write("Simulação de ambiente produtivo utilizando MLflow e Streamlit.")