import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# --- Load dataset and model ---
@st.cache_data
def load_data():
    return pd.read_csv("data/titanic.csv")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return model, feature_names

df = load_data()
model, feature_names = load_model()

# --- Sidebar navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Visualisations", "Model Prediction", "Model Performance"])

# --- Home page ---
if page == "Home":
    st.title("Titanic Survival Prediction App")
    st.write("""
    This app allows you to explore the Titanic dataset, visualise trends, 
    and predict passenger survival using a trained machine learning model.
    """)

    st.subheader("Dataset Quick Info")
    st.write(f"**Rows:** {df.shape[0]}")
    st.write(f"**Columns:** {df.shape[1]}")
    st.write(f"**Survival Rate:** {df['Survived'].mean():.2%}")

# --- Data Exploration page ---
elif page == "Data Exploration":
    st.header("Data Overview")
    st.write("Shape:", df.shape)
    st.write("Columns and Data Types:")
    st.write(df.dtypes)
    
    if st.checkbox("Show sample data"):
        st.write(df.sample(10))

    st.subheader("Interactive Data Filter")
    col_filter = st.multiselect("Select columns to view", df.columns.tolist(), default=df.columns.tolist())
    sex_filter = st.selectbox("Filter by Sex", options=["All"] + list(df["Sex"].unique()))
    pclass_filter = st.selectbox("Filter by Pclass", options=["All"] + sorted(df["Pclass"].unique()))

    filtered_df = df.copy()
    if sex_filter != "All":
        filtered_df = filtered_df[filtered_df["Sex"] == sex_filter]
    if pclass_filter != "All":
        filtered_df = filtered_df[filtered_df["Pclass"] == pclass_filter]

    st.write(filtered_df[col_filter].head(20))

# --- Visualisations page ---
elif page == "Visualisations":
    st.header("Visualisations")

    st.subheader("Survival Count")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="Survived", palette="pastel", ax=ax1)
    st.pyplot(fig1)

    st.subheader("Survival by Pclass")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x="Pclass", hue="Survived", palette="Set2", ax=ax2)
    st.pyplot(fig2)

    st.subheader("Age Distribution by Survival")
    fig3 = sns.histplot(df, x="Age", hue="Survived", bins=30, kde=True)
    st.pyplot(fig3.figure)

# --- Model Prediction page ---
elif page == "Model Prediction":
    st.header("Predict Survival")

    with st.form("prediction_form"):
        Pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
        Sex = st.selectbox("Sex", ["male", "female"])
        Age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
        SibSp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
        Parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
        Fare = st.number_input("Passenger Fare", min_value=0.0, max_value=600.0, value=32.0)
        Embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Create the base dictionary with all features except Embarked encoding
        input_dict = {
            "Pclass": Pclass,
            "Sex": 0 if Sex == "male" else 1,
            "Age": Age,
            "SibSp": SibSp,
            "Parch": Parch,
            "Fare": Fare,
            "FamilySize": SibSp + Parch + 1,
            # Initialize all Embarked one-hot columns as 0 first
            "Embarked_C": 0,
            "Embarked_Q": 0,
            "Embarked_S": 0,
        }

        # Set the correct Embarked dummy column to 1
        if Embarked == "C":
            input_dict["Embarked_C"] = 1
        elif Embarked == "Q":
            input_dict["Embarked_Q"] = 1
        else:  # Embarked == "S"
            input_dict["Embarked_S"] = 1

        # Create DataFrame
        input_df = pd.DataFrame([input_dict])

        # IMPORTANT: Reorder columns exactly as model expects
        model_features = model.feature_names_in_
        input_df = input_df[model_features]

        prediction = model.predict(input_df)[0]
        try:
            probability = model.predict_proba(input_df)[0][1]
        except:
            probability = None

        if prediction == 1:
            st.success(f"Survived ✅ (Confidence: {probability:.2%})" if probability is not None else "Survived ✅")
        else:
            st.error(f"Did not survive ❌ (Confidence: {1-probability:.2%})" if probability is not None else "Did not survive ❌")

# --- Model Performance page ---
elif page == "Model Performance":
    st.header("Model Evaluation")

    # --- Recreate preprocessing so columns match training ---
    df_eval = df.copy()

    # Feature engineering: FamilySize
    df_eval["FamilySize"] = df_eval["SibSp"] + df_eval["Parch"] + 1

    # Encode Sex
    df_eval["Sex"] = df_eval["Sex"].map({"male": 0, "female": 1})

    # One-hot encode Embarked
    df_eval = pd.get_dummies(df_eval, columns=["Embarked"], drop_first=True)

    # Ensure all expected columns exist
    for col in feature_names:
        if col not in df_eval.columns:
            df_eval[col] = 0

    # Reorder columns to match training
    df_eval = df_eval[feature_names]

    # Target
    y = df["Survived"]

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_eval, y, test_size=0.2, stratify=y, random_state=42)

    # Predictions
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except:
        y_prob = None

    # Metrics
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig4, ax4 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax4)
    ax4.set_xlabel("Predicted")
    ax4.set_ylabel("Actual")
    st.pyplot(fig4)

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        st.subheader(f"ROC Curve (AUC = {roc_auc:.3f})")
        fig5, ax5 = plt.subplots()
        ax5.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax5.plot([0, 1], [0, 1], 'k--')
        ax5.set_xlabel("False Positive Rate")
        ax5.set_ylabel("True Positive Rate")
        ax5.legend()
        st.pyplot(fig5)

        
        
        