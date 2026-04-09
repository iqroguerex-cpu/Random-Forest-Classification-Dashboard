import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap

# Page Configuration
st.set_page_config(page_title="Random Forest Classifier", layout="wide")

st.title("🌲 Random Forest Classification Dashboard")
st.markdown("""
This dashboard visualizes the **Random Forest** algorithm. By combining multiple decision trees, 
this ensemble method reduces overfitting and provides a more robust decision boundary.
""")

# --- Sidebar: Hyperparameters ---
st.sidebar.header("Ensemble Settings")
n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 1, 100, 10)
criterion = st.sidebar.selectbox("Criterion", ("entropy", "gini"))
max_depth = st.sidebar.slider("Max Depth of Trees", 1, 20, 10)
test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 25) / 100
random_state = st.sidebar.number_input("Random State", value=0)

# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        return pd.read_csv('Social_Network_Ads.csv')
    except:
        return None

dataset = load_data()

if dataset is not None:
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Feature Scaling
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    # Training
    classifier = RandomForestClassifier(
        n_estimators=n_estimators, 
        criterion=criterion, 
        max_depth=max_depth,
        random_state=random_state
    )
    classifier.fit(X_train_scaled, y_train)

    # --- Layout ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🔮 Predict New User")
        input_age = st.number_input("Age", 18, 60, 30)
        input_salary = st.number_input("Salary ($)", 15000, 150000, 87000)
        
        # Prediction logic
        new_pred = classifier.predict(sc.transform([[input_age, input_salary]]))
        label = "Will Buy" if new_pred[0] == 1 else "Won't Buy"
        color = "green" if new_pred[0] == 1 else "red"
        st.markdown(f"### Result: :{color}[{label}]")

        st.divider()
        
        st.subheader("📊 Performance Metrics")
        y_pred = classifier.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        st.metric("Model Accuracy", f"{acc:.2%}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm).plot(ax=ax_cm, cmap='Purples', colorbar=False)
        st.pyplot(fig_cm)

    with col2:
        st.subheader("🗺️ Forest Decision Boundary")
        set_choice = st.radio("Show results for:", ("Training Set", "Test Set"), horizontal=True)
        
        def plot_forest_boundary(X_data, y_data, title):
            X_set, y_set = sc.inverse_transform(X_data), y_data
            
            # Create meshgrid with slightly higher step for web performance
            x_min, x_max = X_set[:, 0].min() - 5, X_set[:, 0].max() + 5
            y_min, y_max = X_set[:, 1].min() - 500, X_set[:, 1].max() + 500
            
            X1, X2 = np.meshgrid(
                np.arange(x_min, x_max, 1),
                np.arange(y_min, y_max, 500)
            )
            
            # Grid Predictions
            Z = classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape)
            
            fig, ax = plt.subplots()
            ax.contourf(X1, X2, Z, alpha=0.7, cmap=ListedColormap(['#FA8072', '#1E90FF']))
            
            # Scatter points
            for i, j in enumerate(np.unique(y_set)):
                ax.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                           c=ListedColormap(['#FA8072', '#1E90FF'])(i), label=j, edgecolors='black')
            
            ax.set_title(title)
            ax.set_xlabel('Age')
            ax.set_ylabel('Estimated Salary')
            ax.legend()
            return fig

        if set_choice == "Training Set":
            st.pyplot(plot_forest_boundary(X_train_scaled, y_train, "Forest (Training Set)"))
        else:
            st.pyplot(plot_forest_boundary(X_test_scaled, y_test, "Forest (Test Set)"))

    # Data Source
    with st.expander("📂 View Dataset"):
        st.dataframe(dataset)

else:
    st.warning("Please ensure 'Social_Network_Ads.csv' is available in the app directory.")
