import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Konfigurasi halaman
st.set_page_config(
    page_title="Decision Tree Classifier",
    page_icon="🌳",
    layout="wide"
)

class DecisionTreeApp:
    def __init__(self):
        self.model = None
        self.le_dict = {}
        self.feature_names = []
        self.target_name = ""
        
    def load_sample_data(self):
        """Data sample pada Pekerjaan (contoh)"""
        data = {
            'Umur': [25, 35, 45, 20, 35, 52, 23, 40, 60, 28],
            'Pendapatan': [50000, 75000, 100000, 25000, 80000, 120000, 30000, 90000, 150000, 55000],
            'Pekerjaan': ['Police', 'Doctor', 'Soldier', 'Teacher', 'Engineer', 
                         'Programmer', 'Fisherman', 'Farmer', 'Nurse', 'Lawyers'],
            'Menikah': ['tidak', 'Ya', 'Ya', 'tidak', 'Ya', 'Ya', 'tidak', 'Ya', 'Ya', 'tidak'],
            # 'Buy_C': ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No']
        }
        return pd.DataFrame(data)
    
    def preprocess_data(self, df, target_column):
        """Preprocessing data untuk kategorikal"""
        df_processed = df.copy()
        self.target_name = target_column
        
        # Encode target variable
        le_target = LabelEncoder()
        df_processed[target_column] = le_target.fit_transform(df_processed[target_column])
        self.le_dict[target_column] = le_target
        
        # Encode categorical features
        for column in df_processed.columns:
            if df_processed[column].dtype == 'object' and column != target_column:
                le = LabelEncoder()
                df_processed[column] = le.fit_transform(df_processed[column])
                self.le_dict[column] = le
                
        return df_processed
    
    # digunakan men train decision tree
    def train_model(self, X, y, max_depth=3):
        """Train decision tree model"""
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42
        )
        self.model.fit(X, y)
        return self.model
    
    def visualize_tree(self):
        """Visualisasi decision tree"""
        dot_data = StringIO()
        export_graphviz(
            self.model,
            out_file=dot_data,
            feature_names=self.feature_names,
            class_names=[str(x) for x in self.model.classes_],
            filled=True,
            rounded=True,
            special_characters=True
        )
        
        graph = graphviz.Source(dot_data.getvalue())
        return graph
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        if self.model is None:
            return None
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df['feature'], importance_df['importance'])
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        plt.tight_layout()
        return fig

def main():
    st.title("🌳 Decision Tree Classifier dengan UI/UX")
    st.markdown("Aplikasi interaktif untuk membangun dan memvisualisasikan Decision Tree")
    
    app = DecisionTreeApp()
    
    # Sidebar untuk input
    st.sidebar.header("📊 Konfigurasi Data")
    
    # Pilihan data source
    data_source = st.sidebar.radio(
        "Pilih Sumber Data:",
        ["Sample Data", "Upload CSV"]
    )
    
    df = None
    
    if data_source == "Sample Data":
        df = app.load_sample_data()
        st.sidebar.success("Menggunakan sample data")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("File berhasil diupload")
    
    if df is not None:
        # Tampilkan data
        st.subheader("📋 Preview Data")
        st.dataframe(df.head(), use_container_width=True)
        
        st.write(f"**Shape:** {df.shape}")
        
        # Pilih target column
        target_column = st.sidebar.selectbox(
            "Pilih Target Column:",
            df.columns
        )
        
        # Konfigurasi model
        st.sidebar.header("⚙️ Konfigurasi Model")
        max_depth = st.sidebar.slider("Max Depth:", 1, 10, 3)
        test_size = st.sidebar.slider("Test Size:", 0.1, 0.5, 0.2, 0.05)
        
        if st.sidebar.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                # Preprocessing
                df_processed = app.preprocess_data(df, target_column)
                
                # Split features and target
                X = df_processed.drop(columns=[target_column])
                y = df_processed[target_column]
                
                app.feature_names = X.columns.tolist()
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Train model
                app.train_model(X_train, y_train, max_depth)
                
                # Predictions
                y_pred = app.model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Tampilkan hasil
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Akurasi", f"{accuracy:.2%}")
                
                with col2:
                    st.metric("Sample Latihan", len(X_train))
                
                with col3:
                    st.metric("Test Sample", len(X_test))
                
                # Visualisasi tree
                st.subheader("🌿 Visualisasi Decision Tree")
                tree_graph = app.visualize_tree()
                st.graphviz_chart(tree_graph, use_container_width=True)
                
                # Feature importance
                st.subheader("📈 Fitur Penting")
                fig = app.plot_feature_importance()
                if fig:
                    st.pyplot(fig)
                
                # Classification report
                st.subheader("📊 Laporan Klasifikasi")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
        
        # Prediction interface
        st.sidebar.header("🔮 Prediksi")
        
        if app.model is not None:
            st.subheader("Prediksi Data Baru")
            
            input_data = {}
            col1, col2 = st.columns(2)
            
            for i, feature in enumerate(app.feature_names):
                if i % 2 == 0:
                    with col1:
                        if feature in app.le_dict:
                            # Untuk categorical features
                            options = list(app.le_dict[feature].classes_)
                            input_data[feature] = st.selectbox(f"{feature}:", options)
                        else:
                            # Untuk numerical features
                            min_val = float(df[feature].min())
                            max_val = float(df[feature].max())
                            input_data[feature] = st.number_input(
                                f"{feature}:", 
                                min_value=min_val, 
                                max_value=max_val,
                                value=float(df[feature].mean())
                            )
                else:
                    with col2:
                        if feature in app.le_dict:
                            options = list(app.le_dict[feature].classes_)
                            input_data[feature] = st.selectbox(f"{feature}:", options)
                        else:
                            min_val = float(df[feature].min())
                            max_val = float(df[feature].max())
                            input_data[feature] = st.number_input(
                                f"{feature}:", 
                                min_value=min_val, 
                                max_value=max_val,
                                value=float(df[feature].mean())
                            )
            
            if st.button("Predict"):
                # Prepare input for prediction
                input_df = pd.DataFrame([input_data])
                
                # Encode categorical inputs
                for feature in app.feature_names:
                    if feature in app.le_dict:
                        input_df[feature] = app.le_dict[feature].transform([input_df[feature].iloc[0]])[0]
                
                # Make prediction
                prediction = app.model.predict(input_df)[0]
                probability = app.model.predict_proba(input_df)[0]
                
                # Decode prediction
                target_le = app.le_dict[target_column]
                predicted_class = target_le.inverse_transform([prediction])[0]
                
                # Display results
                st.success(f"**Prediksi** {predicted_class}")
                
                # Probability chart
                prob_df = pd.DataFrame({
                    'Class': target_le.classes_,
                    'Probability': probability
                })
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.barh(prob_df['Class'], prob_df['Probability'])
                ax.set_xlabel('Probability')
                ax.set_title('Prediction Probabilities')
                plt.tight_layout()
                st.pyplot(fig)

if __name__ == "__main__":
    main()