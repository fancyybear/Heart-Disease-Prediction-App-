# Heart Disease Detection App 🚀
I am excited to share my latest project: Heart Disease Detection App, a machine learning-based system designed to predict heart disease risk. Here's a detailed breakdown of the project:
# 1️⃣ Scope & Objectives
This app leverages patient health metrics to provide early warnings about heart disease, enhancing diagnostic efficiency and aiding healthcare professionals. Key goals include improving prediction accuracy and minimizing invasive procedures.
# 2️⃣ Data Sources & Preprocessing
Using the Cleveland Heart Disease dataset, the data underwent cleaning, scaling (via `StandardScaler`), and one-hot encoding for categorical features like chest pain type. Data was split into training, validation, and testing sets for robust model training.
# 3️⃣ Machine Learning Models
Implemented models include:
- **Logistic Regression:** Baseline performance.
- **Random Forest Classifier:** Non-linear relationships and feature importance.
- **Gradient Boosting Machines (GBMs):** High prediction accuracy.
- **Ensemble Learning:** Combines strengths of multiple models for robustness.
Techniques like grid search for hyperparameter tuning and cross-validation ensured optimal performance.
# 4️⃣ System Architecture
- **Input Layer:** Users provide health metrics via a Streamlit interface.
- **Processing Layer:** Features are preprocessed and passed through the trained ML model.
- **Output Layer:** Displays predictions and actionable insights.
# 5️⃣ Technologies & Tools
Developed using Python, Scikit-learn, Pandas, and NumPy for model building, with a Streamlit frontend for interactivity. Docker was employed for scalable deployment.
# 6️⃣ Results & Evaluation
- **Random Forest:** Achieved 92% accuracy with high recall.
- **Gradient Boosting:** AUC of 0.95.
The app reduced false negatives, a critical metric in healthcare.
# 7️⃣ Future Directions
Plans include exploring deep learning, adding genetic data, and integrating advanced interpretability tools like SHAP or LIME.
