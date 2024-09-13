import streamlit as st
import pandas as pd
import plotly.express as px
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Δημιουργία πλαϊνής μπάρας για επιλογές
page = st.sidebar.selectbox("Επιλέξτε σελίδα:", ["Αρχική","Info"])
if page == "Info":
    st.header("Πληροφορίες Σχετικά με την Εφαρμογή")

    st.subheader("Σχετικά με την Εφαρμογή")
    st.write("""
    Αυτή η εφαρμογή δημιουργήθηκε για την ανάλυση δεδομένων και την εφαρμογή αλγορίθμων μηχανικής μάθησης. Παρέχει εργαλεία για την οπτικοποίηση δεδομένων, την εκτέλεση EDA, και την εφαρμογή αλγορίθμων ταξινόμησης.
    """)

    st.subheader("Τρόπος Λειτουργίας")
    st.write("""
    - **2D/3D Οπτικοποίηση**: Ο χρήστης μπορεί να επιλέξει μεθόδους μείωσης διαστάσεων όπως PCA και UMAP και να οπτικοποιήσει τα δεδομένα του σε 2 ή 3 διαστάσεις.
    - **Exploratory Data Analysis (EDA)**: Παρέχει εργαλεία για τη δημιουργία γραφημάτων και την εξερεύνηση των δεδομένων.
    - **Μηχανική Μάθηση**: Δυνατότητα εφαρμογής αλγορίθμων επιλογής χαρακτηριστικών και ταξινόμησης, με συγκριτική αξιολόγηση των αποτελεσμάτων.
    """)

    st.subheader("Ομάδα Ανάπτυξης")
    st.write("""
    Αυτή η εφαρμογή αναπτύχθηκε από την ομάδα μας:
    """)

    team_info = {
        "ΓΕΤΕΡΙΔΟΥ ΑΝΑΣΤΑΣΙΑ": "Συμμετείχε σε όλη την διαδικασία ανάπτυξης της εφαρμογής.",
        "ΠΙΣΠΑΣ ΓΕΩΡΓΙΟΣ": "Συμμετείχε σε όλη την διαδικασία ανάπτυξης της εφαρμογής.",
        "ΒΑΒΛΙΑΡΑ ΚΥΡΙΑΚΗ": "Συμμετείχε σε όλη την διαδικασία ανάπτυξης της εφαρμογής.",
    }

    for member, task in team_info.items():
        st.write(f"- **{member}**: {task}")

elif page == "Αρχική":
# Φόρτωση αρχείου
    uploaded_file = st.file_uploader("Φορτώστε ένα αρχείο δεδομένων", type=['csv', 'xlsx', 'tsv'])
    file_type = st.selectbox("Επιλέξτε τύπο αρχείου:", ["CSV", "Excel", "TSV"])

    if uploaded_file is not None:
        try:
        # Ανάγνωση αρχείου ανάλογα με τον τύπο του
            if file_type == 'CSV':
                df = pd.read_csv(uploaded_file)
            elif file_type == 'Excel':
                df = pd.read_excel(uploaded_file)
            elif file_type == 'TSV':
                df = pd.read_csv(uploaded_file, sep='\t')

            st.write(df)

        # Δεύτερο ερώτημα - Έλεγχος διαστάσεων πίνακα
            if df.shape[1] < 2:
                st.error("Μεγεθος S x F, με F>=2")
            else:
                st.write(f"Ο πίνακας περιέχει {df.shape[0]} δείγματα και {df.shape[1] - 1} χαρακτηριστικά.")
                st.write(f"Η τελευταία στήλη θεωρείται ετικέτα.")

            # Διαχωρισμός σε Χαρακτηριστικά (X) και Ετικέτα (y)
                X = df.iloc[:, :-1]  # Χαρακτηριστικά
                y = df.iloc[:, -1]  # Ετικέτα

            # Μετατροπή των ονομάτων στηλών σε string
                X.columns = X.columns.astype(str)

            # Συμπλήρωση των NaN τιμών με τον μέσο όρο κάθε στήλης
                imputer = SimpleImputer(strategy='mean')
                X_imputed = imputer.fit_transform(X)

            # Έλεγχος τύπου ετικετών και μετατροπή αν χρειάζεται
                if y.dtype in ['float64', 'int64']:
                # Προσπάθεια μετατροπής σε κατηγορικές τιμές
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    st.write("Οι ετικέτες έχουν μετατραπεί σε κατηγορικές τιμές.")
                else:
                    y = y.astype('category').cat.codes
                    st.write("Οι ετικέτες είναι ήδη κατηγορικές.")

                st.write("Χαρακτηριστικά (μετά τη συμπλήρωση των NaN):")
                st.dataframe(X)

                st.write("Ετικέτα:")
                st.dataframe(y)

            # Επιλογή λειτουργίας
                option = st.selectbox("Επιλέξτε τη λειτουργία:",
                                  ["2D/3D Οπτικοποίηση", "Exploratory Data Analysis (EDA)", "Μηχανική Μάθηση"])

                if option == "2D/3D Οπτικοποίηση":
                    st.header("2D και 3D Οπτικοποιήσεις")

                # Κανονικοποίηση των χαρακτηριστικών
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_imputed)

                # Επιλογή μεθόδου μείωσης διάστασης
                    method = st.selectbox("Επιλέξτε μέθοδο μείωσης διάστασης:", ["PCA", "UMAP"])
                    dimensions = st.radio("Επιλέξτε διαστάσεις:", [2, 3])

                # Εφαρμογή της επιλεγμένης μεθόδου
                    if method == "PCA":
                        pca = PCA(n_components=dimensions)
                        components = pca.fit_transform(X_scaled)
                    elif method == "UMAP":
                        umap_model = umap.UMAP(n_components=dimensions)
                        components = umap_model.fit_transform(X_scaled)

                # Οπτικοποίηση ανάλογα με τις διαστάσεις
                    if dimensions == 2:
                        fig = px.scatter(components, x=0, y=1, color=y, labels={'0': 'Component 1', '1': 'Component 2'})
                        st.plotly_chart(fig)

                    elif dimensions == 3:
                        fig = px.scatter_3d(components, x=0, y=1, z=2, color=y,
                                        labels={'0': 'Component 1', '1': 'Component 2', '2': 'Component 3'})
                        st.plotly_chart(fig)

                elif option == "Exploratory Data Analysis (EDA)":
                    st.header("Exploratory Data Analysis (EDA)")

                # Διάγραμμα Κατανομής (Distribution Plot) για όλα τα χαρακτηριστικά
                    st.subheader("Διάγραμμα Κατανομής Χαρακτηριστικών")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for col in X.columns:
                        sns.histplot(X[col], kde=True, ax=ax, label=col)
                    ax.legend()
                    st.pyplot(fig)

                # Διάγραμμα Ζεύξης (Pair Plot) αν υπάρχουν λιγότεροι από 10 χαρακτηριστικά
                    if X.shape[1] <= 10:
                        st.subheader("Διάγραμμα Ζεύξης Χαρακτηριστικών")
                        pairplot_fig = sns.pairplot(pd.concat([X, pd.Series(y, name='Label')], axis=1), hue='Label')
                        st.pyplot(pairplot_fig.figure)
                    else:
                        st.warning("Διάγραμμα Ζεύξης δεν μπορεί να δημιουργηθεί με περισσότερα από 10 χαρακτηριστικά.")

                elif option == "Μηχανική Μάθηση":
                    st.header("Μηχανική Μάθηση")
                    ml_tabs = st.tabs(["Επιλογή Χαρακτηριστικών", "Κατηγοριοποίηση"])

                # Tab 1: Επιλογή Χαρακτηριστικών
                    with ml_tabs[0]:
                        st.subheader("Επιλογή Χαρακτηριστικών")
                        num_features = st.slider("Επιλέξτε αριθμό χαρακτηριστικών:", 1, X.shape[1])

                        selector = SelectKBest(score_func=f_classif, k=num_features)
                        X_selected = selector.fit_transform(X_imputed, y)

                        st.write(f"Δεδομένα μετά την επιλογή {num_features} χαρακτηριστικών:")
                        st.dataframe(pd.DataFrame(X_selected, columns=[f'Feature {i}' for i in range(X_selected.shape[1])]))

                # Tab 2: Κατηγοριοποίηση
                    with ml_tabs[1]:
                        st.subheader("Κατηγοριοποίηση")

                    # Διαχωρισμός δεδομένων σε train/test
                        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)
                        X_train_selected, X_test_selected, _, _ = train_test_split(X_selected, y, test_size=0.3,
                                                                               random_state=42)

                    # Επιλογή αλγορίθμων και παραμέτρων
                        classifier_name = st.selectbox("Επιλέξτε αλγόριθμο:", ["KNN", "SVM"])

                        if classifier_name == "KNN":
                            k = st.slider("Επιλέξτε το k για KNN:", 1, 15, value=3)
                            classifier = KNeighborsClassifier(n_neighbors=k)
                        elif classifier_name == "SVM":
                            c = st.slider("Επιλέξτε την παράμετρο C για SVM:", 0.01, 10.0, value=1.0)
                            multi_class = st.selectbox("Επιλέξτε τη στρατηγική multi-class για SVM:", ['ovo', 'ovr'])
                            classifier = SVC(C=c, probability=True, decision_function_shape=multi_class)

                    # Εκπαίδευση και αξιολόγηση χωρίς επιλογή χαρακτηριστικών
                        classifier.fit(X_train, y_train)
                        y_pred = classifier.predict(X_test)
                        y_prob = classifier.predict_proba(X_test)[:, 1] if hasattr(classifier, "predict_proba") else None

                        st.write("Αποτελέσματα χωρίς επιλογή χαρακτηριστικών:")
                        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
                        st.write(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
                        st.write(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}" if y_prob is not None else "ROC-AUC: Δεν είναι διαθέσιμο.")
                        st.text(classification_report(y_test, y_pred))

                    # Εκπαίδευση και αξιολόγηση με επιλογή χαρακτηριστικών
                        classifier.fit(X_train_selected, y_train)
                        y_pred_selected = classifier.predict(X_test_selected)
                        y_prob_selected = classifier.predict_proba(X_test_selected)[:, 1] if hasattr(classifier, "predict_proba") else None

                        st.write("Αποτελέσματα με επιλογή χαρακτηριστικών:")
                        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_selected):.4f}")
                        st.write(f"F1-Score: {f1_score(y_test, y_pred_selected, average='weighted'):.4f}")
                        st.write(f"ROC-AUC: {roc_auc_score(y_test, y_prob_selected):.4f}" if y_prob_selected is not None else "ROC-AUC: Δεν είναι διαθέσιμο.")
                        st.text(classification_report(y_test, y_pred_selected))

                    # Σύγκριση των αποτελεσμάτων
                        st.subheader("Σύγκριση Αποτελεσμάτων")
                        st.write("Σύγκριση αποτελεσμάτων πριν και μετά την επιλογή χαρακτηριστικών:")

                        st.write("Χωρίς Επιλογή Χαρακτηριστικών:")
                        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
                        st.write(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
                        st.write(f"ROC-AUC: {roc_auc_score(y_test, y_pred):.4f}" if y_prob is not None else "ROC-AUC: Δεν είναι διαθέσιμο.")

                        st.write("Με Επιλογή Χαρακτηριστικών:")
                        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_selected):.4f}")
                        st.write(f"F1-Score: {f1_score(y_test, y_pred_selected, average='weighted'):.4f}")
                        st.write(f"ROC-AUC: {roc_auc_score(y_test, y_pred_selected):.4f}" if y_prob_selected is not None else "ROC-AUC: Δεν είναι διαθέσιμο.")

                    # Συμπέρασμα
                        if accuracy_score(y_test, y_pred_selected) > accuracy_score(y_test, y_pred):
                            st.write("Η επιλογή χαρακτηριστικών βελτίωσε την ακρίβεια του μοντέλου.")
                        else:
                            st.write("Η επιλογή χαρακτηριστικών δεν βελτίωσε την ακρίβεια του μοντέλου.")

                        if f1_score(y_test, y_pred_selected, average='weighted') > f1_score(y_test, y_pred, average='weighted'):
                            st.write("Η επιλογή χαρακτηριστικών βελτίωσε το F1-Score του μοντέλου.")
                        else:
                            st.write("Η επιλογή χαρακτηριστικών δεν βελτίωσε το F1-Score του μοντέλου.")

                        if roc_auc_score(y_test, y_prob_selected) > roc_auc_score(y_test, y_prob):
                            st.write("Η επιλογή χαρακτηριστικών βελτίωσε το ROC-AUC του μοντέλου.")
                        else:
                            st.write("Η επιλογή χαρακτηριστικών δεν βελτίωσε το ROC-AUC του μοντέλου.")

        except Exception as e:
            st.error(f"Σφάλμα κατά τη φόρτωση των δεδομένων: {e}")

    else:
        st.info("Παρακαλώ φορτώστε ένα αρχείο.")

