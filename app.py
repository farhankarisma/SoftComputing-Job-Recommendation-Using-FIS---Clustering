import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Hybrid Fuzzy Job Recommendation System",
    page_icon="💼",
    layout="wide"
)

class FuzzyJobRecommendationSystem:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.clusters = {}
        self.membership_functions = {}
        self.fis = None
        self.job_categories = [
            "UI/UX Designer", "Data Analyst", "Data Scientist", 
            "Backend Developer", "Frontend Developer", "Full-stack Developer",
            "Game Developer", "Cybersecurity Analyst", "AI/ML Engineer",
            "DevOps Engineer", "Mobile Developer", "Product Manager"
        ]
        
    def load_data(self, uploaded_file=None):
        """Load and preprocess the dataset"""
        if uploaded_file is not None:
            self.data = pd.read_csv(uploaded_file)
        else:
            # Use sample data for demo
            self.data = self._create_sample_data()
        
        # Clean column names
        self.data.columns = [col.strip() for col in self.data.columns]
        return self.data
    
    
    def _create_sample_data(self):
        """Create sample data based on the CSV structure"""
        sample_data = {
            'Timestamp': ['5/31/2025 15:44:08', '5/31/2025 15:48:06', '5/31/2025 15:48:28'],
            'Berapa rata-rata IPK anda selama anda berkuliah': ['3.51 - 4.00', '3.51 - 4.00', '3.51 - 4.00'],
            'dari skala 1-5 seberapa minat kamu ke bidang pengembangan web?': [3, 4, 4],
            'dari skala 1-5 seberapa minat kamu ke bidang Data Science?': [5, 5, 2],
            'dari skala 1-5 seberapa minat kamu ke bidang kecerdasan buatan?': [5, 3, 2],
            'dari skala 1-5 seberapa minat kamu ke bidang UI/UX?': [2, 5, 4],
            'dari skala 1-5 seberapa minat kamu ke bidang Game Development?': [2, 2, 1],
            'dari skala 1-5 seberapa minat kamu ke bidang Cyber Security (1-5)': [2, 2, 3],
            'Ketika bekerja dalam tim, kamu lebih suka': ['Memimpin diskusi', 'Menyusun rencana', 'Menyusun rencana'],
            'Saya lebih suka bekerja dalam suasana': ['Netral', 'Netral', 'Netral']
        }
        return pd.DataFrame(sample_data)
    
    def preprocess_data(self):
        """Preprocess data for clustering and FIS"""
        df = self.data.copy()
        
        # Map IPK to numeric values
        ipk_mapping = {
            '< 2.50': 2.25,
            '2.51 - 3.00': 2.75,
            '3.01 - 3.50': 3.25,
            '3.51 - 4.00': 3.75
        }
        
        # Map team style
        team_mapping = {
            'Memimpin diskusi': 'Tim',
            'Menyusun rencana': 'Tim',
            'Mengerjakan bagian teknis': 'Tim/Individu',
            'Bekerja sendiri': 'Individu'
        }
        
        # Map atmosphere
        atmosphere_mapping = {
            'Sangat tenang': 'Fokus',
            'Netral': 'Netral',
            'Ramai dan dinamis': 'Dinamis'
        }
        
        # Process data
        processed = pd.DataFrame()
        processed['IPK'] = df.iloc[:, 1].map(ipk_mapping)
        processed['WebDev'] = df.iloc[:, 2]
        processed['DataScience'] = df.iloc[:, 3]
        processed['AI'] = df.iloc[:, 4]
        processed['UIUX'] = df.iloc[:, 5]
        processed['GameDev'] = df.iloc[:, 6]
        processed['CyberSec'] = df.iloc[:, 7]
        processed['TeamStyle'] = df.iloc[:, 8].map(team_mapping)
        processed['Atmosphere'] = df.iloc[:, 9].map(atmosphere_mapping)
        
        self.processed_data = processed
        return processed
    
    def fuzzy_clustering(self, feature, n_clusters=5):
        """Perform fuzzy K-means clustering"""
        data = self.processed_data[feature].values.reshape(-1, 1)
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Perform fuzzy clustering
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data_scaled.T, n_clusters, 2, error=0.005, maxiter=1000, init=None
        )
        
        # Calculate cluster centers in original scale
        centers_original = scaler.inverse_transform(cntr)
        
        self.clusters[feature] = {
            'centers': centers_original.flatten(),
            'membership': u,
            'fpc': fpc,
            'scaler': scaler
        }
        
        return self.clusters[feature]
    
    # def generate_membership_functions(self, feature):
    #     """Generate trapezoidal membership functions from clustering results"""
    #     if feature not in self.clusters:
    #         self.fuzzy_clustering(feature)
        
    #     centers = sorted(self.clusters[feature]['centers'])
    #     min_val = self.processed_data[feature].min()
    #     max_val = self.processed_data[feature].max()
        
    #     # Generate trapezoidal membership functions
    #     mf_params = []
    #     labels = ["Very Low", "Low", "Medium", "High", "Very High"]
        
    #     for i, center in enumerate(centers):
    #         if i == 0:
    #             # First MF: trapezoid starting from min
    #             a = min_val
    #             b = min_val
    #             c = center
    #             d = (center + centers[i+1]) / 2 if i+1 < len(centers) else center + (center - min_val) / 2
    #         elif i == len(centers) - 1:
    #             # Last MF: trapezoid ending at max
    #             a = (centers[i-1] + center) / 2
    #             b = center
    #             c = max_val
    #             d = max_val
    #         else:
    #             # Middle MFs
    #             a = (centers[i-1] + center) / 2
    #             b = center
    #             c = center
    #             d = (center + centers[i+1]) / 2
            
    #         mf_params.append([a, b, c, d])
        
    #     self.membership_functions[feature] = {
    #         'params': mf_params,
    #         'labels': labels[:len(centers)]
    #     }
        
    #     return self.membership_functions[feature]
    
    def create_fuzzy_system(self):
        """Create Mamdani fuzzy inference system"""
        # Create input variables
        webdev = ctrl.Antecedent(np.arange(1, 6, 0.1), 'webdev')
        datascience = ctrl.Antecedent(np.arange(1, 6, 0.1), 'datascience')
        ai = ctrl.Antecedent(np.arange(1, 6, 0.1), 'ai')
        uiux = ctrl.Antecedent(np.arange(1, 6, 0.1), 'uiux')
        gamedev = ctrl.Antecedent(np.arange(1, 6, 0.1), 'gamedev')
        cybersec = ctrl.Antecedent(np.arange(1, 6, 0.1), 'cybersec')
        ipk = ctrl.Antecedent(np.arange(2.0, 4.1, 0.1), 'ipk')
        
        # Create output variable
        job_score = ctrl.Consequent(np.arange(0, 101, 1), 'job_score')
        
        # Define membership functions for inputs
        for var in [webdev, datascience, ai, uiux, gamedev, cybersec]:
            var['very_low'] = fuzz.trimf(var.universe, [1, 1, 2])
            var['low'] = fuzz.trimf(var.universe, [1, 2, 3])
            var['medium'] = fuzz.trimf(var.universe, [2, 3, 4])
            var['high'] = fuzz.trimf(var.universe, [3, 4, 5])
            var['very_high'] = fuzz.trimf(var.universe, [4, 5, 5])
        
        # IPK membership functions
        ipk['low'] = fuzz.trimf(ipk.universe, [2.0, 2.0, 2.75])
        ipk['medium_low'] = fuzz.trimf(ipk.universe, [2.25, 2.75, 3.25])
        ipk['medium'] = fuzz.trimf(ipk.universe, [2.75, 3.25, 3.75])
        ipk['high'] = fuzz.trimf(ipk.universe, [3.25, 3.75, 4.0])
        ipk['very_high'] = fuzz.trimf(ipk.universe, [3.5, 4.0, 4.0])
        
        # Output membership functions
        job_score['very_low'] = fuzz.trimf(job_score.universe, [0, 0, 25])
        job_score['low'] = fuzz.trimf(job_score.universe, [0, 25, 50])
        job_score['medium'] = fuzz.trimf(job_score.universe, [25, 50, 75])
        job_score['high'] = fuzz.trimf(job_score.universe, [50, 75, 100])
        job_score['very_high'] = fuzz.trimf(job_score.universe, [75, 100, 100])
        
        # Create rules based on clustering results
        rules = self._generate_fuzzy_rules(webdev, datascience, ai, uiux, gamedev, cybersec, ipk, job_score)
        
        # Create control system
        self.fis = ctrl.ControlSystem(rules)
        
        return self.fis
    
    def _generate_fuzzy_rules(self, webdev, datascience, ai, uiux, gamedev, cybersec, ipk, job_score):
        """Generate fuzzy rules automatically based on job categories"""
        rules = []
        
        # Rule 1: UI/UX Designer
        rules.append(ctrl.Rule(uiux['high'] | uiux['very_high'], job_score['high']))
        
        # Rule 2: Data Scientist
        rules.append(ctrl.Rule(datascience['high'] & ai['high'], job_score['very_high']))
        
        # Rule 3: Backend Developer
        rules.append(ctrl.Rule(webdev['high'] & (datascience['medium'] | ai['medium']), job_score['high']))
        
        # Rule 4: Game Developer
        rules.append(ctrl.Rule(gamedev['high'] | gamedev['very_high'], job_score['high']))
        
        # Rule 5: Cybersecurity Analyst
        rules.append(ctrl.Rule(cybersec['high'] | cybersec['very_high'], job_score['high']))
        
        # Rule 6: General high performer
        rules.append(ctrl.Rule(ipk['high'] | ipk['very_high'], job_score['medium']))
        
        # Add more rules as needed
        
        return rules
    
    def predict_job(self, input_data):
        """Predict job recommendation using FIS"""
        if self.fis is None:
            self.create_fuzzy_system()
        
        simulation = ctrl.ControlSystemSimulation(self.fis)
        
        # Set inputs
        simulation.input['webdev'] = input_data['WebDev']
        simulation.input['datascience'] = input_data['DataScience']
        simulation.input['ai'] = input_data['AI']
        simulation.input['uiux'] = input_data['UIUX']
        simulation.input['gamedev'] = input_data['GameDev']
        simulation.input['cybersec'] = input_data['CyberSec']
        simulation.input['ipk'] = input_data['IPK']
        
        # Compute result
        simulation.compute()
        
        return simulation.output['job_score']
    
    def calculate_scf_validation(self):
        """Calculate Silhouette Coefficient for Fuzzy clustering validation"""
        scf_scores = {}
        
        for feature in ['WebDev', 'DataScience', 'AI', 'UIUX', 'GameDev', 'CyberSec']:
            if feature in self.clusters:
                # Get cluster assignments
                membership = self.clusters[feature]['membership']
                cluster_labels = np.argmax(membership, axis=0)
                
                # Calculate silhouette score
                data = self.processed_data[feature].values.reshape(-1, 1)
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data)
                
                if len(np.unique(cluster_labels)) > 1:
                    scf_scores[feature] = silhouette_score(data_scaled, cluster_labels)
                else:
                    scf_scores[feature] = 0
        
        return scf_scores

# Streamlit App
def main():
    st.title("💼 Hybrid Fuzzy Clustering and Inference System for IT Job Recommendation")
    st.subheader("Universitas Padjadjaran")
    
    # Initialize system
    if 'system' not in st.session_state:
        st.session_state.system = FuzzyJobRecommendationSystem()
    
    system = st.session_state.system
    
    # Sidebar
    st.sidebar.header("⚙️ System Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=['csv'])
    
    # Load data
    if st.sidebar.button("Load Data") or uploaded_file:
        with st.spinner("Loading data..."):
            data = system.load_data(uploaded_file)
            st.success(f"Data loaded successfully! Shape: {data.shape}")
    
    # Main content
    if system.data is not None:
        tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🔍 Fuzzy Clustering", "⚡ FIS Inference"])
        
        with tab1:
            st.header("Data Overview")
            
            # Preprocess data
            if st.button("Preprocess Data"):
                processed_data = system.preprocess_data()
                st.success("Data preprocessed successfully!")
            
            if system.processed_data is not None:
                st.subheader("Processed Dataset")
                st.dataframe(system.processed_data)
                
                st.subheader("Data Statistics")
                st.dataframe(system.processed_data.describe())
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Interest Distribution")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    interest_cols = ['WebDev', 'DataScience', 'AI', 'UIUX', 'GameDev', 'CyberSec']
                    system.processed_data[interest_cols].boxplot(ax=ax)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    st.subheader("IPK Distribution")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    system.processed_data['IPK'].hist(bins=20, ax=ax)
                    plt.xlabel('IPK')
                    plt.ylabel('Frequency')
                    st.pyplot(fig)
        
        with tab2:
            st.header("Fuzzy Clustering Analysis")
            
            # if system.processed_data is not None:
            #     feature_select = st.selectbox("Select Feature for Clustering", 
            #                                 ['WebDev', 'DataScience', 'AI', 'UIUX', 'GameDev', 'CyberSec', 'IPK'])
                
            #     if st.button(f"Perform Fuzzy Clustering on {feature_select}"):
            #         with st.spinner("Performing fuzzy clustering..."):
            #             cluster_result = system.fuzzy_clustering(feature_select)
            #             st.success(f"Clustering completed! FPC: {cluster_result['fpc']:.3f}")
                        
            #             # Display cluster centers
            #             st.subheader("Cluster Centers")
            #             centers_df = pd.DataFrame({
            #                 'Cluster': [f"Cluster {i+1}" for i in range(len(cluster_result['centers']))],
            #                 'Center': cluster_result['centers']
            #             })
            #             st.dataframe(centers_df)
                        
            #             # Visualization
            #             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
            #             # Scatter plot with cluster membership
            #             membership = cluster_result['membership']
            #             max_membership = np.argmax(membership, axis=0)
                        
            #             scatter = ax1.scatter(range(len(system.processed_data[feature_select])), 
            #                                system.processed_data[feature_select], 
            #                                c=max_membership, cmap='viridis', alpha=0.7)
            #             ax1.set_xlabel('Data Point Index')
            #             ax1.set_ylabel(feature_select)
            #             ax1.set_title(f'Fuzzy Clustering Results - {feature_select}')
            #             plt.colorbar(scatter, ax=ax1)
                        
            #             # Membership degrees
            #             for i in range(len(cluster_result['centers'])):
            #                 ax2.plot(membership[i], label=f'Cluster {i+1}', alpha=0.7)
            #             ax2.set_xlabel('Data Point Index')
            #             ax2.set_ylabel('Membership Degree')
            #             ax2.set_title('Membership Degrees')
            #             ax2.legend()
                        
            #             st.pyplot(fig)
            if st.button("🔁 Perform Fuzzy Clustering for All Features"):
                    all_features = ['WebDev', 'DataScience', 'AI', 'UIUX', 'GameDev', 'CyberSec', 'IPK']
                    
                    for feature_select in all_features:
                        with st.spinner(f"Clustering {feature_select}..."):
                            cluster_result = system.fuzzy_clustering(feature_select)
                            st.success(f"[{feature_select}] Clustering completed! FPC: {cluster_result['fpc']:.3f}")
                            
                            # Tampilkan Cluster Centers
                            st.subheader(f"📍 Cluster Centers - {feature_select}")
                            centers_df = pd.DataFrame({
                                'Cluster': [f"Cluster {i+1}" for i in range(len(cluster_result['centers']))],
                                'Center': cluster_result['centers']
                            })
                            st.dataframe(centers_df)

                            # Visualisasi
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                            
                            # Scatter plot dengan cluster membership
                            membership = cluster_result['membership']
                            max_membership = np.argmax(membership, axis=0)

                            ax1.scatter(range(len(system.processed_data[feature_select])), 
                                        system.processed_data[feature_select], 
                                        c=max_membership, cmap='viridis', alpha=0.7)
                            ax1.set_xlabel('Data Point Index')
                            ax1.set_ylabel(feature_select)
                            ax1.set_title(f'Fuzzy Clustering Results - {feature_select}')

                            plt.colorbar(ax1.collections[0], ax=ax1)

                            # Membership degrees plot
                            for i in range(len(cluster_result['centers'])):
                                ax2.plot(membership[i], label=f'Cluster {i+1}', alpha=0.7)
                            ax2.set_xlabel('Data Point Index')
                            ax2.set_ylabel('Membership Degree')
                            ax2.set_title('Membership Degrees')
                            ax2.legend()

                            st.pyplot(fig)
            else:
                st.info("Preprocess the data before starting clustering.")
        with tab3:
            st.header("Fuzzy Inference System")
            
            if system.processed_data is not None:
                if st.button("Create Fuzzy Inference System"):
                    with st.spinner("Creating FIS..."):
                        fis = system.create_fuzzy_system()
                        st.success("Fuzzy Inference System created successfully!")
                
                st.subheader("Job Recommendation Prediction")
                
                # Input form for prediction
                col1, col2 = st.columns(2)
                
                with col1:
                    webdev_input = st.slider("Web Development Interest", 1, 5, 3)
                    datascience_input = st.slider("Data Science Interest", 1, 5, 3)
                    ai_input = st.slider("AI Interest", 1, 5, 3)
                    uiux_input = st.slider("UI/UX Interest", 1, 5, 3)
                
                with col2:
                    gamedev_input = st.slider("Game Development Interest", 1, 5, 3)
                    cybersec_input = st.slider("Cybersecurity Interest", 1, 5, 3)
                    ipk_input = st.slider("IPK", 2.0, 4.0, 3.5)
                
                if st.button("Predict Job Recommendation"):
                    try:
                        input_data = {
                            'WebDev': webdev_input,
                            'DataScience': datascience_input,
                            'AI': ai_input,
                            'UIUX': uiux_input,
                            'GameDev': gamedev_input,
                            'CyberSec': cybersec_input,
                            'IPK': ipk_input
                        }
                        
                        score = system.predict_job(input_data)
                        
                        st.subheader("Prediction Results")
                        st.metric("Job Recommendation Score", f"{score:.2f}/100")
                        
                        # Recommend based on highest interests
                        interests = {
                            'UI/UX Designer': uiux_input,
                            'Data Scientist': (datascience_input + ai_input) / 2,
                            'Backend Developer': webdev_input,
                            'Game Developer': gamedev_input,
                            'Cybersecurity Analyst': cybersec_input,
                            'Full-stack Developer': webdev_input
                        }
                        
                        top_jobs = sorted(interests.items(), key=lambda x: x[1], reverse=True)[:3]
                        
                        st.subheader("Top Job Recommendations")
                        for i, (job, interest_score) in enumerate(top_jobs, 1):
                            st.write(f"{i}. **{job}** (Interest Score: {interest_score}/5)")
                            
                    except Exception as e:
                        st.error(f"Error in prediction: {str(e)}")
            else:
                st.info("Cluster the data before generating recommendations.")
    else:
        st.info("Please load a dataset to begin analysis.")

if __name__ == "__main__":
    main()