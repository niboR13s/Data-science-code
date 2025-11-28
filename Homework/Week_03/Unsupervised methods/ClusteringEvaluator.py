# class for evaluating clustering models with various preprocessing options



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
# from sklearn.model_selection import ParameterGrid
# from sklearn.base import clone

# # Clustering Algorithms (Common ones)
# from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering

# class ClusteringEvaluator:
#     def __init__(self, name="Clustering Evaluator", random_state=42):
#         """
#         Initialize the evaluator for unsupervised clustering tasks.
#         """
#         self.name = name
#         self.random_state = random_state
#         self.models = {}
#         self.options = []
#         self.results = {}
#         # Unsupervised metrics (Higher is better for Silhouette/Calinski, Lower for Davies)
#         self.metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
        
#         warnings.filterwarnings("ignore")

#     def add_model(self, name, model, params=None):
#         """ 
#         Add a clustering model and a grid of parameters to test.
#         Example: params={'n_clusters': [2, 3, 4]} 
#         """
#         if params is None: params = {}
#         if hasattr(model, 'random_state'):
#             model.random_state = self.random_state
#         self.models[name] = {'model': model, 'params': params}

#     def add_option(self, scaling=True, method='standard'):
#         """ 
#         Add a preprocessing option.
#         :param scaling: Whether to scale data (Crucial for clustering!).
#         :param method: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler).
#         """
#         self.options.append({'scaling': scaling, 'method': method})

#     def run_experiments(self, df):
#         """
#         Runs the clustering experiments.
#         Iterates through options -> models -> parameter combinations.
#         """
#         self.results = {}
#         print(f"\n{'='*60}")
#         print(f"🚀 START CLUSTERING EXPERIMENT: {self.name.upper()}")
#         print(f"{'='*60}")
#         print(f"📊 Data Shape: {df.shape}")

#         for i, option in enumerate(self.options):
#             opt_name = f"Opt{i}_{'Scaled' if option['scaling'] else 'Raw'}"
#             print(f"\n--- ⚙️ Processing {opt_name} ---")

#             # 1. Preprocessing
#             X = df.copy()
#             if option['scaling']:
#                 if option['method'] == 'minmax':
#                     scaler = MinMaxScaler()
#                 else:
#                     scaler = StandardScaler()
#                 X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

#             # 2. Iterate Models
#             for model_name, config in self.models.items():
#                 # Generate all parameter combinations
#                 param_grid = list(ParameterGrid(config['params']))
                
#                 # If no params provided, run once with defaults
#                 if not param_grid: param_grid = [{}]

#                 print(f"   Testing {model_name} ({len(param_grid)} configurations)...")

#                 for params in param_grid:
#                     # Clone model to reset it
#                     model = clone(config['model'])
#                     model.set_params(**params)
                    
#                     try:
#                         # Fit & Predict
#                         labels = model.fit_predict(X)
                        
#                         # Check unique clusters (ignore noise -1)
#                         n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        
#                         # Metrics require at least 2 clusters to calculate
#                         if n_clusters < 2:
#                             # Skip calculation if everything is noise or 1 cluster
#                             scores = {m: np.nan for m in self.metrics}
#                         else:
#                             scores = {
#                                 'silhouette': silhouette_score(X, labels),
#                                 'davies_bouldin': davies_bouldin_score(X, labels),
#                                 'calinski_harabasz': calinski_harabasz_score(X, labels)
#                             }
                        
#                         # Store Results
#                         # Create a unique key for this run
#                         param_str = "_".join([f"{k}{v}" for k,v in params.items()])
#                         key = f"{opt_name}_{model_name}_{param_str}"
                        
#                         self.results[key] = {
#                             'Model': model_name,
#                             'Option': opt_name,
#                             'Params': params,
#                             'Labels': labels, # Store labels for plotting later
#                             'n_clusters': n_clusters,
#                             'n_noise': list(labels).count(-1),
#                             'X_processed': X, # Store processed data for plotting
#                             **scores
#                         }
                        
#                     except Exception as e:
#                         print(f"      ❌ Error with {params}: {e}")

#         print(f"\n🏁 Experiments completed! Total runs: {len(self.results)}")

#     def print_results(self, sort_by='silhouette', top_n=20):
#         """ Prints a sorted table of results. """
#         if not self.results:
#             print("No results to display.")
#             return

#         # Convert dictionary to DataFrame
#         # Filter out 'Labels' and 'X_processed' for the table view (too large)
#         simple_results = []
#         for k, v in self.results.items():
#             entry = {key: val for key, val in v.items() if key not in ['Labels', 'X_processed']}
#             # Flatten params into string for display
#             entry['Params_Text'] = str(v['Params'])
#             simple_results.append(entry)
            
#         df_res = pd.DataFrame(simple_results)
        
#         # Sorting logic (Silhouette/Calinski: High=Good, Davies: Low=Good)
#         ascending = True if sort_by == 'davies_bouldin' else False
        
#         if sort_by in df_res.columns:
#             df_res = df_res.sort_values(by=sort_by, ascending=ascending)
            
#         print(f"\n=== [{self.name}] Top {top_n} Results (Sorted by {sort_by}) ===")
        
#         cols = ['Model', 'Option', 'n_clusters', 'silhouette', 'davies_bouldin', 'Params_Text']
#         print(df_res[cols].head(top_n).to_string(index=False))
#         return df_res

#     # --- VISUALIZATION ---

#     def plot_clusters(self, top_n=3, sort_by='silhouette', reduction_method='pca'):
#         """
#         Visualizes the best clustering results in 2D.
#         :param reduction_method: 'pca' (fast) or 'tsne' (better for complex shapes).
#         """
#         # Get top N results
#         df_res = self.print_results(sort_by=sort_by, top_n=top_n) # Reuse sorting logic
#         if df_res.empty: return

#         # Setup plot grid
#         cols = min(top_n, 3)
#         rows = (top_n + cols - 1) // cols
#         fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
#         if top_n == 1: axes = [axes]
#         axes = np.array(axes).flatten()

#         for i in range(len(df_res)):
#             # Retrieve data for this run
#             # We need to find the key in self.results again or store it in df_res
#             # Let's iterate to find the matching result
#             # A safer way is to store the key in print_results, but here we'll just grab the i-th row
            
#             # Reconstruct the object from the stored results
#             # Note: This simple mapping assumes print_results returns the same order as we iterate
#             # Ideally we pass the result dictionary directly.
            
#             # Better approach: Sort internal results list
#             all_results_list = sorted(
#                 self.results.values(), 
#                 key=lambda x: x.get(sort_by, -1), 
#                 reverse=(sort_by != 'davies_bouldin')
#             )
#             res = all_results_list[i]
            
#             X = res['X_processed']
#             labels = res['Labels']
#             model_name = res['Model']
#             score = res[sort_by]
            
#             # Dimensionality Reduction to 2D
#             if X.shape[1] > 2:
#                 if reduction_method == 'tsne':
#                     reducer = TSNE(n_components=2, random_state=self.random_state)
#                 else:
#                     reducer = PCA(n_components=2)
                
#                 coords = reducer.fit_transform(X)
#                 x_axis, y_axis = coords[:, 0], coords[:, 1]
#                 label_prefix = reduction_method.upper()
#             else:
#                 x_axis, y_axis = X.iloc[:, 0], X.iloc[:, 1]
#                 label_prefix = "Feat"

#             # Plot
#             ax = axes[i]
#             sns.scatterplot(x=x_axis, y=y_axis, hue=labels, palette='tab10', ax=ax, s=50, legend='full')
            
#             title = f"{model_name}\nClusters: {res['n_clusters']} | {sort_by}: {score:.3f}"
#             ax.set_title(title, fontsize=12)
#             ax.set_xlabel(f"{label_prefix} 1")
#             ax.set_ylabel(f"{label_prefix} 2")
#             ax.grid(True, alpha=0.3)
            
#             # Move legend if too crowded
#             if res['n_clusters'] > 10:
#                 ax.get_legend().remove()

#         # Hide unused subplots
#         for j in range(i + 1, len(axes)):
#             axes[j].axis('off')
            
#         plt.tight_layout()
#         plt.show()

#     def plot_elbow_curve(self, model_name='KMeans', param_name='n_clusters', param_range=range(2, 15)):
#         """
#         Special plot for KMeans/Hierarchical to find optimal cluster number (Elbow Method).
#         Calculates 'Inertia' (for KMeans) or Silhouette for the range.
#         """
#         # Only works for KMeans easily (Inertia)
#         if model_name != 'KMeans':
#             print("⚠️ Elbow curve is typically for KMeans (Inertia). Using Silhouette instead.")
#             metric = 'silhouette'
#         else:
#             metric = 'inertia' # Special internal metric for KMeans

#         scores = []
        
#         # Use the first scaled option data
#         if not self.options: self.add_option()
#         opt = self.options[0] # Assume first option is best for elbow check
        
#         # Preprocess (Quickly redo here)
#         # ... (Assume df is passed or stored... wait, run_experiments passes df)
#         # This function needs 'df' input or we store 'X' in the class.
#         print("⚠️ Please call run_experiments first to process data, or modify this to accept df.")
#         return



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly.express as px # Voor de 3D plots

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone

# Veelgebruikte Clustering Algoritmes
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture

class ClusteringEvaluator:
    def __init__(self, name="Clustering Evaluator", random_state=42):
        self.name = name
        self.random_state = random_state
        self.models = {}
        self.options = []
        self.results = {}
        self.metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
        self.original_feature_names = [] # Opslaan voor PCA heatmap
        
        warnings.filterwarnings("ignore")

    def add_model(self, name, model, params=None):
        """ Add a model and parameter grid. """
        if params is None: params = {}
        if hasattr(model, 'random_state'):
            model.random_state = self.random_state
        self.models[name] = {'model': model, 'params': params}

    def add_option(self, scaling=True, method='standard', pca=False, n_components=2):
        """ 
        Add preprocessing option.
        :param pca: True/False om PCA toe te passen.
        :param n_components: Aantal PC's (int) of variantie (float, bv 0.95) om te behouden.
        """
        self.options.append({
            'scaling': scaling, 
            'method': method, 
            'pca': pca, 
            'n_components': n_components
        })

    def run_experiments(self, df):
        self.results = {}
        print(f"\n{'='*60}")
        print(f"🚀 START CLUSTERING EXPERIMENT: {self.name.upper()}")
        print(f"{'='*60}")
        print(f"📊 Data Shape: {df.shape}")
        
        # Bewaar originele kolomnamen voor visualisatie later
        self.original_feature_names = df.columns.tolist()

        for i, option in enumerate(self.options):
            opt_name = f"Opt{i}_{'Scaled' if option['scaling'] else 'Raw'}"
            if option['pca']:
                opt_name += f"_PCA({option['n_components']})"
                
            print(f"\n--- ⚙️ Processing {opt_name} ---")

            # 1. Preprocessing (Scaling)
            X = df.copy()
            if option['scaling']:
                if option['method'] == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()
                # We behouden dataframe structuur voor de features
                X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                X = X_scaled
            
            # 2. PCA (Optioneel)
            pca_obj = None
            if option['pca']:
                pca_obj = PCA(n_components=option['n_components'], random_state=self.random_state)
                X_pca = pca_obj.fit_transform(X)
                # Maak nieuwe DataFrame met PC namen
                n_pcs = X_pca.shape[1]
                cols = [f"PC{j+1}" for j in range(n_pcs)]
                X = pd.DataFrame(X_pca, columns=cols)
                print(f"   -> PCA toegepast: {len(self.original_feature_names)} features gereduceerd naar {n_pcs} componenten.")

            # 3. Modellen Itereren
            for model_name, config in self.models.items():
                param_grid = list(ParameterGrid(config['params']))
                if not param_grid: param_grid = [{}]

                print(f"   Testing {model_name} ({len(param_grid)} configurations)...")

                for params in param_grid:
                    model = clone(config['model'])
                    model.set_params(**params)
                    
                    try:
                        labels = model.fit_predict(X)
                        
                        # Tel unieke clusters (exclusief noise -1)
                        unique_labels = set(labels)
                        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
                        
                        scores = {m: np.nan for m in self.metrics}
                        if n_clusters >= 2:
                            scores = {
                                'silhouette': silhouette_score(X, labels),
                                'davies_bouldin': davies_bouldin_score(X, labels),
                                'calinski_harabasz': calinski_harabasz_score(X, labels)
                            }
                        
                        # Unieke key maken
                        param_str = "_".join([f"{k}{v}" for k,v in params.items()])
                        key = f"{opt_name}_{model_name}_{param_str}"
                        
                        self.results[key] = {
                            'Model': model_name,
                            'Option_ID': f"Opt{i}", # Korte ID voor filteren
                            'Option_Full': opt_name,
                            'Params': params,
                            'Labels': labels,
                            'n_clusters': n_clusters,
                            'n_noise': list(labels).count(-1),
                            'X_processed': X, # De data waarop getraind is (kan PCA zijn)
                            'pca_obj': pca_obj, # Het getrainde PCA object (voor plots)
                            **scores
                        }
                        
                    except Exception as e:
                        print(f"      ❌ Error with {params}: {e}")

        print(f"\n🏁 Experiments completed! Total runs: {len(self.results)}")

    def print_results(self, sort_by='silhouette', top_n=20):
        if not self.results:
            print("No results to display.")
            return pd.DataFrame()

        simple_results = []
        for k, v in self.results.items():
            entry = {key: val for key, val in v.items() if key not in ['Labels', 'X_processed', 'pca_obj']}
            entry['Params_Text'] = str(v['Params'])
            simple_results.append(entry)
            
        df_res = pd.DataFrame(simple_results)
        ascending = True if sort_by == 'davies_bouldin' else False
        
        if sort_by in df_res.columns:
            df_res = df_res.sort_values(by=sort_by, ascending=ascending)
            
        print(f"\n=== [{self.name}] Top {top_n} Results (Sorted by {sort_by}) ===")
        cols = ['Model', 'Option_Full', 'n_clusters', 'silhouette', 'davies_bouldin', 'Params_Text']
        print(df_res[cols].head(top_n).to_string(index=False))
        return df_res

    # --- PLOT 1: 2D Clusters (Statisch) ---
    def plot_clusters(self, top_n=3, sort_by='silhouette', x_col=0, y_col=1):
        """
        Plot de top N resultaten in 2D.
        :param x_col: Index (int) of Naam (str) van de kolom voor X-as.
                      Bij PCA is 0='PC1', 1='PC2'.
        """
        df_res = self.print_results(sort_by=sort_by, top_n=top_n)
        if df_res.empty: return

        cols = min(top_n, 3)
        rows = (top_n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if top_n == 1: axes = [axes]
        axes = np.array(axes).flatten()

        # Sorteer de originele resultaten om te matchen met de tabel
        sorted_keys = df_res.index # De index van df_res verwijst naar de volgorde
        
        for i, idx in enumerate(df_res.index): # Loop door de gesorteerde dataframe index
            # We moeten de juiste key in self.results vinden.
            # Omdat df_res een extract is, is de koppeling via row index lastig als we niet oppassen.
            # Betere manier: We zoeken de resultaat-entry die matcht met de rij uit df_res.
            
            # Voor nu, een simpele lookup in de lijst van values op basis van score match
            # (Dit is niet 100% robuust bij identieke scores, maar werkt voor visualisatie)
            target_score = df_res.loc[idx, sort_by]
            target_model = df_res.loc[idx, 'Model']
            
            # Zoek in results
            res = None
            for r in self.results.values():
                if r['Model'] == target_model and np.isclose(r[sort_by], target_score):
                    res = r
                    break
            
            if not res: continue

            X = res['X_processed']
            labels = res['Labels']
            
            # Bepaal assen
            if isinstance(x_col, int): x_data = X.iloc[:, x_col]
            else: x_data = X[x_col]
            
            if isinstance(y_col, int): y_data = X.iloc[:, y_col]
            else: y_data = X[y_col]

            # Plot
            ax = axes[i]
            sns.scatterplot(x=x_data, y=y_data, hue=labels, palette='tab10', ax=ax, s=50, legend='full')
            
            ax.set_title(f"{res['Model']} ({res['Option_ID']})\nClusters: {res['n_clusters']} | {sort_by}: {target_score:.3f}")
            ax.set_xlabel(x_data.name)
            ax.set_ylabel(y_data.name)
            ax.grid(True, alpha=0.3)
            
            if res['n_clusters'] > 10: ax.get_legend().remove()

        for j in range(i + 1, len(axes)): axes[j].axis('off')
        plt.tight_layout()
        plt.show()

    # --- PLOT 2: 3D Interactief (Plotly) ---
    def plot_3d_clusters(self, model_filter=None, option_filter=None, param_filter=None, 
                         x_col=0, y_col=1, z_col=2, metric_sort='silhouette'):
        """
        Maakt een interactieve 3D plot met Plotly.
        Selecteert automatisch het beste resultaat op basis van de filters.
        """
        # 1. Zoek het beste resultaat met de filters
        best_score = -1 if metric_sort != 'davies_bouldin' else float('inf')
        best_res = None
        
        for key, res in self.results.items():
            # Check Filters
            if model_filter and res['Model'] != model_filter: continue
            if option_filter and res['Option_ID'] != option_filter: continue
            if param_filter:
                match = True
                for p_k, p_v in param_filter.items():
                    if str(res['Params'].get(p_k)) != str(p_v): match = False
                if not match: continue
            
            # Check Score (Maximaliseren of Minimaliseren)
            score = res.get(metric_sort, -1)
            if pd.isna(score): continue
            
            if metric_sort == 'davies_bouldin':
                if score < best_score:
                    best_score = score
                    best_res = res
            else:
                if score > best_score:
                    best_score = score
                    best_res = res
        
        if not best_res:
            print("⚠️ No results found matching your filters.")
            return

        # 2. Data voorbereiden
        X = best_res['X_processed']
        labels = best_res['Labels']
        
        # Kolommen ophalen (op index of naam)
        def get_col_name(c):
            return X.columns[c] if isinstance(c, int) else c
            
        x_name, y_name, z_name = get_col_name(x_col), get_col_name(y_col), get_col_name(z_col)
        
        # Maak tijdelijke df voor plotly
        df_plot = X.copy()
        df_plot['Cluster'] = labels.astype(str)
        
        print(f"✨ Plotting 3D for: {best_res['Model']} | Option: {best_res['Option_Full']}")
        print(f"   Score ({metric_sort}): {best_score:.4f}")

        fig = px.scatter_3d(
            df_plot, x=x_name, y=y_name, z=z_name,
            color='Cluster',
            title=f"3D Clusters: {best_res['Model']} ({best_res['Option_ID']})",
            opacity=0.7
        )
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        fig.show()

    # --- PLOT 3: PCA Analyse ---
    def plot_pca_analysis(self, option_filter=None):
        """
        Toont de Explained Variance en de Components Heatmap.
        :param option_filter: (Optioneel) Welk Option ID (bv 'Opt1') te gebruiken.
                              Pakt anders de eerste optie waar PCA is gebruikt.
        """
        # Zoek een resultaat waar PCA is gebruikt
        target_res = None
        for res in self.results.values():
            if res['pca_obj'] is not None:
                if option_filter and res['Option_ID'] != option_filter:
                    continue
                target_res = res
                break
        
        if not target_res:
            print("⚠️ No PCA results found. Did you run an option with pca=True?")
            return

        pca = target_res['pca_obj']
        
        # --- Plot 1: Explained Variance ---
        plt.figure(figsize=(10, 4))
        
        # Bar chart
        n_comps = len(pca.explained_variance_ratio_)
        x_range = range(1, n_comps + 1)
        plt.bar(x_range, pca.explained_variance_ratio_, alpha=0.7, label='Individual Var')
        
        # Cumulative line
        plt.step(x_range, np.cumsum(pca.explained_variance_ratio_), where='mid', color='red', label='Cumulative Var')
        
        plt.title(f'PCA Analysis: Explained Variance ({target_res["Option_ID"]})')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.xticks(x_range)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()

        # --- Plot 2: Heatmap of Components (Loadings) ---
        # We gebruiken de originele feature namen als die beschikbaar zijn
        cols = self.original_feature_names if len(self.original_feature_names) == pca.components_.shape[1] else None
        
        comps_df = pd.DataFrame(
            pca.components_, 
            columns=cols, 
            index=[f"PC{i+1}" for i in range(n_comps)]
        )
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(comps_df, cmap='RdBu', center=0, annot=False)
        plt.title('PCA Components Heatmap (Feature Loadings)')
        plt.ylabel('Principal Component')
        plt.xlabel('Original Feature')
        plt.show()