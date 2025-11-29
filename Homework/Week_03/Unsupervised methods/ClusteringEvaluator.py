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



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# import plotly.express as px # Voor de 3D plots

# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
# from sklearn.model_selection import ParameterGrid
# from sklearn.base import clone

# # Veelgebruikte Clustering Algoritmes
# from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
# from sklearn.mixture import GaussianMixture

# class ClusteringEvaluator:
#     def __init__(self, name="Clustering Evaluator", random_state=42):
#         self.name = name
#         self.random_state = random_state
#         self.models = {}
#         self.options = []
#         self.results = {}
#         self.metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
#         self.original_feature_names = [] # Opslaan voor PCA heatmap
        
#         warnings.filterwarnings("ignore")

#     def add_model(self, name, model, params=None):
#         """ Add a model and parameter grid. """
#         if params is None: params = {}
#         if hasattr(model, 'random_state'):
#             model.random_state = self.random_state
#         self.models[name] = {'model': model, 'params': params}

#     def add_option(self, scaling=True, method='standard', pca=False, n_components=2):
#         """ 
#         Add preprocessing option.
#         :param pca: True/False om PCA toe te passen.
#         :param n_components: Aantal PC's (int) of variantie (float, bv 0.95) om te behouden.
#         """
#         self.options.append({
#             'scaling': scaling, 
#             'method': method, 
#             'pca': pca, 
#             'n_components': n_components
#         })

#     def run_experiments(self, df):
#         self.results = {}
#         print(f"\n{'='*60}")
#         print(f"🚀 START CLUSTERING EXPERIMENT: {self.name.upper()}")
#         print(f"{'='*60}")
#         print(f"📊 Data Shape: {df.shape}")
        
#         # Bewaar originele kolomnamen voor visualisatie later
#         self.original_feature_names = df.columns.tolist()

#         for i, option in enumerate(self.options):
#             opt_name = f"Opt{i}_{'Scaled' if option['scaling'] else 'Raw'}"
#             if option['pca']:
#                 opt_name += f"_PCA({option['n_components']})"
                
#             print(f"\n--- ⚙️ Processing {opt_name} ---")

#             # 1. Preprocessing (Scaling)
#             X = df.copy()
#             if option['scaling']:
#                 if option['method'] == 'minmax':
#                     scaler = MinMaxScaler()
#                 else:
#                     scaler = StandardScaler()
#                 # We behouden dataframe structuur voor de features
#                 X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
#                 X = X_scaled
            
#             # 2. PCA (Optioneel)
#             pca_obj = None
#             if option['pca']:
#                 pca_obj = PCA(n_components=option['n_components'], random_state=self.random_state)
#                 X_pca = pca_obj.fit_transform(X)
#                 # Maak nieuwe DataFrame met PC namen
#                 n_pcs = X_pca.shape[1]
#                 cols = [f"PC{j+1}" for j in range(n_pcs)]
#                 X = pd.DataFrame(X_pca, columns=cols)
#                 print(f"   -> PCA toegepast: {len(self.original_feature_names)} features gereduceerd naar {n_pcs} componenten.")

#             # 3. Modellen Itereren
#             for model_name, config in self.models.items():
#                 param_grid = list(ParameterGrid(config['params']))
#                 if not param_grid: param_grid = [{}]

#                 print(f"   Testing {model_name} ({len(param_grid)} configurations)...")

#                 for params in param_grid:
#                     model = clone(config['model'])
#                     model.set_params(**params)
                    
#                     try:
#                         labels = model.fit_predict(X)
                        
#                         # Tel unieke clusters (exclusief noise -1)
#                         unique_labels = set(labels)
#                         n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
                        
#                         scores = {m: np.nan for m in self.metrics}
#                         if n_clusters >= 2:
#                             scores = {
#                                 'silhouette': silhouette_score(X, labels),
#                                 'davies_bouldin': davies_bouldin_score(X, labels),
#                                 'calinski_harabasz': calinski_harabasz_score(X, labels)
#                             }
                        
#                         # Unieke key maken
#                         param_str = "_".join([f"{k}{v}" for k,v in params.items()])
#                         key = f"{opt_name}_{model_name}_{param_str}"
                        
#                         self.results[key] = {
#                             'Model': model_name,
#                             'Option_ID': f"Opt{i}", # Korte ID voor filteren
#                             'Option_Full': opt_name,
#                             'Params': params,
#                             'Labels': labels,
#                             'n_clusters': n_clusters,
#                             'n_noise': list(labels).count(-1),
#                             'X_processed': X, # De data waarop getraind is (kan PCA zijn)
#                             'pca_obj': pca_obj, # Het getrainde PCA object (voor plots)
#                             **scores
#                         }
                        
#                     except Exception as e:
#                         print(f"      ❌ Error with {params}: {e}")

#         print(f"\n🏁 Experiments completed! Total runs: {len(self.results)}")

#     def print_results(self, sort_by='silhouette', top_n=20):
#         if not self.results:
#             print("No results to display.")
#             return pd.DataFrame()

#         simple_results = []
#         for k, v in self.results.items():
#             entry = {key: val for key, val in v.items() if key not in ['Labels', 'X_processed', 'pca_obj']}
#             entry['Params_Text'] = str(v['Params'])
#             simple_results.append(entry)
            
#         df_res = pd.DataFrame(simple_results)
#         ascending = True if sort_by == 'davies_bouldin' else False
        
#         if sort_by in df_res.columns:
#             df_res = df_res.sort_values(by=sort_by, ascending=ascending)
            
#         print(f"\n=== [{self.name}] Top {top_n} Results (Sorted by {sort_by}) ===")
#         cols = ['Model', 'Option_Full', 'n_clusters', 'silhouette', 'davies_bouldin', 'Params_Text']
#         print(df_res[cols].head(top_n).to_string(index=False))
#         return df_res

#     # --- PLOT 1: 2D Clusters (Statisch) ---
#     def plot_clusters(self, top_n=3, sort_by='silhouette', x_col=0, y_col=1):
#         """
#         Plot de top N resultaten in 2D.
#         :param x_col: Index (int) of Naam (str) van de kolom voor X-as.
#                       Bij PCA is 0='PC1', 1='PC2'.
#         """
#         df_res = self.print_results(sort_by=sort_by, top_n=top_n)
#         if df_res.empty: return

#         cols = min(top_n, 3)
#         rows = (top_n + cols - 1) // cols
#         fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
#         if top_n == 1: axes = [axes]
#         axes = np.array(axes).flatten()

#         # Sorteer de originele resultaten om te matchen met de tabel
#         sorted_keys = df_res.index # De index van df_res verwijst naar de volgorde
        
#         for i, idx in enumerate(df_res.index): # Loop door de gesorteerde dataframe index
#             # We moeten de juiste key in self.results vinden.
#             # Omdat df_res een extract is, is de koppeling via row index lastig als we niet oppassen.
#             # Betere manier: We zoeken de resultaat-entry die matcht met de rij uit df_res.
            
#             # Voor nu, een simpele lookup in de lijst van values op basis van score match
#             # (Dit is niet 100% robuust bij identieke scores, maar werkt voor visualisatie)
#             target_score = df_res.loc[idx, sort_by]
#             target_model = df_res.loc[idx, 'Model']
            
#             # Zoek in results
#             res = None
#             for r in self.results.values():
#                 if r['Model'] == target_model and np.isclose(r[sort_by], target_score):
#                     res = r
#                     break
            
#             if not res: continue

#             X = res['X_processed']
#             labels = res['Labels']
            
#             # Bepaal assen
#             if isinstance(x_col, int): x_data = X.iloc[:, x_col]
#             else: x_data = X[x_col]
            
#             if isinstance(y_col, int): y_data = X.iloc[:, y_col]
#             else: y_data = X[y_col]

#             # Plot
#             ax = axes[i]
#             sns.scatterplot(x=x_data, y=y_data, hue=labels, palette='tab10', ax=ax, s=50, legend='full')
            
#             ax.set_title(f"{res['Model']} ({res['Option_ID']})\nClusters: {res['n_clusters']} | {sort_by}: {target_score:.3f}")
#             ax.set_xlabel(x_data.name)
#             ax.set_ylabel(y_data.name)
#             ax.grid(True, alpha=0.3)
            
#             if res['n_clusters'] > 10: ax.get_legend().remove()

#         for j in range(i + 1, len(axes)): axes[j].axis('off')
#         plt.tight_layout()
#         plt.show()

#     # --- PLOT 2: 3D Interactief (Plotly) ---
#     def plot_3d_clusters(self, model_filter=None, option_filter=None, param_filter=None, 
#                          x_col=0, y_col=1, z_col=2, metric_sort='silhouette'):
#         """
#         Maakt een interactieve 3D plot met Plotly.
#         Selecteert automatisch het beste resultaat op basis van de filters.
#         """
#         # 1. Zoek het beste resultaat met de filters
#         best_score = -1 if metric_sort != 'davies_bouldin' else float('inf')
#         best_res = None
        
#         for key, res in self.results.items():
#             # Check Filters
#             if model_filter and res['Model'] != model_filter: continue
#             if option_filter and res['Option_ID'] != option_filter: continue
#             if param_filter:
#                 match = True
#                 for p_k, p_v in param_filter.items():
#                     if str(res['Params'].get(p_k)) != str(p_v): match = False
#                 if not match: continue
            
#             # Check Score (Maximaliseren of Minimaliseren)
#             score = res.get(metric_sort, -1)
#             if pd.isna(score): continue
            
#             if metric_sort == 'davies_bouldin':
#                 if score < best_score:
#                     best_score = score
#                     best_res = res
#             else:
#                 if score > best_score:
#                     best_score = score
#                     best_res = res
        
#         if not best_res:
#             print("⚠️ No results found matching your filters.")
#             return

#         # 2. Data voorbereiden
#         X = best_res['X_processed']
#         labels = best_res['Labels']
        
#         # Kolommen ophalen (op index of naam)
#         def get_col_name(c):
#             return X.columns[c] if isinstance(c, int) else c
            
#         x_name, y_name, z_name = get_col_name(x_col), get_col_name(y_col), get_col_name(z_col)
        
#         # Maak tijdelijke df voor plotly
#         df_plot = X.copy()
#         df_plot['Cluster'] = labels.astype(str)
        
#         print(f"✨ Plotting 3D for: {best_res['Model']} | Option: {best_res['Option_Full']}")
#         print(f"   Score ({metric_sort}): {best_score:.4f}")

#         fig = px.scatter_3d(
#             df_plot, x=x_name, y=y_name, z=z_name,
#             color='Cluster',
#             title=f"3D Clusters: {best_res['Model']} ({best_res['Option_ID']})",
#             opacity=0.7
#         )
#         fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
#         fig.show()

#     # --- PLOT 3: PCA Analyse ---
#     def plot_pca_analysis(self, option_filter=None):
#         """
#         Toont de Explained Variance en de Components Heatmap.
#         :param option_filter: (Optioneel) Welk Option ID (bv 'Opt1') te gebruiken.
#                               Pakt anders de eerste optie waar PCA is gebruikt.
#         """
#         # Zoek een resultaat waar PCA is gebruikt
#         target_res = None
#         for res in self.results.values():
#             if res['pca_obj'] is not None:
#                 if option_filter and res['Option_ID'] != option_filter:
#                     continue
#                 target_res = res
#                 break
        
#         if not target_res:
#             print("⚠️ No PCA results found. Did you run an option with pca=True?")
#             return

#         pca = target_res['pca_obj']
        
#         # --- Plot 1: Explained Variance ---
#         plt.figure(figsize=(10, 4))
        
#         # Bar chart
#         n_comps = len(pca.explained_variance_ratio_)
#         x_range = range(1, n_comps + 1)
#         plt.bar(x_range, pca.explained_variance_ratio_, alpha=0.7, label='Individual Var')
        
#         # Cumulative line
#         plt.step(x_range, np.cumsum(pca.explained_variance_ratio_), where='mid', color='red', label='Cumulative Var')
        
#         plt.title(f'PCA Analysis: Explained Variance ({target_res["Option_ID"]})')
#         plt.xlabel('Principal Component')
#         plt.ylabel('Explained Variance Ratio')
#         plt.xticks(x_range)
#         plt.legend()
#         plt.grid(axis='y', linestyle='--', alpha=0.5)
#         plt.show()

#         # --- Plot 2: Heatmap of Components (Loadings) ---
#         # We gebruiken de originele feature namen als die beschikbaar zijn
#         cols = self.original_feature_names if len(self.original_feature_names) == pca.components_.shape[1] else None
        
#         comps_df = pd.DataFrame(
#             pca.components_, 
#             columns=cols, 
#             index=[f"PC{i+1}" for i in range(n_comps)]
#         )
        
#         plt.figure(figsize=(12, 6))
#         sns.heatmap(comps_df, cmap='RdBu', center=0, annot=False)
#         plt.title('PCA Components Heatmap (Feature Loadings)')
#         plt.ylabel('Principal Component')
#         plt.xlabel('Original Feature')
#         plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly.express as px 

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone

# Common Clustering Algorithms
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture

class ClusteringEvaluator:
    def __init__(self, name="Clustering Evaluator", random_state=42):
        """
        Initialize the Clustering Evaluator.
        """
        self.name = name
        self.random_state = random_state
        self.models = {}
        self.options = []
        self.results = {}
        # Standard metrics
        self.metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'adjusted_rand']
        self.custom_metrics = {} 
        self.original_feature_names = [] 
        
        warnings.filterwarnings("ignore")

    def add_model(self, name, model, params=None):
        """ Add a clustering model and parameter grid. """
        if params is None: params = {}
        if hasattr(model, 'random_state'):
            model.random_state = self.random_state
        self.models[name] = {'model': model, 'params': params}

    def add_option(self, scaling=True, method='standard', pca=False, n_components=2, cols_to_drop=None):
        """ 
        Add preprocessing option.
        :param pca: True/False to apply PCA.
        :param n_components: Number of Principal Components to keep.
        :param cols_to_drop: List of column names to remove BEFORE scaling/clustering.
        """
        if cols_to_drop is None: cols_to_drop = []
        self.options.append({
            'scaling': scaling, 
            'method': method, 
            'pca': pca, 
            'n_components': n_components,
            'cols_to_drop': cols_to_drop
        })

    def add_custom_metric(self, name, func):
        """ Add a custom metric function(X, labels) -> float. """
        self.custom_metrics[name] = func
        print(f"✅ Custom metric '{name}' added.")

    def run_experiments(self, df, target_col=None):
        """
        Runs clustering experiments.
        :param target_col: (Optional) Name of the column containing Ground Truth labels (e.g. 'Breed').
                           If provided, Adjusted Rand Score is calculated.
        """
        self.results = {}
        print(f"\n{'='*60}")
        print(f"🚀 START CLUSTERING EXPERIMENT: {self.name.upper()}")
        print(f"{'='*60}")
        
        # 1. Prepare Base Data (Separate target if provided)
        if target_col and target_col in df.columns:
            df_base = df.drop(columns=[target_col])
            y_true = df[target_col]
            print(f"📊 Base Data Shape: {df_base.shape} (Target '{target_col}' separated for evaluation)")
        else:
            df_base = df.copy()
            y_true = None
            print(f"📊 Base Data Shape: {df_base.shape} (No Ground Truth provided)")

        for i, option in enumerate(self.options):
            # Build readable option name
            opt_name = f"Opt{i}"
            if option['cols_to_drop']:
                opt_name += f"_Drop({len(option['cols_to_drop'])})"
            opt_name += f"_{'Scaled' if option['scaling'] else 'Raw'}"
            if option['pca']:
                opt_name += f"_PCA({option['n_components']})"
                
            print(f"\n--- ⚙️ Processing {opt_name} ---")

            # 2. Drop Columns (Feature Selection)
            X = df_base.drop(columns=option['cols_to_drop'], errors='ignore')
            
            # Store feature names for this specific option (needed for PCA heatmap)
            current_feature_names = X.columns.tolist()

            # 3. Preprocessing (Scaling)
            if option['scaling']:
                if option['method'] == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                X = X_scaled
            
            # 4. PCA (Optional)
            pca_obj = None
            if option['pca']:
                pca_obj = PCA(n_components=option['n_components'], random_state=self.random_state)
                X_pca = pca_obj.fit_transform(X)
                n_pcs = X_pca.shape[1]
                cols = [f"PC{j+1}" for j in range(n_pcs)]
                X = pd.DataFrame(X_pca, columns=cols)
                print(f"   -> PCA Applied: {len(current_feature_names)} features -> {n_pcs} components.")

            # 5. Iterate Models
            for model_name, config in self.models.items():
                param_grid = list(ParameterGrid(config['params']))
                if not param_grid: param_grid = [{}]

                print(f"   Testing {model_name} ({len(param_grid)} configurations)...")

                for params in param_grid:
                    model = clone(config['model'])
                    model.set_params(**params)
                    
                    try:
                        # Fit & Predict
                        if 'GaussianMixture' in str(type(model)):
                            model.fit(X)
                            labels = model.predict(X)
                        else:
                            labels = model.fit_predict(X)
                        
                        unique_labels = set(labels)
                        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
                        
                        scores = {m: np.nan for m in self.metrics}
                        
                        # Unsupervised Metrics (Need >= 2 clusters)
                        if n_clusters >= 2:
                            scores['silhouette'] = silhouette_score(X, labels)
                            scores['davies_bouldin'] = davies_bouldin_score(X, labels)
                            scores['calinski_harabasz'] = calinski_harabasz_score(X, labels)
                        
                        # Supervised Metric (Adjusted Rand)
                        if y_true is not None:
                            scores['adjusted_rand'] = adjusted_rand_score(y_true, labels)

                        # Custom Metrics
                        for metric_name, metric_func in self.custom_metrics.items():
                            try:
                                scores[metric_name] = metric_func(X, labels)
                            except:
                                scores[metric_name] = np.nan
                        
                        param_str = "_".join([f"{k}{v}" for k,v in params.items()])
                        key = f"{opt_name}_{model_name}_{param_str}"
                        
                        self.results[key] = {
                            'Model': model_name,
                            'Option_ID': f"Opt{i}",
                            'Option_Full': opt_name,
                            'Params': params,
                            'Labels': labels,
                            'n_clusters': n_clusters,
                            'n_noise': list(labels).count(-1),
                            'X_processed': X,
                            'pca_obj': pca_obj,
                            'feature_names': current_feature_names, # Store for PCA plots
                            **scores
                        }
                        
                    except Exception as e:
                        print(f"      ❌ Error with {params}: {e}")

        print(f"\n🏁 Experiments completed! Total runs: {len(self.results)}")

    def print_results(self, sort_by='silhouette', top_n=20):
        """ Prints results table. Default sort is Silhouette (or Adjusted Rand if available). """
        if not self.results:
            print("No results to display.")
            return pd.DataFrame()

        simple_results = []
        for k, v in self.results.items():
            entry = {key: val for key, val in v.items() 
                     if key not in ['Labels', 'X_processed', 'pca_obj', 'feature_names']}
            entry['Params_Text'] = str(v['Params'])
            simple_results.append(entry)
            
        df_res = pd.DataFrame(simple_results)
        
        # Sorting logic
        ascending = True if sort_by == 'davies_bouldin' else False
        
        if sort_by in df_res.columns:
            if df_res[sort_by].isna().all():
                print(f"⚠️ Metric '{sort_by}' is not available (all NaN). Sorting by Silhouette instead.")
                sort_by = 'silhouette'
                ascending = False
            df_res = df_res.sort_values(by=sort_by, ascending=ascending)
            
        print(f"\n=== [{self.name}] Top {top_n} Results (Sorted by {sort_by}) ===")
        
        base_cols = ['Model', 'Option_Full', 'n_clusters']
        all_metrics = self.metrics + list(self.custom_metrics.keys())
        valid_metrics = [m for m in all_metrics if m in df_res.columns and not df_res[m].isna().all()]
        
        final_cols = base_cols + valid_metrics + ['Params_Text']
        print(df_res[final_cols].head(top_n).to_string(index=False))
        return df_res

    # --- PLOT 1: 2D CLUSTERS ---
    def plot_clusters(self, top_n=3, sort_by='silhouette', x_col=0, y_col=1):
        df_res = self.print_results(sort_by=sort_by, top_n=top_n)
        if df_res.empty: return

        cols = min(top_n, 3)
        rows = (top_n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if top_n == 1: axes = [axes]
        axes = np.array(axes).flatten()

        for i, idx in enumerate(df_res.index):
            # Robustly find result based on key characteristics if index doesn't match directly
            target_model = df_res.loc[idx, 'Model']
            target_opt = df_res.loc[idx, 'Option_Full']
            target_params = df_res.loc[idx, 'Params_Text']
            
            res = None
            for r in self.results.values():
                if (r['Model'] == target_model and 
                    r['Option_Full'] == target_opt and 
                    str(r['Params']) == target_params):
                    res = r
                    break
            
            if not res: continue

            X = res['X_processed']
            labels = res['Labels']
            score = res.get(sort_by, np.nan)
            
            if isinstance(x_col, int) and x_col < X.shape[1]: x_data = X.iloc[:, x_col]
            elif isinstance(x_col, str) and x_col in X.columns: x_data = X[x_col]
            else: x_data = X.iloc[:, 0]

            if isinstance(y_col, int) and y_col < X.shape[1]: y_data = X.iloc[:, y_col]
            elif isinstance(y_col, str) and y_col in X.columns: y_data = X[y_col]
            else: y_data = X.iloc[:, 1]

            ax = axes[i]
            sns.scatterplot(x=x_data, y=y_data, hue=labels, palette='tab10', ax=ax, s=50, legend='full')
            
            ax.set_title(f"{res['Model']}\nClusters: {res['n_clusters']} | {sort_by}: {score:.3f}")
            ax.set_xlabel(x_data.name)
            ax.set_ylabel(y_data.name)
            ax.grid(True, alpha=0.3)
            if res['n_clusters'] > 10: ax.get_legend().remove()

        for j in range(i + 1, len(axes)): axes[j].axis('off')
        plt.tight_layout()
        plt.show()

    # --- PLOT 2: 3D CLUSTERS (PLOTLY) ---
    def plot_3d_clusters(self, model_filter=None, option_filter=None, x_col=0, y_col=1, z_col=2, metric_sort='silhouette'):
        best_score = -1 if metric_sort != 'davies_bouldin' else float('inf')
        best_res = None
        
        for key, res in self.results.items():
            if model_filter and res['Model'] != model_filter: continue
            if option_filter and res['Option_ID'] != option_filter: continue
            
            score = res.get(metric_sort, -1)
            if pd.isna(score): continue
            
            if metric_sort == 'davies_bouldin':
                if score < best_score: best_score, best_res = score, res
            else:
                if score > best_score: best_score, best_res = score, res
        
        if not best_res:
            print("⚠️ No results found matching your filters.")
            return

        X = best_res['X_processed']
        if X.shape[1] < 3:
            print(f"⚠️ Cannot plot 3D: Data has only {X.shape[1]} dimensions.")
            return

        def get_col_name(c):
            return X.columns[c] if isinstance(c, int) else c
            
        x_name, y_name, z_name = get_col_name(x_col), get_col_name(y_col), get_col_name(z_col)
        
        df_plot = X.copy()
        df_plot['Cluster'] = best_res['Labels'].astype(str)
        
        print(f"✨ Plotting 3D: {best_res['Model']} | Option: {best_res['Option_Full']}")
        print(f"   Score ({metric_sort}): {best_score:.4f}")

        fig = px.scatter_3d(
            df_plot, x=x_name, y=y_name, z=z_name,
            color='Cluster',
            title=f"3D Clusters: {best_res['Model']} ({best_res['Option_ID']})",
            opacity=0.7
        )
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        fig.show()

    # --- PLOT 3: PCA ANALYSIS ---
    def plot_pca_analysis(self, option_filter=None):
        target_res = None
        # Find the first result that used PCA matching the filter
        for res in self.results.values():
            if res['pca_obj'] is not None:
                if option_filter and res['Option_ID'] != option_filter: continue
                target_res = res
                break
        
        if not target_res:
            print("⚠️ No PCA results found.")
            return

        pca = target_res['pca_obj']
        feature_names = target_res['feature_names']
        
        # Variance Plot
        plt.figure(figsize=(10, 4))
        n_comps = len(pca.explained_variance_ratio_)
        x_range = range(1, n_comps + 1)
        plt.bar(x_range, pca.explained_variance_ratio_, alpha=0.7, label='Individual Var')
        plt.step(x_range, np.cumsum(pca.explained_variance_ratio_), where='mid', color='red', label='Cumulative Var')
        plt.title(f'PCA Analysis: Explained Variance ({target_res["Option_Full"]})')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Ratio')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()

        # Loadings Heatmap
        # Use stored feature names from that specific run (because cols_to_drop might have changed them)
        cols = feature_names if len(feature_names) == pca.components_.shape[1] else None
        comps_df = pd.DataFrame(pca.components_, columns=cols, index=[f"PC{i+1}" for i in range(n_comps)])
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(comps_df, cmap='RdBu', center=0, annot=False)
        plt.title('PCA Components Heatmap (Loadings)')
        plt.ylabel('Principal Component')
        plt.xlabel('Original Feature')
        plt.show()

    def plot_side_by_side_3d(self, result_key1, result_key2, x_col=0, y_col=1, z_col=2):
        """
        Plot twee 3D-resultaten naast elkaar met gesynchroniseerde camera's.
        Je moet de exacte 'keys' van de resultaten weten (of gebruik een helper om ze te vinden).
        """
        if result_key1 not in self.results or result_key2 not in self.results:
            print("⚠️ One or both keys not found in results.")
            return

        res1 = self.results[result_key1]
        res2 = self.results[result_key2]

        # Data ophalen
        def get_data(res):
            X = res['X_processed']
            labels = res['Labels']
            if X.shape[1] < 3: return None, None, None, None
            
            x = X.iloc[:, x_col]
            y = X.iloc[:, y_col]
            z = X.iloc[:, z_col]
            return x, y, z, labels

        x1, y1, z1, l1 = get_data(res1)
        x2, y2, z2, l2 = get_data(res2)
        
        if x1 is None or x2 is None:
            print("⚠️ Data has less than 3 dimensions.")
            return

        # Maak subplots (type='scene' is nodig voor 3D)
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=(f"{res1['Model']} ({res1['Option_ID']})", f"{res2['Model']} ({res2['Option_ID']})")
        )

        # Voeg Plot 1 toe
        fig.add_trace(
            go.Scatter3d(x=x1, y=y1, z=z1, mode='markers', marker=dict(size=4, color=l1, colorscale='Viridis'), name='Plot 1'),
            row=1, col=1
        )

        # Voeg Plot 2 toe
        fig.add_trace(
            go.Scatter3d(x=x2, y=y2, z=z2, mode='markers', marker=dict(size=4, color=l2, colorscale='Viridis'), name='Plot 2'),
            row=1, col=2
        )

        # Update layout om assen en camera te syncen
        fig.update_layout(
            title_text="Side-by-Side 3D Comparison",
            scene=dict(
                xaxis_title=x1.name, yaxis_title=y1.name, zaxis_title=z1.name,
                aspectmode='cube'
            ),
            scene2=dict(
                xaxis_title=x2.name, yaxis_title=y2.name, zaxis_title=z2.name,
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        fig.show()




    # --- PLOT: Comparison (Model vs Model OR Model vs Ground Truth) ---
    def plot_comparison(self, model_1=None, model_2=None, option_id=None, target_col=None, sort_by='silhouette', x_col=0, y_col=1):
        """
        Plots two scatter plots side-by-side.
        - Can compare Model 1 vs Model 2.
        - Can compare Model 1 vs Ground Truth (if model_2 is None and target_col is provided).
        - Auto-selects best option if option_id is None.
        """
        df_res = self.print_results(sort_by=sort_by, top_n=1000) # Get all results sorted
        if df_res.empty: return

        # 1. Determine Option ID
        if option_id is None:
            # Auto-select best option based on the top result
            best_idx = df_res.index[0]
            option_id = df_res.loc[best_idx, 'Option_ID'] # e.g., 'Opt0'
            print(f"ℹ️ Auto-selected Option: {option_id} (Best {sort_by})")
        else:
             # Ensure format matches (e.g. user gives 0, we need 'Opt0')
             if isinstance(option_id, int): option_id = f"Opt{option_id}"

        # Filter results for this option
        df_opt = df_res[df_res['Option_ID'] == option_id]
        
        if df_opt.empty:
            print(f"⚠️ No results found for option {option_id}.")
            return

        # 2. Helper to get data for a specific model name
        def get_model_data(model_name):
            # Find best run for this model in this option
            # Assuming df_opt is sorted, the first occurrence of model_name is the best for that model
            row = df_opt[df_opt['Model'] == model_name]
            if row.empty: return None
            
            # We need to find the full result object to get Labels and X
            # We use a unique combination to identify the run. 
            # (Params_Text is unique enough usually)
            target_params = row.iloc[0]['Params_Text']
            
            for r in self.results.values():
                if (r['Model'] == model_name and 
                    r['Option_ID'] == option_id and 
                    str(r['Params']) == target_params):
                    return r
            return None

        # 3. Prepare Data for Plot 1
        res1 = get_model_data(model_1) if model_1 else get_model_data(df_opt.iloc[0]['Model']) # Default to best model
        
        if not res1:
            print(f"⚠️ Model '{model_1}' not found in results for {option_id}.")
            return

        X = res1['X_processed']
        labels1 = res1['Labels']
        title1 = f"{res1['Model']}\n{sort_by}: {res1.get(sort_by, 0):.3f}"

        # 4. Prepare Data for Plot 2
        labels2 = None
        title2 = "Comparison"
        
        if model_2:
            # Compare with another model
            res2 = get_model_data(model_2)
            if res2:
                labels2 = res2['Labels']
                title2 = f"{res2['Model']}\n{sort_by}: {res2.get(sort_by, 0):.3f}"
            else:
                print(f"⚠️ Model '{model_2}' not found. Plotting Ground Truth instead (if avail).")
        
        # Fallback: If no model_2, try Ground Truth
        if labels2 is None:
             if target_col is not None:
                 # We need the target labels corresponding to X.
                 # Since X might be processed/subsetted, we assume df passed to run_experiments 
                 # was the source. If rows were dropped (e.g. NaN), lengths might mismatch.
                 # However, in clustering we usually keep X intact.
                 # Ideally, we stored y_true in results or can pass it here.
                 # For now, let's try to retrieve it from the dataframe if passed, assuming index alignment.
                 # NOTE: This part assumes 'df' variable is available or passed globally, which might be tricky inside class.
                 # BETTER: The user passes the labels series directly to target_col? 
                 # OR: We assume the user passes the dataframe in run_experiments and we stored y_true in self?
                 # Let's assume target_col is a Series/List here for simplicity in plotting? 
                 # No, standard is column name.
                 
                 # Simple fix: We need the actual labels array here. 
                 # If target_col is a string, we can't get data without the df.
                 # Let's change signature to accept y_true data OR accept we can't plot it without storage.
                 
                 # Let's assume the user passes the Series to `target_col` argument in this function call 
                 # for maximum flexibility if they want Ground Truth.
                 if isinstance(target_col, (pd.Series, np.ndarray, list)):
                     labels2 = target_col
                     title2 = "Ground Truth"
                 else:
                     title2 = "No Comparison Available"
             else:
                 title2 = "No Comparison Selected"


        # 5. Plotting
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Helper for axes selection
        def select_data(data, col):
            if isinstance(col, int) and col < data.shape[1]: return data.iloc[:, col]
            if isinstance(col, str) and col in data.columns: return data[col]
            return data.iloc[:, 0]

        x_data = select_data(X, x_col)
        y_data = select_data(X, y_col)

        # Plot 1
        sns.scatterplot(x=x_data, y=y_data, hue=labels1, palette='tab10', ax=axes[0], s=50, legend='full')
        axes[0].set_title(title1)
        axes[0].set_xlabel(x_data.name)
        axes[0].set_ylabel(y_data.name)
        axes[0].grid(True, alpha=0.3)

        # Plot 2
        if labels2 is not None:
            # Ensure length match (simple check)
            if len(labels2) != len(x_data):
                print(f"⚠️ Length mismatch: Data={len(x_data)}, Labels2={len(labels2)}. Truncating to min.")
                min_len = min(len(x_data), len(labels2))
                sns.scatterplot(x=x_data[:min_len], y=y_data[:min_len], hue=labels2[:min_len], palette='deep', ax=axes[1], s=50, legend='full')
            else:
                sns.scatterplot(x=x_data, y=y_data, hue=labels2, palette='deep', ax=axes[1], s=50, legend='full')
            
            axes[1].set_title(title2)
            axes[1].set_xlabel(x_data.name)
            axes[1].set_ylabel(y_data.name)
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "Target Data Not Provided\n(Pass Series to target_col)", 
                         ha='center', va='center', transform=axes[1].transAxes)

        plt.tight_layout()
        plt.show()