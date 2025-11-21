import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning

class ModelEvaluator:
    def __init__(self, random_state=42):
        """
        Initialize the evaluator with a fixed random state.
        """
        self.random_state = random_state
        self.models = {}
        self.options = []
        self.results = {}
        # Default metrics. User can override this in run_experiments
        self.metrics = ['accuracy', 'f1_macro', 'recall_macro']
        
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    def add_model(self, name, model, params):
        """ Add a model to test. """
        if hasattr(model, 'random_state'):
            model.random_state = self.random_state
        self.models[name] = {'model': model, 'params': params}

    def add_option(self, scaling=False, dummies=False, cols_to_drop=None):
        """ Add a preprocessing option. """
        if cols_to_drop is None: cols_to_drop = []
        self.options.append({'scaling': scaling, 'dummys': dummies, 'cols_to_drop': cols_to_drop})

    def run_experiments(self, df, target_col, test_size=0.2, cv=5, n_jobs=-1, 
                        scoring=None, refit='accuracy'):
        """
        Runs the main loop.
        :param scoring: List of metrics to calculate (e.g., ['accuracy', 'f1_macro']).
        :param refit: The metric used to determine the 'best_estimator_' (must be in scoring).
        """
        self.results = {}
        
        # Update metrics if provided, otherwise use defaults
        if scoring:
            self.metrics = scoring
        
        # Ensure refit metric is in the scoring list
        if refit not in self.metrics:
            raise ValueError(f"Refit metric '{refit}' must be in scoring list {self.metrics}")

        print(f"🚀 Starting with {len(self.options)} options and {len(self.models)} models...")
        print(f"📊 Calculating metrics: {self.metrics} (Optimizing for: {refit})")

        for i, option in enumerate(self.options):
            opt_name = f"Opt{i}_S{'1' if option['scaling'] else '0'}_D{'1' if option['dummys'] else '0'}"
            print(f"\n--- ⚙️ Processing {opt_name} ---")

            # 1. Prepare Data
            X = df.drop(columns=[target_col] + option['cols_to_drop'])
            y = df[target_col]

            # 2. Split Data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )

            # 3. Build Preprocessor
            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            num_cols = X.select_dtypes(include=['number']).columns.tolist()

            transformers = []
            if option['scaling']:
                transformers.append(('num', StandardScaler(), num_cols))
            else:
                transformers.append(('num', 'passthrough', num_cols))
            
            if option['dummys']:
                transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), cat_cols))
            else:
                transformers.append(('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols))

            preprocessor = ColumnTransformer(transformers=transformers)

            # 4. Train Models
            for model_name, config in self.models.items():
                clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', config['model'])])
                pipe_params = {f'classifier__{k}': v for k, v in config['params'].items()}

                # GridSearch with Multi-Metric support
                grid = GridSearchCV(
                    clf, pipe_params, cv=cv, 
                    scoring=self.metrics, # List of metrics
                    refit=refit,          # Which one determines 'best_'
                    n_jobs=n_jobs
                )
                grid.fit(X_train, y_train)

                # Save Results
                # We save the full cv_results_ because it contains scores for ALL metrics
                self.results[f"{opt_name}_{model_name}"] = {
                    'best_score': grid.best_score_, # Score of the 'refit' metric
                    'best_params': grid.best_params_,
                    'best_estimator': grid.best_estimator_,
                    'cv_results': grid.cv_results_
                }
                
                print(f"   ✅ {model_name}: {refit}={grid.best_score_:.4f}")

        print("\n🏁 Experiments completed!")

    def _create_results_df(self, target_model=None):
        """Helper to create a DataFrame containing ALL metrics."""
        rows = []
        for key, val in self.results.items():
            parts = key.split('_')
            model_name = parts[-1]
            if target_model and model_name != target_model: continue

            opt_id = parts[0]
            scaling = 'S1' in parts
            dummies = 'D1' in parts
            
            cv_results = val['cv_results']
            params = cv_results['params']
            n_runs = len(params)
            
            for i in range(n_runs):
                row = {
                    'Model': model_name,
                    'Scaling': scaling,
                    'Dummies': dummies,
                    'Option_ID': opt_id,
                    'Config': f"{opt_id} (S:{'✅' if scaling else '❌'} D:{'✅' if dummies else '❌'})"
                }
                
                # Extract scores for EACH metric in self.metrics
                for metric in self.metrics:
                    # sklearn keys are like 'mean_test_accuracy', 'mean_test_f1_macro'
                    metric_key = f"mean_test_{metric}"
                    if metric_key in cv_results:
                        row[metric] = cv_results[metric_key][i]
                
                # Add hyperparameters
                for pk, pv in params[i].items():
                    clean_key = pk.replace('classifier__', '')
                    row[clean_key] = pv
                
                rows.append(row)
        return pd.DataFrame(rows)

    # --- VISUALIZATIONS ---

    def plot_heatmap(self, metric='accuracy'):
        """
        Plot heatmap.
        :param metric: Which metric to visualize (e.g. 'accuracy', 'f1_macro').
        """
        if metric not in self.metrics:
            print(f"⚠️ Metric '{metric}' not found. Available: {self.metrics}")
            return

        data = {}
        annot_indices = {}
        
        for key, val in self.results.items():
            opt_full, mod = key.rsplit('_', 1)
            # Retrieve the list of scores for the requested metric
            scores = val['cv_results'][f"mean_test_{metric}"]
            
            # We always plot the MAX score found during tuning for that metric
            score = np.max(scores)
            idx = np.argmax(scores)
                
            if opt_full not in data: data[opt_full] = {}
            data[opt_full][mod] = score
            annot_indices[(opt_full, mod)] = idx

        df_heat = pd.DataFrame(data).T
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_heat, annot=False, cmap='viridis', fmt='.1%', ax=ax, cbar_kws={'label': metric})
        
        global_max = df_heat.max().max()
        for i, row_val in enumerate(df_heat.index):
            for j, col_val in enumerate(df_heat.columns):
                score = df_heat.loc[row_val, col_val]
                idx = annot_indices.get((row_val, col_val), '')
                ax.text(j+0.5, i+0.4, f"{score:.1%}", ha='center', va='center', color='white', weight='bold')
                if idx != '-':
                    ax.text(j+0.5, i+0.7, f"cfg:{idx}", ha='center', va='center', color='lightgrey', fontsize=8)
                if np.isclose(score, global_max):
                    rect = patches.Rectangle((j, i), 1, 1, linewidth=4, edgecolor='#39FF14', facecolor='none')
                    ax.add_patch(rect)
        
        plt.title(f"Model Performance ({metric})", size=14)
        plt.tight_layout()
        plt.show()

    def plot_scatter(self, target_model=None, metric='accuracy'):
        """Scatter plot of results for a specific metric."""
        df = self._create_results_df(target_model)
        if metric not in df.columns:
            print(f"⚠️ Metric '{metric}' not available.")
            return

        df = df.sort_values('Option_ID')
        
        plt.figure(figsize=(12, 6))
        sns.stripplot(data=df, x='Config', y=metric, hue='Model', 
                      jitter=0.2, dodge=True, size=6, alpha=0.7, palette='deep')
        
        plt.title(f"Performance Distribution: {metric} {'(Focus: ' + target_model + ')' if target_model else ''}", fontsize=14)
        plt.xlabel("Preprocessing Option", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def plot_parameter_impact(self, target_model=None, metric='accuracy', best_only=False):
        """
        Line plots for hyperparameters based on a specific metric.
        """
        # Auto-select best model based on the requested metric
        df_all = self._create_results_df(target_model)
        if metric not in df_all.columns:
            print(f"⚠️ Metric '{metric}' not found.")
            return

        if target_model is None:
            best_idx = df_all[metric].idxmax()
            target_model = df_all.loc[best_idx, 'Model']
            print(f"ℹ️ Auto-selected model: '{target_model}' (Highest {metric})")
            # Filter df for this model
            df = df_all[df_all['Model'] == target_model]
        else:
            df = df_all

        if best_only:
            best_idx = df[metric].idxmax()
            best_opt_id = df.loc[best_idx, 'Option_ID']
            df = df[df['Option_ID'] == best_opt_id]

        std_cols = ['Model', 'Scaling', 'Dummies', 'Option_ID', 'Config'] + self.metrics
        params = [c for c in df.columns if c not in std_cols]
        
        num_params = len(params)
        fig, axes = plt.subplots(nrows=1, ncols=num_params, figsize=(6 * num_params, 5), sharey=True)
        if num_params == 1: axes = [axes]

        for i, param in enumerate(params):
            ax = axes[i]
            df[param] = df[param].fillna("None").astype(str)
            try:
                df['sort_col'] = pd.to_numeric(df[param].replace('None', -1))
                df = df.sort_values('sort_col')
            except:
                df = df.sort_values(param)

            sns.lineplot(data=df, x=param, y=metric, hue='Config', style='Config',
                         markers=True, estimator='mean', ci=100, ax=ax, palette='viridis')
            
            ax.set_title(f"Impact of {param}")
            ax.set_xlabel(param)
            ax.set_ylabel(metric if i == 0 else "")
            ax.grid(True, linestyle='--', alpha=0.5)
            
            if i != num_params - 1: 
                if ax.get_legend(): ax.get_legend().remove()
            else:
                ax.legend(title="Preprocessing", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.suptitle(f"Hyperparameter Analysis: {target_model} ({metric})", fontsize=16)
        plt.tight_layout()
        plt.show()