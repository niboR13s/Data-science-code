#class to compare classification models
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
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

class ModelEvaluator:
    def __init__(self, name="Model Evaluator", random_state=42):
        """
        Initialize the evaluator with a name and fixed random state.
        :param name: Name of this evaluator instance (e.g. 'TWF Analysis').
        """
        self.name = name
        self.random_state = random_state
        self.models = {}
        self.options = []
        self.results = {}
        # Store test sets here to reproduce plots later without needing original df
        self.test_sets = {} 
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
        """ Runs the main loop. """
        self.results = {}
        self.test_sets = {} 
        
        if scoring: self.metrics = scoring
        if refit not in self.metrics:
            print(f"⚠️ Refit metric '{refit}' not in scoring list. Using '{self.metrics[0]}' instead.")
            refit = self.metrics[0]

        # --- Header with Name ---
        print(f"\n{'='*60}")
        print(f"🚀 START EXPERIMENT: {self.name.upper()}")
        print(f"{'='*60}")
        print(f"📊 Target Column: '{target_col}'")
        print(f"📊 Metrics: {self.metrics} (Optimizing for: {refit})")

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
            
            # SAVE TEST SET
            self.test_sets[i] = {'X_test': X_test, 'y_test': y_test}

            # 3. Preprocessor
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

                grid = GridSearchCV(clf, pipe_params, cv=cv, scoring=self.metrics, refit=refit, n_jobs=n_jobs)
                grid.fit(X_train, y_train)

                self.results[f"{opt_name}_{model_name}"] = {
                    'best_score': grid.best_score_,
                    'best_params': grid.best_params_,
                    'best_estimator': grid.best_estimator_, # <--- FIXED: Added underscore
                    'cv_results': grid.cv_results_,
                    'option_idx': i 
                }
                print(f"   ✅ {model_name}: {refit}={grid.best_score_:.4f}")

        print(f"\n🏁 Experiments for {self.name} completed!")

    def _create_results_df(self, target_model=None):
        """Helper to create a DataFrame containing ALL metrics."""
        rows = []
        for key, val in self.results.items():
            parts = key.split('_')
            model_name = parts[-1]
            if target_model and model_name != target_model: continue

            opt_id = parts[0]
            option_idx = val['option_idx']
            opt_config = self.options[option_idx]
            cv_results = val['cv_results']
            params = cv_results['params']
            
            for i in range(len(params)):
                row = {
                    'Model': model_name,
                    'Scaling': opt_config['scaling'],
                    'Dummies': opt_config['dummys'],
                    'Option_ID': opt_id,
                    'Config': f"{opt_id} (S:{'✅' if opt_config['scaling'] else '❌'} D:{'✅' if opt_config['dummys'] else '❌'})"
                }
                for metric in self.metrics:
                    metric_key = f"mean_test_{metric}"
                    if metric_key in cv_results:
                        row[metric] = cv_results[metric_key][i]
                for pk, pv in params[i].items():
                    row[pk.replace('classifier__', '')] = pv
                rows.append(row)
        return pd.DataFrame(rows)

    def print_results(self, sort_by=None, model_filter=None, top_n=20, decimals=4):
        """ Print a filtered and sorted table of results. """
        df = self._create_results_df()
        if sort_by is None: sort_by = self.metrics[0]
        if model_filter: df = df[df['Model'] == model_filter]
        if sort_by in df.columns: df = df.sort_values(by=sort_by, ascending=False)
        
        print(f"\n=== [{self.name}] Top {top_n} Results (Sorted by {sort_by}) ===")
        
        for col in df.select_dtypes(include=['float']).columns:
            if col in self.metrics or col == 'Score':
                df[col] = df[col].round(decimals)

        std_cols = ['Model', 'Config', 'Option_ID']
        metric_cols = [m for m in self.metrics if m in df.columns]
        param_cols = [c for c in df.columns if c not in std_cols + metric_cols + ['Scaling', 'Dummies']]
        param_cols = [c for c in param_cols if df[c].notna().any()]
        
        print(df[std_cols + metric_cols + param_cols].head(top_n).to_string(index=False))

    # --- VOTING ENSEMBLE ---
    def evaluate_voting_ensemble(self, option_id=None, decimals=4):
        df_res = self._create_results_df()
        
        if option_id is None:
            best_metric = self.metrics[0]
            best_idx = df_res[best_metric].idxmax()
            option_id = df_res.loc[best_idx, 'Option_ID']
            print(f"ℹ️ Auto-selected option for Ensemble: {option_id} (based on best {best_metric})")

        df_opt = df_res[df_res['Option_ID'] == option_id]
        best_models_indices = df_opt.groupby('Model')[self.metrics[0]].idxmax()
        
        estimators = []
        model_names = []
        option_idx = None

        for idx in best_models_indices:
            row = df_opt.loc[idx]
            model_name = row['Model']
            
            found_key = None
            for k in self.results.keys():
                if k.endswith(f"_{model_name}") and k.startswith(f"{option_id}_"):
                    found_key = k
                    break
            
            if found_key:
                res = self.results[found_key]
                estimators.append(res['best_estimator']) # Stored without underscore in dict key
                model_names.append(model_name)
                option_idx = res['option_idx']

        if not estimators:
            print("⚠️ No models found to ensemble.")
            return

        test_data = self.test_sets[option_idx]
        X_test = test_data['X_test']
        y_test = test_data['y_test']

        print(f"\n🤝 Creating Soft Voting Ensemble from: {model_names}")
        
        probs = []
        valid_estimators = []
        
        for name, model in zip(model_names, estimators):
            if hasattr(model, "predict_proba"):
                try:
                    p = model.predict_proba(X_test)
                    probs.append(p)
                    valid_estimators.append(name)
                except:
                    print(f"⚠️ Skipping {name}: predict_proba failed.")
            else:
                print(f"⚠️ Skipping {name}: No predict_proba support.")

        if not probs:
            print("❌ Ensemble failed: No models support probabilities.")
            return

        avg_probs = np.mean(probs, axis=0)
        y_pred = np.argmax(avg_probs, axis=1) 

        acc = accuracy_score(y_test, y_pred)
        if len(np.unique(y_test)) > 2:
            f1 = f1_score(y_test, y_pred, average='macro')
        else:
            f1 = f1_score(y_test, y_pred, pos_label=1)
        
        print(f"\n=== [{self.name}] Ensemble Results (Option: {option_id}) ===")
        print(f"Models used: {valid_estimators}")
        print(f"Accuracy: {acc:.{decimals}f}")
        print(f"F1 Score: {f1:.{decimals}f}")
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Greens', colorbar=False)
        plt.title(f"Ensemble Confusion Matrix\n(Acc: {acc:.{decimals}f}, F1: {f1:.{decimals}f})")
        plt.show()

    # --- VISUALIZATIONS ---

    def plot_heatmap(self, metric=None, decimals=4):
        if metric is None: metric = self.metrics[0]
        data = {}
        
        for key, val in self.results.items():
            opt_full, mod = key.rsplit('_', 1)
            scores = val['cv_results'][f"mean_test_{metric}"]
            score = np.max(scores)
            if opt_full not in data: data[opt_full] = {}
            data[opt_full][mod] = score

        df_heat = pd.DataFrame(data).T
        fig, ax = plt.subplots(figsize=(10, 6))
        fmt_str = f".{decimals}f"
        sns.heatmap(df_heat, annot=False, cmap='viridis', fmt=fmt_str, ax=ax, cbar_kws={'label': metric})
        
        global_max = df_heat.max().max()
        for i, row_val in enumerate(df_heat.index):
            for j, col_val in enumerate(df_heat.columns):
                score = df_heat.loc[row_val, col_val]
                ax.text(j+0.5, i+0.4, f"{score:.{decimals}f}", ha='center', va='center', color='white', weight='bold')
                if np.isclose(score, global_max):
                    rect = patches.Rectangle((j, i), 1, 1, linewidth=4, edgecolor='#39FF14', facecolor='none')
                    ax.add_patch(rect)
        
        plt.title(f"[{self.name}] Model Performance ({metric})", size=14)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, model_filter=None, option_filter=None, metric_for_selection=None, decimals=4):
        if metric_for_selection is None: metric_for_selection = self.metrics[0]

        df_res = self._create_results_df()
        if model_filter: df_res = df_res[df_res['Model'] == model_filter]
        if option_filter: df_res = df_res[df_res['Option_ID'] == option_filter]

        best_indices = df_res.groupby('Model')[metric_for_selection].idxmax()
        if best_indices.empty: 
            print("⚠️ No results found matching filters.")
            return

        num_plots = len(best_indices)
        cols = min(num_plots, 3)
        rows = (num_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
        if num_plots == 1: axes = [axes]
        axes = np.array(axes).flatten()

        for i, idx in enumerate(best_indices):
            row = df_res.loc[idx]
            model_name = row['Model']
            opt_id = row['Option_ID']
            
            found_key = None
            for k in self.results.keys():
                if k.endswith(f"_{model_name}") and k.startswith(f"{opt_id}_"):
                    found_key = k
                    break
            
            res_data = self.results[found_key]
            best_model = res_data['best_estimator']
            opt_idx = res_data['option_idx']
            
            test_data = self.test_sets[opt_idx]
            y_test = test_data['y_test']
            y_pred = best_model.predict(test_data['X_test'])
            
            ax = axes[i]
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues', colorbar=False)
            ax.set_title(f"{model_name}\n({opt_id})", fontsize=14, fontweight='bold')
            
            info_text = f"Config: {row['Config']}\n\nHyperparameters:\n"
            clean_params = {k.replace('classifier__', ''): v for k, v in res_data['best_params'].items()}
            for k, v in clean_params.items():
                info_text += f"- {k}: {v}\n"
            
            info_text += f"\nMetrics:\n"
            for m in self.metrics:
                val = row.get(m, 0)
                info_text += f"- {m}: {val:.{decimals}f}\n"

            ax.text(1.35, 0.5, info_text, transform=ax.transAxes, 
                    fontsize=10, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        for j in range(i + 1, len(axes)): axes[j].axis('off')
        plt.suptitle(f"[{self.name}] Confusion Matrices (Sel: {metric_for_selection})", fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_scatter(self, target_model=None, metric=None):
        if metric is None: metric = self.metrics[0]
        df = self._create_results_df(target_model)
        df = df.sort_values('Option_ID')
        plt.figure(figsize=(12, 6))
        sns.stripplot(data=df, x='Config', y=metric, hue='Model', jitter=0.2, dodge=True, size=6, alpha=0.7, palette='deep')
        title_suffix = f"(Focus: {target_model})" if target_model else ""
        plt.title(f"[{self.name}] Performance Distribution: {metric} {title_suffix}", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def plot_parameter_impact(self, target_model=None, metric=None, best_only=False):
        if metric is None: metric = self.metrics[0]
        df_all = self._create_results_df(target_model)
        if target_model is None:
            best_idx = df_all[metric].idxmax()
            target_model = df_all.loc[best_idx, 'Model']
            print(f"ℹ️ Auto-selected model: '{target_model}'")
            df = df_all[df_all['Model'] == target_model]
        else:
            df = df_all 
        df = df.dropna(axis=1, how='all')
        if best_only:
            best_idx = df[metric].idxmax()
            best_opt_id = df.loc[best_idx, 'Option_ID']
            df = df[df['Option_ID'] == best_opt_id]
        std_cols = ['Model', 'Scaling', 'Dummies', 'Option_ID', 'Config'] + self.metrics
        params = [c for c in df.columns if c not in std_cols]
        if not params: return
        num_params = len(params)
        fig, axes = plt.subplots(nrows=1, ncols=num_params, figsize=(6 * num_params, 5), sharey=True)
        if num_params == 1: axes = [axes]
        for i, param in enumerate(params):
            ax = axes[i]
            df[param] = df[param].fillna("None").astype(str)
            try:
                df['sort_col'] = pd.to_numeric(df[param].replace('None', -1))
                df = df.sort_values('sort_col')
            except: df = df.sort_values(param)
            sns.lineplot(data=df, x=param, y=metric, hue='Config', style='Config', markers=True, estimator='mean', ci=100, ax=ax, palette='viridis')
            ax.set_title(f"Impact of {param}")
            ax.set_ylabel(metric if i == 0 else "")
            ax.grid(True, linestyle='--', alpha=0.5)
            if i != num_params - 1: 
                if ax.get_legend(): ax.get_legend().remove()
            else: ax.legend(title="Preprocessing", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.suptitle(f"[{self.name}] Hyperparameter Analysis: {target_model} ({metric})", fontsize=16)
        plt.tight_layout()
        plt.show()