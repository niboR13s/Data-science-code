#code to comprare Regression models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.exceptions import ConvergenceWarning

class RegressionEvaluator:
    def __init__(self, name="Regression Evaluator", random_state=42):
        """
        Initialize the evaluator with a name and fixed random state for reproducibility.
        """
        self.name = name
        self.random_state = random_state
        self.models = {}
        self.options = []
        self.results = {}
        self.test_sets = {}
        # Default metrics for regression (scikit-learn uses neg_ for maximization)
        self.metrics = ['neg_root_mean_squared_error', 'r2', 'neg_mean_absolute_error']
        
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    def add_model(self, name, model, params):
        """ Add a model configuration to the evaluator. """
        if hasattr(model, 'random_state'):
            model.random_state = self.random_state
        self.models[name] = {'model': model, 'params': params}

    def add_option(self, scaling=False, dummies=False, cols_to_drop=None):
        """ Add a preprocessing option (experiment setup). """
        if cols_to_drop is None: cols_to_drop = []
        self.options.append({'scaling': scaling, 'dummys': dummies, 'cols_to_drop': cols_to_drop})

    def run_experiments(self, df, target_col, test_size=0.2, cv=5, n_jobs=-1, refit='r2'):
        """
        Runs the main training loop: tests every model with every preprocessing option.
        """
        self.results = {}
        self.test_sets = {}
        
        print(f"\n{'='*60}")
        print(f"🚀 START REGRESSION EXPERIMENT: {self.name.upper()}")
        print(f"{'='*60}")
        print(f"📊 Target Column: '{target_col}'")
        print(f"📊 Optimizing for: {refit}")

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
            # Save test set for later visualization
            self.test_sets[i] = {'X_test': X_test, 'y_test': y_test}

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
                # Fallback for models that can't handle strings
                transformers.append(('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols))

            preprocessor = ColumnTransformer(transformers=transformers)

            # 4. Train Models
            for model_name, config in self.models.items():
                clf = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', config['model'])])
                
                pipe_params = {f'regressor__{k}': v for k, v in config['params'].items()}

                # Run GridSearch
                grid = GridSearchCV(clf, pipe_params, cv=cv, scoring=self.metrics, refit=refit, n_jobs=n_jobs)
                grid.fit(X_train, y_train)

                # Store results
                self.results[f"{opt_name}_{model_name}"] = {
                    'best_score': grid.best_score_,
                    'best_params': grid.best_params_,
                    'best_estimator': grid.best_estimator_, # Stored with Key 'best_estimator' (no underscore)
                    'cv_results': grid.cv_results_,
                    'option_idx': i
                }
                print(f"   ✅ {model_name}: {refit}={grid.best_score_:.4f}")

        print(f"\n🏁 Experiments completed!")

    def _create_results_df(self, target_model=None):
        """ Internal helper to flatten results into a DataFrame. """
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
                    'Option_ID': opt_id,
                    'Config': f"{opt_id} (Scale:{'Yes' if opt_config['scaling'] else 'No'} Dum:{'Yes' if opt_config['dummys'] else 'No'})"
                }
                
                # Extract scores and handle negative metrics
                for metric in self.metrics:
                    mean_score = cv_results[f"mean_test_{metric}"][i]
                    if 'neg_' in metric:
                        row[metric.replace('neg_', '')] = abs(mean_score)
                    else:
                        row[metric] = mean_score
                
                # Add hyperparameters
                for pk, pv in params[i].items():
                    row[pk.replace('regressor__', '')] = pv
                rows.append(row)
        return pd.DataFrame(rows)

    def _is_higher_better(self, metric):
        """ Helper to determine sort order. """
        if 'r2' in metric.lower() or 'score' in metric.lower() or 'accuracy' in metric.lower():
            return True
        return False

    def print_results(self, sort_by='r2', top_n=20):
        """ Print a sorted table of the best results. """
        df = self._create_results_df()
        ascending = not self._is_higher_better(sort_by)
        
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=ascending)
            
        print(f"\n=== [{self.name}] Top {top_n} Results (Sorted by {sort_by}) ===")
        df_display = df.dropna(axis=1, how='all')
        print(df_display.head(top_n).to_string(index=False))

    # --- VISUALIZATIONS ---

    def plot_actual_vs_predicted(self, model_filter=None, option_filter=None, best_metric='r2'):
        """ Plots Actual vs Predicted values. Ideal is a diagonal line. """
        df_res = self._create_results_df()
        
        if model_filter: df_res = df_res[df_res['Model'] == model_filter]
        if option_filter: df_res = df_res[df_res['Option_ID'] == option_filter]
        
        if df_res.empty:
            print("⚠️ No results found matching filters.")
            return

        # Find best model index
        if self._is_higher_better(best_metric):
            best_idx = df_res[best_metric].idxmax()
        else:
            best_idx = df_res[best_metric].idxmin()
            
        row = df_res.loc[best_idx]
        
        model_name = row['Model']
        opt_id = row['Option_ID']
        
        # Retrieve best estimator from results
        found_key = None
        for k in self.results.keys():
            if k.endswith(f"_{model_name}") and k.startswith(f"{opt_id}_"):
                found_key = k
                break
        
        if not found_key:
            print("⚠️ Could not find model key in results.")
            return

        # <--- FIX: Key is 'best_estimator' (without underscore) to match storage
        best_model = self.results[found_key]['best_estimator'] 
        opt_idx = self.results[found_key]['option_idx']
        
        # Get Data
        test_data = self.test_sets[opt_idx]
        y_test = test_data['y_test']
        y_pred = best_model.predict(test_data['X_test'])
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, edgecolor='k', label='Data Points')
        
        # Diagonal line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        plt.title(f"Actual vs Predicted: {model_name} ({opt_id})\nR2: {r2:.3f} | RMSE: {rmse:.2f}")
        plt.xlabel("Actual Value")
        plt.ylabel("Predicted Value")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def plot_parameter_impact(self, target_model=None, metric='r2'):
        """ Visualizes how different hyperparameters affect the score. """
        df = self._create_results_df(target_model)
        
        # Auto-select model if None
        if target_model is None:
            if self._is_higher_better(metric):
                best_idx = df[metric].idxmax()
            else:
                best_idx = df[metric].idxmin()
            target_model = df.loc[best_idx, 'Model']
            print(f"ℹ️ Auto-selected model: '{target_model}' (Best {metric})")
            df = df[df['Model'] == target_model]
        
        # Identify Parameters
        std_cols = ['Model', 'Option_ID', 'Config', 'root_mean_squared_error', 'r2', 'mean_absolute_error']
        params = [c for c in df.columns if c not in std_cols]
        params = [c for c in params if df[c].notna().any()] 
        
        if not params:
            print("No varying parameters to plot.")
            return

        # Plot
        fig, axes = plt.subplots(1, len(params), figsize=(6*len(params), 5))
        if len(params) == 1: axes = [axes]
        
        for i, p in enumerate(params):
            df[p] = df[p].fillna('None').astype(str)
            
            # Show Boxplot
            sns.boxplot(data=df, x=p, y=metric, hue='Config', ax=axes[i], palette='viridis')
            
            axes[i].set_title(f"Impact of {p}")
            axes[i].grid(True, linestyle='--', alpha=0.5)
            
            # Legend handling
            if i == len(params) - 1:
                axes[i].legend(title="Preprocessing", bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                if axes[i].get_legend():
                    axes[i].get_legend().remove()
            
        plt.suptitle(f"Hyperparameter Impact: {target_model} ({metric})", fontsize=16)
        plt.tight_layout()
        plt.show()