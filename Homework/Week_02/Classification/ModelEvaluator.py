#class to compare classification models









import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score
from sklearn.exceptions import ConvergenceWarning

# TensorFlow / Keras imports for Neural Networks
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping 

class ClassificationEvaluator:
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
        self.test_sets = {} 
        self.target_encoder = None
        self.metrics = ['accuracy', 'f1_macro', 'recall_macro']
        self.nn_histories = {} # Store training history for learning curves
        
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    def add_model(self, name, model, params):
        """ Add a standard Scikit-learn compatible model to test. """
        if hasattr(model, 'random_state'):
            model.random_state = self.random_state
        self.models[name] = {'model': model, 'params': params, 'type': 'sklearn'}

    def add_neural_network(self, name, layers=[(64, 'relu'), (32, 'relu')], epochs=50, batch_size=32, patience=5):
        """
        Add a Neural Network model configuration.
        :param layers: List of tuples (units, activation).
        :param patience: Number of epochs with no improvement after which training will be stopped.
        """
        self.models[name] = {
            'type': 'nn',
            'layers': layers,
            'epochs': epochs,
            'batch_size': batch_size,
            'patience': patience, 
            'params': {} 
        }

    def add_option(self, scaling=False, dummies=False, cols_to_drop=None, pca=False, n_components=0.95):
        """ 
        Add a preprocessing option to the experiment list.
        """
        if cols_to_drop is None: cols_to_drop = []
        self.options.append({
            'scaling': scaling, 
            'dummys': dummies, 
            'cols_to_drop': cols_to_drop,
            'pca': pca,
            'n_components': n_components
        })

    def _build_keras_model(self, n_features, n_classes, layers):
        """ Internal helper to build a compiled Keras model. """
        model = Sequential()
        # Input layer
        model.add(Dense(layers[0][0], activation=layers[0][1], input_shape=(n_features,)))
        
        # Hidden layers
        for units, activation in layers[1:]:
            model.add(Dense(units, activation=activation))
            model.add(Dropout(0.2)) 
            
        # Output layer
        if n_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(Dense(n_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
            
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        return model

    def explore_data_with_pca(self, df, target_col=None, cols_to_drop=None, n_components=0.95):
        """ Performs PCA Analysis directly on the dataframe without model training. """
        print(f"\n{'='*60}")
        print(f"🔍 STARTING PCA EXPLORATION: {self.name.upper()}")
        print(f"{'='*60}")
        
        # 1. Prepare Data
        X = df.copy()
        drop_list = []
        if target_col and target_col in X.columns:
            drop_list.append(target_col)
        if cols_to_drop:
            drop_list.extend(cols_to_drop)
            
        if drop_list:
            X = X.drop(columns=drop_list, errors='ignore')
            print(f"   -> Dropped columns: {drop_list}")

        # 2. Build Pipeline
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include=['number']).columns.tolist()
        
        transformers = []
        if num_cols:
            transformers.append(('num', StandardScaler(), num_cols))
        if cat_cols:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))
            
        if not transformers:
            print("⚠️ No features available for PCA.")
            return

        preprocessor = ColumnTransformer(transformers=transformers)
        
        # 3. Fit PCA
        print("   -> Applying Preprocessing...")
        X_processed = preprocessor.fit_transform(X)
        
        print(f"   -> Fitting PCA (n_components={n_components})...")
        pca = PCA(n_components=n_components, random_state=self.random_state)
        pca.fit(X_processed)
        
        self._plot_pca_visuals(pca, preprocessor, title_suffix="(Exploration)")
        print(f"✅ PCA Exploration Completed.")

    def run_experiments(self, df, target_col, test_size=0.2, cv=5, n_jobs=-1, scoring=None, refit='accuracy'):
        """ Runs the main training loop for all models and options. """
        self.results = {}
        self.test_sets = {} 
        self.nn_histories = {} 
        
        if scoring: self.metrics = scoring
        if refit not in self.metrics:
            refit = self.metrics[0]
        
        self.last_test_size = test_size 

        print(f"\n{'='*60}")
        print(f"🚀 START EXPERIMENT: {self.name.upper()}")
        print(f"{'='*60}")
        print(f"📊 Target Column: '{target_col}'")

        # --- Encode Target ---
        y_raw = df[target_col]
        if y_raw.dtype == 'object' or not np.issubdtype(y_raw.dtype, np.number):
            print(f"ℹ️  Target is categorical. Encoding with LabelEncoder.")
            self.target_encoder = LabelEncoder()
            y_encoded = self.target_encoder.fit_transform(y_raw)
            n_classes = len(self.target_encoder.classes_)
        else:
            self.target_encoder = None
            y_encoded = y_raw.values
            n_classes = len(np.unique(y_encoded))

        for i, option in enumerate(self.options):
            opt_name = f"Opt{i}_S{'1' if option['scaling'] else '0'}_D{'1' if option['dummys'] else '0'}"
            if option['pca']: opt_name += "_PCA"
            
            print(f"\n--- ⚙️ Processing {opt_name} ---")

            X = df.drop(columns=[target_col] + option['cols_to_drop'])
            y = y_encoded

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
            self.test_sets[i] = {'X_test': X_test, 'y_test': y_test}

            # Preprocessor construction
            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            num_cols = X.select_dtypes(include=['number']).columns.tolist()

            transformers = []
            if len(num_cols) > 0:
                if option['scaling']:
                    transformers.append(('num', StandardScaler(), num_cols))
                else:
                    transformers.append(('num', 'passthrough', num_cols))
            
            if len(cat_cols) > 0:
                if option['dummys']:
                    transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), cat_cols))
                else:
                    transformers.append(('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols))

            steps = []
            if transformers:
                steps.append(('preprocessor', ColumnTransformer(transformers=transformers)))
            else:
                steps.append(('preprocessor', ColumnTransformer([('all', 'passthrough', X.columns)])))
            
            if option['pca']:
                steps.append(('pca', PCA(n_components=option['n_components'], random_state=self.random_state)))

            # Fit preprocessor on train data to determine input shape for NN
            prep_pipeline = Pipeline(steps=steps)
            X_train_processed = prep_pipeline.fit_transform(X_train, y_train)
            n_features = X_train_processed.shape[1]

            # Train Models
            for model_name, config in self.models.items():
                
                # --- SKLEARN MODELS ---
                if config['type'] == 'sklearn':
                    model_steps = steps.copy()
                    model_steps.append(('classifier', config['model']))
                    clf = Pipeline(steps=model_steps)
                    pipe_params = {f'classifier__{k}': v for k, v in config['params'].items()}

                    try:
                        grid = GridSearchCV(clf, pipe_params, cv=cv, scoring=self.metrics, refit=refit, n_jobs=n_jobs)
                        grid.fit(X_train, y_train)

                        self.results[f"{opt_name}_{model_name}"] = {
                            'best_score': grid.best_score_,
                            'best_params': grid.best_params_,
                            'best_estimator': grid.best_estimator_, 
                            'cv_results': grid.cv_results_,
                            'option_idx': i,
                            'type': 'sklearn'
                        }
                        print(f"   ✅ {model_name}: {refit}={grid.best_score_:.4f}")
                    except Exception as e:
                        print(f"   ❌ {model_name} Failed: {str(e)}")

                # --- NEURAL NETWORKS ---
                elif config['type'] == 'nn':
                    print(f"   🧠 Training Neural Network: {model_name}...")
                    try:
                        model = self._build_keras_model(n_features, n_classes, config['layers'])
                        
                        # Define Early Stopping Callback
                        early_stop = EarlyStopping(
                            monitor='val_loss', 
                            patience=config.get('patience', 5), 
                            restore_best_weights=True,
                            verbose=0
                        )

                        history = model.fit(
                            X_train_processed, y_train,
                            epochs=config['epochs'],
                            batch_size=config['batch_size'],
                            validation_split=0.2,
                            callbacks=[early_stop], 
                            verbose=0
                        )
                        
                        hist_key = f"{opt_name}_{model_name}"
                        self.nn_histories[hist_key] = history.history
                        
                        # Evaluate on Test set
                        X_test_processed = prep_pipeline.transform(self.test_sets[i]['X_test'])
                        loss, acc = model.evaluate(X_test_processed, self.test_sets[i]['y_test'], verbose=0)
                        
                        self.results[hist_key] = {
                            'best_score': acc,
                            'best_params': str(config['layers']),
                            'best_estimator': model, # The Keras model
                            'cv_results': {'mean_test_accuracy': [acc]},
                            'option_idx': i,
                            'type': 'nn',
                            'preprocessor': prep_pipeline 
                        }
                        print(f"   ✅ {model_name}: accuracy={acc:.4f} (Stopped at epoch {len(history.epoch)})")
                        
                    except Exception as e:
                        print(f"   ❌ {model_name} Failed: {str(e)}")

        print(f"\n🏁 Experiments for {self.name} completed!")

    def _create_results_df(self, target_model=None, show_all_nn=False):
        """Helper to create a DataFrame containing ALL metrics."""
        rows = []
        for key, val in self.results.items():
            parts = key.split('_')
            opt_id = parts[0]
            model_name = parts[-1]
            
            # Reconstruct model name if it contains underscores (e.g. Simple_NN)
            # Find matching model name in self.models keys
            actual_model_name = "Unknown"
            for m_key in self.models.keys():
                if key.endswith(f"_{m_key}"):
                    actual_model_name = m_key
                    break
            
            # Filter if needed
            if target_model and actual_model_name != target_model: continue

            # --- LOGIC FOR SHOWING ALL NNs ---
            # If show_all_nn is True AND this result is a Neural Network,
            # we modify the 'Model' name to include specific details (e.g. layer config)
            # so it appears as a distinct row/column in plots.
            display_model_name = actual_model_name
            if show_all_nn and val.get('type') == 'nn':
                # Use a unique identifier, e.g. the params string or append something unique
                # key is unique per option+model run.
                # Let's append a snippet of params to distinguish
                params_str = str(val.get('best_params', ''))
                # Clean up params string for display
                short_params = params_str.replace('[', '').replace(']', '').replace('relu', 'R').replace('sigmoid', 'S')[:15]
                display_model_name = f"{actual_model_name} ({short_params})"

            option_idx = val['option_idx']
            opt_config = self.options[option_idx]
            cv_results = val['cv_results']
            params = val.get('best_params', {})
            
            scaling_str = 'Yes' if opt_config['scaling'] else 'No'
            pca_str = 'Yes' if opt_config['pca'] else 'No'
            config_str = f"{opt_id} (Scale:{scaling_str} PCA:{pca_str})"
            
            row = {
                'Model': display_model_name, # Use the (potentially modified) name
                'Original_Model': actual_model_name, # Keep original for reference
                'Option_ID': opt_id,
                'Config': config_str,
                'PCA': opt_config['pca'],
                'Type': val.get('type', 'sklearn'), 
                'Result_Key': key 
            }
            
            for metric in self.metrics:
                metric_key = f"mean_test_{metric}"
                if metric_key in cv_results:
                     row[metric] = np.max(cv_results[metric_key])
                elif val['type'] == 'nn' and metric == 'accuracy':
                     row[metric] = val['best_score']

            if isinstance(params, dict):
                for pk, pv in params.items():
                    row[pk.replace('classifier__', '')] = pv
            else:
                row['params'] = params
            
            rows.append(row)
            
        return pd.DataFrame(rows)

    def print_results(self, sort_by=None, top_n=20, decimals=4, show_all_nn=False):
        """ Print a filtered and sorted table of results. """
        df = self._create_results_df(show_all_nn=show_all_nn)
        if df.empty:
            print("No results to display.")
            return

        if sort_by is None: sort_by = self.metrics[0]
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=False)
        
        print(f"\n=== [{self.name}] Top {top_n} Results (Sorted by {sort_by}) ===")
        
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = df[col].round(decimals)

        base_cols = ['Model', 'Config', 'Option_ID']
        metric_cols = [m for m in self.metrics if m in df.columns]
        param_cols = [c for c in df.columns if c not in base_cols + metric_cols + ['PCA', 'Type', 'Result_Key', 'Original_Model']]
        
        print(df[base_cols + metric_cols + param_cols].head(top_n).to_string(index=False))

    def plot_heatmap(self, metric=None, decimals=4, show_all_nn=False):
        """
        Plots a heatmap of the results.
        :param show_all_nn: If True, each Neural Network run will be a separate column.
        """
        if metric is None: metric = self.metrics[0]
        
        df = self._create_results_df(show_all_nn=show_all_nn)
        if metric not in df.columns: return

        # Pivot: Rows=Option, Cols=Model (which is now unique for NNs if show_all_nn=True)
        pivot_df = df.pivot_table(index='Option_ID', columns='Model', values=metric, aggfunc='max')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt=f".{decimals}f", ax=ax, cbar_kws={'label': metric})
        plt.title(f"[{self.name}] Model Performance ({metric})", size=14)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, model_filter=None, option_filter=None, metric_for_selection=None, decimals=4, show_all_nn=False):
        if metric_for_selection is None: metric_for_selection = self.metrics[0]

        df_res = self._create_results_df(show_all_nn=show_all_nn)
        if model_filter: 
            # Check against 'Original_Model' to allow filtering on base type
            df_res = df_res[df_res['Original_Model'] == model_filter]
        if option_filter: df_res = df_res[df_res['Option_ID'] == option_filter]
        if df_res.empty: return

        # Group by 'Model' (which is unique per NN config if show_all_nn=True) and take best
        best_rows = df_res.sort_values(by=metric_for_selection, ascending=False).groupby('Model').first().reset_index()

        num_plots = len(best_rows)
        cols = min(num_plots, 3)
        rows = (num_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
        if num_plots == 1: axes = [axes]
        axes = np.array(axes).flatten()

        for i, (_, row) in enumerate(best_rows.iterrows()):
            display_model_name = row['Model'] # This might be "Simple_NN (params)"
            original_model_name = row['Original_Model']
            opt_id = row['Option_ID']
            target_key = row.get('Result_Key')
            
            if not target_key: continue

            res_data = self.results[target_key]
            opt_idx = res_data['option_idx']
            X_test = self.test_sets[opt_idx]['X_test']
            y_test = self.test_sets[opt_idx]['y_test']
            
            if res_data['type'] == 'sklearn':
                best_model = res_data['best_estimator'] 
                y_pred = best_model.predict(X_test)
            else: # NN
                model = res_data['best_estimator'] 
                if 'preprocessor' in res_data:
                    pipeline = res_data['preprocessor']
                    X_test_processed = pipeline.transform(X_test)
                    y_prob = model.predict(X_test_processed, verbose=0)
                    if y_prob.shape[1] > 1:
                        y_pred = np.argmax(y_prob, axis=1)
                    else:
                         y_pred = (y_prob > 0.5).astype(int).flatten()
                else:
                    print(f"⚠️ Cannot plot CM for '{display_model_name}'. Pipeline not stored.")
                    continue

            labels = self.target_encoder.classes_ if self.target_encoder else None
            
            ax = axes[i]
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues', colorbar=False, display_labels=labels)
            ax.set_title(f"{display_model_name}\n({opt_id})", fontsize=14, fontweight='bold')
            if labels is not None and len(labels) > 5:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        for j in range(i + 1, len(axes)): axes[j].axis('off')
        plt.tight_layout()
        plt.show()

    def _plot_pca_visuals(self, pca, preprocessor, title_suffix=""):
        plt.figure(figsize=(10, 5))
        n_comps = len(pca.explained_variance_ratio_)
        plt.bar(range(1, n_comps + 1), pca.explained_variance_ratio_, alpha=0.7, color='skyblue')
        plt.step(range(1, n_comps + 1), np.cumsum(pca.explained_variance_ratio_), where='mid', color='red')
        plt.title(f"PCA Analysis: Explained Variance {title_suffix}", fontsize=14)
        plt.show()

        try:
            feats = [f.split('__')[-1] for f in preprocessor.get_feature_names_out()]
        except:
            feats = [f"Feat_{i}" for i in range(pca.components_.shape[1])]
            
        comps_df = pd.DataFrame(pca.components_, columns=feats[:pca.components_.shape[1]])
        plt.figure(figsize=(12, 6))
        sns.heatmap(comps_df, cmap='RdBu', center=0)
        plt.title('PCA Feature Loadings', fontsize=14)
        plt.show()

    def plot_learning_curve(self, model_name_filter=None):
        """ Plots training history for Neural Networks. """
        if not self.nn_histories:
            print("⚠️ No Neural Network history found.")
            return

        for key, history in self.nn_histories.items():
            if model_name_filter and model_name_filter not in key: continue
                
            df_hist = pd.DataFrame(history)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            df_hist[['loss', 'val_loss']].plot(ax=axes[0])
            axes[0].set_title(f"Loss: {key}")
            axes[0].grid(True)
            
            if 'accuracy' in df_hist.columns:
                df_hist[['accuracy', 'val_accuracy']].plot(ax=axes[1])
                axes[1].set_title(f"Accuracy: {key}")
                axes[1].set_ylim(0, 1)
                axes[1].grid(True)
            plt.show()

    def plot_parameter_impact(self, target_model, metric='accuracy', show_all_nn=False):
        df = self._create_results_df(target_model, show_all_nn=show_all_nn)
        if df.empty: return
        
        std_cols = ['Model', 'Config', 'Option_ID', 'PCA', 'Type', 'Result_Key', 'Original_Model'] + self.metrics
        params = [c for c in df.columns if c not in std_cols]
        
        if not params: return

        fig, axes = plt.subplots(1, len(params), figsize=(6*len(params), 5))
        if len(params) == 1: axes = [axes]
        
        for i, p in enumerate(params):
            df[p] = df[p].fillna('None').astype(str)
            sns.boxplot(data=df, x=p, y=metric, hue='Config', ax=axes[i], palette='viridis')
            axes[i].set_title(f"Impact of {p}")
            axes[i].grid(True, linestyle='--', alpha=0.5)
            
        plt.suptitle(f"Hyperparameter Impact: {target_model} ({metric})", fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_scatter(self, target_model=None, metric=None, show_all_nn=False):
        if metric is None: metric = self.metrics[0]
        df = self._create_results_df(target_model, show_all_nn=show_all_nn)
        df = df.sort_values('Option_ID')
        plt.figure(figsize=(12, 6))
        sns.stripplot(data=df, x='Config', y=metric, hue='Model', jitter=0.2, dodge=True, size=6, alpha=0.7, palette='deep')
        title_suffix = f"(Focus: {target_model})" if target_model else ""
        plt.title(f"[{self.name}] Performance Distribution: {metric} {title_suffix}", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.show()