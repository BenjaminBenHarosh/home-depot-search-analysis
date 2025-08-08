# === Standard library ===
import time  # for measuring execution time
import warnings  # for suppressing warnings

# === Data manipulation and visualization ===
import numpy as np  # numerical operations
import pandas as pd  # dataframes and CSV I/O
import matplotlib.pyplot as plt  # plotting and visualization

# === Natural language processing (NLP) ===
from nltk.stem.snowball import SnowballStemmer  # word stemming
from nltk import ngrams  # generate n-grams from text
from fuzzywuzzy import fuzz  # fuzzy string matching
from sklearn.feature_extraction.text import TfidfVectorizer  # convert text to TF-IDF vectors

# === Similarity and evaluation metrics ===
from sklearn.metrics import mean_squared_error  # RMSE calculation
from sklearn.metrics.pairwise import cosine_similarity  # cosine similarity between vectors

# === Model selection and preprocessing ===
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score  # data splitting, CV, tuning
from sklearn.preprocessing import StandardScaler  # feature scaling
from sklearn.pipeline import make_pipeline  # pipeline creation
from sklearn.base import clone  # duplicate models

# === Regressors (machine learning models) ===
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.svm import SVR  # support vector regression
from sklearn.neighbors import KNeighborsRegressor  # k-nearest neighbors

# === Statistical distributions for tuning ===
from scipy.stats import randint, uniform  # distributions for randomized search
from scipy.stats import ttest_rel  # paired t-test

# === Jupyter / Display ===
from IPython.display import display  # pretty display in notebooks



materials = {'wood', 'metal', 'steel', 'aluminum', 'plastic', 'mdf', 'copper',
    'brass', 'glass', 'rubber', 'fiberglass', 'vinyl', 'ceramic', 'stone',
    'polyester', 'nylon', 'composite', 'bamboo'}

units = {'in', 'inch', 'inches', 'ft', 'feet', 'mm', 'cm', 'm', 'oz', 'lbs',
    'pound', 'gallon', 'ml', 'liter', 'litre', 'quart', 'yard'}

colors = {'white', 'black', 'gray', 'grey', 'blue', 'red', 'green', 'yellow',
    'brown', 'silver', 'gold', 'beige', 'ivory', 'navy', 'tan', 'orange',
    'pink', 'charcoal', 'bronze', 'teal', 'maroon'}

stemmer = SnowballStemmer('english')

all_features = [
'query_length',               # Number of words in the search query
'common_words',               # Common word count between query and both product title and description
'brand_match',                # Overlap with product brand
'tfidf_similarity',           # Cosine similarity between query and title+description
'query_has_number',           # Presence of a numeric value in query
'unit_match',                 # Match on measurement units (e.g., ft, inch)
'initial_term_match',         # Do the first query words appear early in the title?
'material_match',             # Match between query and product material
'color_match',                # Match between query and product color

'jaccard',              # Jaccard similarity between query and product title/description
'bigram_overlap',       # Count of overlapping bigrams between query and product title
'fuzzy'                 # Fuzzy matching score (token sort ratio) between query and product title/description
] 

# Manually defined feature sets used for experimentation.
# Feature sets are grouped by size (from 1 to 9 features) and were handpicked based on domain knowledge,
# prior results, and feature diversity. Due to time and computational constraints, it wasn't feasible
# to exhaustively search the full feature combination space. While it's possible that better-performing
# combinations exist, these sets were selected to balance practicality with expected effectiveness.

# Mandatory 1-feature sets
feature_sets = [
    ['tfidf_similarity'],
    ['initial_term_match'],
    ['query_length'],
    ['jaccard'],
    ['fuzzy'],
    ['query_has_number'],
    ['bigram_overlap'],
    ['color_match'],
    ['brand_match'],
    ['common_words'],
    ['unit_match'],         
    ['material_match'],]

# 2-feature sets
feature_sets += [ ['query_length', 'common_words'],]
# 3-feature sets
feature_sets += [ ['query_length', 'common_words', 'initial_term_match'], 
                  ['tfidf_similarity', 'query_length', 'common_words'],
                  ['tfidf_similarity', 'query_length', 'initial_term_match'],]
# 4-feature sets
feature_sets += [
    ['tfidf_similarity', 'query_length', 'initial_term_match', 'common_words'],]
# 5-feature sets
feature_sets += [
    ['tfidf_similarity', 'query_length', 'initial_term_match', 'common_words', 'fuzzy'],]
# 6-feature sets
feature_sets += [
    ['tfidf_similarity', 'query_length', 'jaccard', 'query_has_number', 'common_words', 
     'color_match'],
    ['tfidf_similarity', 'query_length', 'initial_term_match', 'common_words', 'fuzzy', 
     'jaccard'],]

# 7-feature sets
feature_sets += [ ['query_length', 'initial_term_match', 'jaccard', 'common_words', 
                   'color_match', 'fuzzy', 'bigram_overlap'],
                  ['tfidf_similarity', 'query_length', 'initial_term_match', 'common_words', 
                   'fuzzy', 'jaccard', 'query_has_number'],]
# 8-feature set
feature_sets += [['tfidf_similarity', 'query_length', 'initial_term_match', 'fuzzy', 'jaccard',
                  'common_words', 'query_has_number', 'bigram_overlap' ],]
# 9-feature set
feature_sets += [['tfidf_similarity', 'query_length', 'initial_term_match','fuzzy', 'jaccard',
                  'common_words', 'query_has_number', 'color_match', 'unit_match'],]

# 10-feature set
feature_sets += [['tfidf_similarity', 'query_length', 'initial_term_match','fuzzy', 'jaccard','common_words', 'query_has_number', 'color_match', 'material_match', 'brand_match'],]

# 11-feature set
feature_sets += [['tfidf_similarity', 'query_length', 'initial_term_match', 'fuzzy', 'jaccard',
                  'common_words', 'query_has_number', 'color_match', 'material_match',
                  'brand_match', 'unit_match']]
# 12-feature set
feature_sets += [['tfidf_similarity', 'query_length', 'initial_term_match', 'fuzzy', 'jaccard',
                  'common_words', 'query_has_number', 'color_match', 'material_match',
                  'brand_match', 'unit_match', 'bigram_overlap']]

def str_stemmer(s):
	return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())

def add_query_length(df):
    df['query_length'] = df['search_term'].map(lambda x: len(x.split())).astype(np.int64)
    return df

def jaccard_similarity(str1, str2):
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0

def bigram_overlap(str1, str2):
    query_ngrams = set(ngrams(str1.lower().split(), 2))
    text_ngrams = set(ngrams(str2.lower().split(), 2))
    return len(query_ngrams.intersection(text_ngrams))

def fuzzy_ratio(str1, str2):
    return fuzz.token_sort_ratio(str1, str2)

def add_jaccard_similarity(df):
    df['jaccard_title'] = df.apply(lambda row: jaccard_similarity(row['search_term'], row['product_title']), axis=1)
    df['jaccard_description'] = df.apply(lambda row: jaccard_similarity(row['search_term'], row['product_description']), axis=1)
    return df

def add_bigram_overlap(df):
    df['bigram_overlap_title'] = df.apply(lambda row: bigram_overlap(row['search_term'], row['product_title']), axis=1)
    df['bigram_overlap_description'] = df.apply(lambda row: bigram_overlap(row['search_term'], row['product_description']), axis=1)
    return df

def add_fuzzy_ratio(df):
    df['fuzzy_title'] = df.apply(lambda row: fuzzy_ratio(row['search_term'], row['product_title']), axis=1)
    df['fuzzy_description'] = df.apply(lambda row: fuzzy_ratio(row['search_term'], row['product_description']), axis=1)
    return df

def add_common_word_features(df):
    df['product_info'] = df['search_term'] + "\t" + df['product_title'] + "\t" + df['product_description']
    df['title_overlap'] = df['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
    df['description_overlap'] = df['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))
    df.drop('product_info', axis=1, inplace=True)
    return df

def add_brand_match(df, df_attr):
    df_brand = df_attr[df_attr['name'] == "MFG Brand Name"][['product_uid', 'value']].rename(columns={'value': 'brand'})
    df = pd.merge(df, df_brand, how='left', on='product_uid')
    df['brand'] = df['brand'].fillna('').map(str_stemmer)
    df['brand_match'] = df.apply(lambda x: str_common_word(x['search_term'], x['brand']), axis=1)
    df.drop('brand', axis=1, inplace=True)
    return df

def add_tfidf_similarity(df):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['product_title'] + " " + df['product_description'])
    query_matrix = tfidf.transform(df['search_term'])
    df['tfidf_similarity'] = [cosine_similarity(query_matrix[i], tfidf_matrix[i])[0, 0] for i in range(len(df))]
    return df

def add_query_has_number(df):
    df['query_has_number'] = df['search_term'].map(
        lambda x: sum(any(char.isdigit() for char in token) for token in x.split())
    )
    return df
    
def add_unit_match(df):
    def unit_overlap(row):
        query_tokens = set(row['search_term'].split())
        product_text = row['product_title'] + " " + row['product_description']
        return int(any(unit in product_text for unit in query_tokens & units))
    df['unit_match'] = df.apply(unit_overlap, axis=1)
    return df
    
def add_initial_term_match(df):
    def count_early_hits(row):
        query_words = row['search_term'].split()[:2]
        return sum(word in row['product_title'] for word in query_words)
    df['initial_term_match'] = df.apply(count_early_hits, axis=1)
    return df

def add_material_match(df, df_attr):
    df_material = df_attr[df_attr['name'].str.contains("Material", case=False, na=False)]
    df_material = df_material[['product_uid', 'value']].rename(columns={'value': 'material'})
    df = pd.merge(df, df_material, how='left', on='product_uid')
    df['material'] = df['material'].fillna('').map(str_stemmer)
    df['material_match'] = df.apply(lambda x: int(any(mat in x['material'] for mat in x['search_term'].split() if mat in materials)), axis=1)
    df.drop('material', axis=1, inplace=True)
    return df

def add_color_match(df, df_attr):
    df_color = df_attr[df_attr['name'].str.contains("Color", case=False, na=False)]
    df_color = df_color[['product_uid', 'value']].rename(columns={'value': 'color'})
    df = pd.merge(df, df_color, how='left', on='product_uid')
    df['color'] = df['color'].fillna('').map(str_stemmer)
    df['color_match'] = df.apply(lambda x: int(any(color in x['search_term'] and color in x['color'] for color in colors)), axis=1)
    df.drop('color', axis=1, inplace=True)
    return df

def build_feature_set(df_raw, df_attr, features_to_include, stem=True):
    """
    Constructs a dataframe containing selected engineered features for machine learning models.

    This function applies a series of feature extraction functions based on a user-specified list.
    It can also optionally apply stemming to the input text columns. 
    """
    df = df_raw.copy()
    
    if stem:
        df[['search_term', 'product_title', 'product_description']] = df[['search_term', 'product_title', 'product_description']].apply(lambda col: col.map(str_stemmer))

    if 'query_length' in features_to_include:
        df = add_query_length(df)
    if 'common_words' in features_to_include:
        df = add_common_word_features(df)
    if 'brand_match' in features_to_include:
        df = add_brand_match(df, df_attr)
    if 'tfidf_similarity' in features_to_include:
        df = add_tfidf_similarity(df)
    if 'query_has_number' in features_to_include:
        df = add_query_has_number(df)
    if 'unit_match' in features_to_include:
        df = add_unit_match(df)
    if 'initial_term_match' in features_to_include:
        df = add_initial_term_match(df)
    if 'material_match' in features_to_include:
        df = add_material_match(df, df_attr)
    if 'color_match' in features_to_include:
        df = add_color_match(df, df_attr)
    if 'jaccard' in features_to_include:
        df = add_jaccard_similarity(df)
    if 'bigram_overlap' in features_to_include:
        df = add_bigram_overlap(df)
    if 'fuzzy' in features_to_include:
        df = add_fuzzy_ratio(df)
        
    df.drop(['search_term', 'product_title', 'product_description','product_uid'], axis=1, inplace=True)
    return df

def evaluate_model(model, X_train, y_train, name="Unnamed Model"):
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv_scores = cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5)
    cv_rmse_mean = -np.mean(cv_scores)
    cv_rmse_std = np.std(cv_scores)
    
    print(f"{name}: RMSE = {rmse:.4f}, CV RMSE = {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}, Train time = {train_time:.2f} sec")
    return name, rmse, train_time

def evaluate_feature_set(df_all, df_attr, features, num_train, stem=True, model_params=None):
    """
    Evaluates a specific set of features using both RMSE on a train/test split and cross-validated RMSE.
    """
    start_time = time.time()

    df_features = build_feature_set(df_all.copy(), df_attr, features, stem=stem)
    df_train = df_features.iloc[:num_train]
    X = df_train.drop(columns=['id', 'relevance'])
    y = df_train['relevance']

    if model_params is None:
        model_params = {
            'n_estimators': 100,
            'max_depth': 7,
            'random_state': 42,
            'n_jobs': -1
        }

    model = RandomForestRegressor(**model_params)
    
    # 80/20 train-test RMSE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse_split = np.sqrt(mean_squared_error(y_test, y_pred))

    # Cross-validated RMSE
    cv_scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=5)
    rmse_cv = -np.mean(cv_scores)
    rmse_std = np.std(cv_scores)

    elapsed_time = time.time() - start_time

    print(f"Features: {features} --> RMSE: {rmse_split:.4f}, CV RMSE: {rmse_cv:.4f} ± {rmse_std:.4f} (train time: {elapsed_time:.2f}s)")
    
    return rmse_split, rmse_cv, rmse_std


def compare_models(df_all, df_attr, feature_set, num_train, stem=True):
    """
    Compare multiple regression models using the same feature set.
    """
    print("\n=== Model Comparison ===")

    # Build features
    df_features = build_feature_set(df_all.copy(), df_attr, feature_set, stem=stem)
    df_train = df_features.iloc[:num_train]
    X = df_train.drop(columns=['id', 'relevance'])
    y = df_train['relevance']

    # Define models
    models = [
        ("Random Forest", RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=15, max_depth=6, random_state=0)),
        ("Support Vector Regressor", make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.0, epsilon=0.2))),
        ("KNN", KNeighborsRegressor(n_neighbors=5))
    ]

    # Evaluate models
    results = []
    for name, model in models:
        model_name, rmse, train_time = evaluate_model(model, X, y, name)
        results.append((model_name, rmse, f"{train_time:.2f} sec"))

    # Display results
    results_df = pd.DataFrame(results, columns=["Model", "RMSE", "Train time (s)"])
    results_df.sort_values("RMSE", inplace=True)
    print("\nModel Comparison Results:")
    print(results_df.to_string(index=False))
    return results_df

def tune_model(model, param_dist, X, y, model_name="Model", n_iter=20, cv=3):
    print(f"\n--- Tuning {model_name} ---")
    
    # Split data for final evaluation (not used in tuning)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Measure training time
    start_time = time.time()
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Cross-validated RMSE from RandomizedSearchCV (best score)
    best_rmse = -search.best_score_
    std_rmse = np.std(search.cv_results_['mean_test_score'])  # std of CV scores (neg RMSEs)

    # Final model RMSE on held-out test split
    y_pred = search.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Display results
    print(f"{model_name} RMSE: {test_rmse:.4f}")
    print(f"\nBest {model_name} CV RMSE: {best_rmse:.4f} ± {std_rmse:.4f}")
    print(f"Training time: {train_time:.2f} seconds")

    print(f"\nBest {model_name} Parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    
    return search

def plot_feature_importance(model, X, y, feature_names, top_n=5, plot=True, save_path=None):
    model.fit(X, y)
    importances = model.feature_importances_
    
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)

    print(f"\nTop {top_n} most important features:")
    print(importance_df.head(top_n))

    if plot:
        plt.figure(figsize=(10, 6))
        bars = plt.barh(importance_df['feature'][:top_n][::-1],
                        importance_df['importance'][:top_n][::-1],
                        color='steelblue')

        plt.xlim(0, 1.075 * max(importance_df['importance']))
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.005,
                     bar.get_y() + bar.get_height() / 2,
                     f"{width:.3f}",
                     va='center')

        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.xlabel("Feature importance")
        plt.title("Top Feature Importances")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
    
    return importance_df

def evaluate_on_test_set(df_all, df_attr, features, num_train, stem=True, model_params=None):
    df_features = build_feature_set(df_all.copy(), df_attr, features, stem=stem)
    df_train = df_features.iloc[:num_train]
    df_test = df_features.iloc[num_train:]

    X_train = df_train.drop(columns=['id', 'relevance'])
    y_train = df_train['relevance']
    X_test = df_test.drop(columns=['id', 'relevance'], errors='ignore')

    # Use tuned params if provided, otherwise use:
    if model_params is None:
        model_params = {
            'n_estimators': 100,
            'max_depth': 7,
            'random_state': 42,
            'n_jobs': -1}

    model = RandomForestRegressor(**model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return pd.DataFrame({'id': df_test['id'].values, 'relevance': y_pred})

def plot_relevance_histogram(df_train, save_path="relevance_histogram_annotated.png"):
    plt.figure(figsize=(6.5, 4.875))
    bins = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    counts, edges, _ = plt.hist(df_train['relevance'], bins=bins, edgecolor='black', rwidth=0.7)

    for count, left_edge in zip(counts, edges[:-1]):
        center = left_edge + (bins[1] - bins[0]) / 2
        plt.text(center, count + 300, f'{int(count)}', ha='center', fontsize=10)
    plt.xlabel("Relevance Score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(bins)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_relevance_boxplot(df_train, save_path="relevance_boxplot.png"):
    median_value = df_train['relevance'].median()
    mean_value = df_train['relevance'].mean()

    plt.figure(figsize=(3.75, 5))
    plt.boxplot(df_train['relevance'], vert=True, patch_artist=True, showfliers=True,
                boxprops=dict(color='black'),
                medianprops=dict(color='#2CA02C', linewidth=2, linestyle='--'), 
                whiskerprops=dict(color='black'), 
                capprops=dict(color='black'))

    x_center = 1  
    box_width = 0.075
    plt.hlines(y=mean_value, xmin=x_center - box_width, xmax=x_center + box_width,
               color='#2CA02C', linestyle='--', linewidth=2)
    plt.ylabel("Relevance score", fontsize=12)
    plt.xlabel("Relevance data", fontsize=12)
    plt.xticks([])
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.text(1.1, median_value - 0.03, f'Median = {median_value:.2f}', color='#2CA02C', fontsize=10, va='center')
    plt.text(1.1, mean_value, f'Mean = {mean_value:.2f}', color='#2CA02C', fontsize=10, va='center')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_overfitting_curve(results_df, save_path="feature_count_vs_rmse.png"):
    plt.figure(figsize=(8, 5))

    for stem_flag in [True, False]:
        subset = results_df[results_df['Stemming'] == stem_flag]
        
        # Use idxmin to get rows with the lowest RMSE (CV)
        grouped = (
            subset.loc[subset.groupby("Num Features")['RMSE (CV)'].idxmin()]
            .sort_values("Num Features")
            .reset_index(drop=True)
        )

        label = "With Stemming" if stem_flag else "Without Stemming"

        plt.errorbar(grouped["Num Features"], grouped["RMSE (CV)"],
                     yerr=grouped["RMSE_STD"], fmt='-o', capsize=5, label=label)

    plt.xlabel("Number of Features")
    plt.ylabel("Cross-Validated RMSE (±1 std)")
    plt.title("Model Complexity vs Performance")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def run_data_exploration(df_train, df_attr):
    print("\n=== Data Exploration ===")
    total_pairs = df_train.shape[0]
    unique_products = df_train['product_uid'].nunique()
    top_products = df_train['product_uid'].value_counts().head(2)
    top1_uid, top2_uid = top_products.index
    top1_title = df_train[df_train['product_uid'] == top1_uid]['product_title'].iloc[0]
    top2_title = df_train[df_train['product_uid'] == top2_uid]['product_title'].iloc[0]
    relevance_stats = df_train['relevance'].describe()

    summary_df = pd.DataFrame({
        "Metric": [
            "Total product-query pairs", "Unique product count", "Top 1 product ID",
            "Top 1 product title", "Top 1 count", "Top 2 product ID",
            "Top 2 product title", "Top 2 count", "Relevance Mean",
            "Relevance Median", "Relevance Std Dev"
        ],
        "Value": [
            total_pairs, unique_products, top1_uid, top1_title,
            top_products.iloc[0], top2_uid, top2_title, top_products.iloc[1],
            f"{relevance_stats['mean']:.3f}",
            f"{df_train['relevance'].median():.3f}",
            f"{relevance_stats['std']:.3f}"
        ]
    })

    print("Data Description Summary")
    display(summary_df)

    print("\nTop-5 Most Common Brand Names")
    top_brands = (df_attr[df_attr['name'] == 'MFG Brand Name']['value']
                  .value_counts().head(6).rename_axis('Brand Name')
                  .reset_index(name='Count'))
    display(top_brands)

    plot_relevance_histogram(df_train)
    plot_relevance_boxplot(df_train)

def run_baseline_evaluation(raw_data, df_attr, num_train):
    print("\n=== Baseline Evaluation ===")
    baseline_features = ['query_length', 'common_words']
    baseline_model = BaggingRegressor(
        estimator=RandomForestRegressor(n_estimators=15, max_depth=6, random_state=42, n_jobs=-1),
        n_estimators=45, random_state=42, n_jobs=-1,max_samples=0.1)

    for stem_flag in [True, False]:
        label = "Stemmed" if stem_flag else "No Stem"
        print(f"\n[Baseline Model - {label}]")
        df_baseline = build_feature_set(raw_data.copy(), df_attr, baseline_features, stem=stem_flag)
        df_train_baseline = df_baseline.iloc[:num_train]
        X = df_train_baseline.drop(columns=['id', 'relevance'])
        y = df_train_baseline['relevance']
        evaluate_model(baseline_model, X, y, name=f"Baseline ({label})")

def run_feature_set_evaluation(raw_data, df_attr, num_train, feature_sets, model_params=None):
    print("\n=== Feature Evaluation ===")
    results = []
    
    for stem_flag in [True, False]:
        print(f"\n--- {'With' if stem_flag else 'Without'} Stemming ---")
        for features in feature_sets:
            start_time = time.time()

            df_features = build_feature_set(raw_data.copy(), df_attr, features, stem=stem_flag)
            df_train = df_features.iloc[:num_train]
            X = df_train.drop(columns=['id', 'relevance'])
            y = df_train['relevance']

            if model_params is None:
                model_params = {
                    'n_estimators': 100,
                    'max_depth': 7,
                    'random_state': 42,
                    'n_jobs': -1
                }

            base_model = GradientBoostingRegressor(**model_params)
            model = BaggingRegressor(estimator=base_model, n_estimators=45, random_state=42, n_jobs=-1, max_samples=0.1)

            # Train-test split RMSE
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse_split = np.sqrt(mean_squared_error(y_test, y_pred))

            # Cross-validation RMSE (with fresh clone)
            cv_model = clone(model)
            cv_scores = cross_val_score(cv_model, X, y, scoring='neg_root_mean_squared_error', cv=5)
            rmse_cv = -np.mean(cv_scores)
            rmse_std = np.std(cv_scores)

            elapsed_time = time.time() - start_time

            results.append({
                "Features": ", ".join(features),
                "Num Features": len(features),
                "Stemming": stem_flag,
                "RMSE (Split)": rmse_split,
                "RMSE (CV)": rmse_cv,
                "RMSE_STD": rmse_std,
                "Train Time (s)": round(elapsed_time, 2)
            })

            print(f"Features: {features} --> Split RMSE: {rmse_split:.4f}, CV RMSE: {rmse_cv:.4f} ± {rmse_std:.4f} (train time: {elapsed_time:.2f}s)")

    results_df = pd.DataFrame(results).sort_values(by='RMSE (CV)')
    display(results_df)
    return results_df

def run_model_tuning_with_ttest(X, y):
    # Define hyperparameter spaces
    rf_param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(4, 15),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 10)
    }
    gb_param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 10)
    }

    print("\n--- Tuning Random Forest ---")
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=0),
        rf_param_dist,
        n_iter=10,
        cv=5,
        scoring='neg_root_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    start_rf = time.time()
    rf_search.fit(X, y)
    rf_time = time.time() - start_rf
    rf_best_rmse = -rf_search.best_score_
    rf_std = np.std(rf_search.cv_results_['mean_test_score'])

    print(f"Random Forest RMSE: {rf_best_rmse:.4f}")
    print(f"\nBest Random Forest CV RMSE: {rf_best_rmse:.4f} ± {rf_std:.4f}")
    print(f"Training time: {rf_time:.2f} seconds")
    print("\nBest Random Forest Parameters:")
    for k, v in rf_search.best_params_.items():
        print(f"  {k}: {v}")

    print("\n--- Tuning Gradient Boosting ---")
    gb_search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=0),
        gb_param_dist,
        n_iter=10,
        cv=5,
        scoring='neg_root_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    start_gb = time.time()
    gb_search.fit(X, y)
    gb_time = time.time() - start_gb
    gb_best_rmse = -gb_search.best_score_
    gb_std = np.std(gb_search.cv_results_['mean_test_score'])

    print(f"Gradient Boosting RMSE: {gb_best_rmse:.4f}")
    print(f"\nBest Gradient Boosting CV RMSE: {gb_best_rmse:.4f} ± {gb_std:.4f}")
    print(f"Training time: {gb_time:.2f} seconds")
    print("\nBest Gradient Boosting Parameters:")
    for k, v in gb_search.best_params_.items():
        print(f"  {k}: {v}")

    # Extract fold scores for t-test
    rf_rmse_folds = np.mean([
        -rf_search.cv_results_[f'split{i}_test_score']
        for i in range(5)
    ], axis=0)
    gb_rmse_folds = np.mean([
        -gb_search.cv_results_[f'split{i}_test_score']
        for i in range(5)
    ], axis=0)

    # Paired t-test
    print("\nPaired t-test result:")
    t_stat, p_val = ttest_rel(rf_rmse_folds, gb_rmse_folds)
    print(f"t-stat = {t_stat:.4f}, p-value = {p_val:.4f}")
    if p_val < 0.05:
        print("The difference is statistically significant.")
    else:
        print("The difference is not statistically significant.")

    return rf_search, gb_search



def generate_submission_file(raw_data, df_attr, num_train, best_features, model_params):
    print("Generating final test predictions with best feature combination...")
    final_submission = evaluate_on_test_set(raw_data, df_attr, best_features, num_train, stem=True, model_params=model_params)
    final_submission['relevance'] = final_submission['relevance'].clip(1.0, 3.0).round(2)
    final_submission.to_csv("submission.csv", index=False)
    print("Saved predictions to submission.csv.")

def main():
    """
    Main pipeline for training, evaluating, and predicting product search relevance.
    """
    warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

    # === Load and merge data ===
    df_train = pd.read_csv('home-depot-product-search-relevance/train.csv', encoding="ISO-8859-1")
    df_test = pd.read_csv('home-depot-product-search-relevance/test.csv', encoding="ISO-8859-1")
    df_attr = pd.read_csv('home-depot-product-search-relevance/attributes.csv', encoding="ISO-8859-1")
    df_pro_desc = pd.read_csv('home-depot-product-search-relevance/product_descriptions.csv')

    num_train = df_train.shape[0]  # Number of training samples
    raw_data = pd.concat((df_train, df_test), axis=0, ignore_index=True)  # Combine train and test sets
    raw_data = pd.merge(raw_data, df_pro_desc, how='left', on='product_uid')  # Merge product descriptions

    # === Explore data visually and statistically ===
    run_data_exploration(df_train, df_attr)

    # === Run simple baseline models ===
    run_baseline_evaluation(raw_data, df_attr, num_train)

    # === Compare different ML models using a small fixed feature set ===
    results_df = compare_models(raw_data, df_attr,
                                ['query_length', 'common_words', 'tfidf_similarity'],
                                num_train, stem=True)

    # === Tune models on a reasonable feature set before full evaluation ===
    df_features = build_feature_set(raw_data, df_attr,
                                    ['query_length', 'common_words', 'tfidf_similarity'], stem=True)
    df_train_features = df_features.iloc[:num_train]
    X = df_train_features.drop(columns=['id', 'relevance'])
    y = df_train_features['relevance']


    rf_search, gb_search = run_model_tuning_with_ttest(X, y)

    # === Extract and apply best parameters from Gradient Boosting ===
    best_params = gb_search.best_params_
    best_params['random_state'] = 42

    # === Evaluate all handpicked feature sets ===
    results_df = run_feature_set_evaluation(raw_data, df_attr, num_train,
                                            feature_sets, model_params=best_params)
    results_df.to_csv("feature_set_evaluation_results.csv", index=False)

    # === Visualize model complexity vs performance ===
    plot_overfitting_curve(results_df)

    # === Manually specified best performing feature set. ===
    best_feature_set = ['query_length', 'initial_term_match', 'jaccard', 'common_words', 
                       'color_match', 'fuzzy', 'bigram_overlap']
    row_match = results_df[results_df['Features'] == ', '.join(best_feature_set)]

    if not row_match.empty:
        best_row = row_match.iloc[0]
        print(f"\nBest Feature Set: {best_feature_set}")
        print(f"   CV RMSE: {best_row['RMSE (CV)']:.4f} ± {best_row['RMSE_STD']:.4f}")
    else:
        print("\nSpecified best feature set not found in results_df.")

    # === Plot feature importance using full feature space (not just best set) ===
    df_full = build_feature_set(raw_data, df_attr, all_features, stem=True)
    X_all = df_full.iloc[:num_train].drop(columns=['id', 'relevance'], errors='ignore')
    y_all = df_full.iloc[:num_train]['relevance']

    plot_feature_importance(GradientBoostingRegressor(**best_params),
                            X_all, y_all, X_all.columns.tolist(),
                            top_n=12, save_path="feature_importance_barplot.png")

    # === Generate and save test set predictions for submission ===
    generate_submission_file(raw_data, df_attr, num_train, best_feature_set, best_params)

    print(f"Final model: GradientBoostingRegressor | Random seed: {best_params['random_state']}, "
      f"Data shape: {raw_data.shape}, "
      f"Features used: {best_feature_set}")
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")