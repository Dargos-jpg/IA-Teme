# ============================================================================
# STUDIU DE CAZ COMPARATIV: REGULARIZARE IN MODELE POLINOMIALE
# Tema 3: Analiza impactului regularizarii asupra overfitting-ului
# Dataset: Concrete Compressive Strength
# ============================================================================

# ----------------------------------------------------------------------------
# IMPORT BIBLIOTECI
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# configurare grafice
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("STUDIU REGULARIZARE - MODELE POLINOMIALE")
print("="*70)

# ----------------------------------------------------------------------------
# INCARCARE DATE
# ----------------------------------------------------------------------------
# pentru Google Colab: Incarca fisierul cu butonul din stanga
# sau ruleaza: from google.colab import files; uploaded = files.upload()

df = pd.read_csv('concrete.csv')
df.columns = df.columns.str.strip()  # curatare nume coloane

print(f"\nDataset incarcat: {df.shape[0]} înregistrări, {df.shape[1]} coloane")
print(f"Primele 3 randuri:")
print(df.head(3))

# ----------------------------------------------------------------------------
# PREGATIRE DATE
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("Pregatire date")
print(f"{'='*70}")

# identificare features si target
target_col = [col for col in df.columns if 'strength' in col.lower()]
target = target_col[0] if target_col else df.columns[-1]
feature_cols = [col for col in df.columns if col != target]

X = df[feature_cols].values
y = df[target].values

print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Target: {target}")
print(f"Total: {X.shape[0]} observatii, {X.shape[1]} features")

# split train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# standardizare
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Standardizare aplicata (mean=0, std=1)")

# ----------------------------------------------------------------------------
# EXPERIMENT 1: OVERFITTING VS GRAD POLINOMIAL
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("Experiment 1: Identificare overfitting")
print(f"{'='*70}")

degrees = [1, 2, 3, 4, 5]
results_no_reg = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train_poly)))
    test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test_poly)))
    train_r2 = r2_score(y_train, model.predict(X_train_poly))
    test_r2 = r2_score(y_test, model.predict(X_test_poly))
    
    results_no_reg.append({
        'degree': degree,
        'n_features': X_train_poly.shape[1],
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'gap': test_rmse - train_rmse
    })
    
    print(f"Grad {degree} ({X_train_poly.shape[1]:4d} features): Train RMSE={train_rmse:7.2f} | "
          f"Test RMSE={test_rmse:7.2f} | Gap={test_rmse - train_rmse:7.2f}")

df_no_reg = pd.DataFrame(results_no_reg)

# ----------------------------------------------------------------------------
# EXPERIMENT 2: REGULARIZARE PE GRAD 3
# ----------------------------------------------------------------------------
optimal_degree = 3
print(f"\n{'='*70}")
print(f"Experiment 2: Regularizare (grad {optimal_degree})")
print(f"{'='*70}")

poly = PolynomialFeatures(degree=optimal_degree, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

print(f"Features polinomiale: {X_train_poly.shape[1]}")

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# Ridge
print(f"\n2.1. RIDGE (L2)")
ridge_results = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_poly, y_train)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, ridge.predict(X_train_poly)))
    test_rmse = np.sqrt(mean_squared_error(y_test, ridge.predict(X_test_poly)))
    
    ridge_results.append({
        'alpha': alpha,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'gap': test_rmse - train_rmse
    })
    print(f"Alpha={alpha:7.3f}: Train={train_rmse:6.2f} | Test={test_rmse:6.2f} | Gap={test_rmse - train_rmse:6.2f}")

df_ridge = pd.DataFrame(ridge_results)

# Lasso
print(f"\n2.2. LASSO (L1)")
lasso_results = []
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000, tol=1e-4)
    lasso.fit(X_train_poly, y_train)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, lasso.predict(X_train_poly)))
    test_rmse = np.sqrt(mean_squared_error(y_test, lasso.predict(X_test_poly)))
    n_nonzero = np.sum(np.abs(lasso.coef_) > 1e-5)
    
    lasso_results.append({
        'alpha': alpha,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'gap': test_rmse - train_rmse,
        'n_features': n_nonzero
    })
    print(f"Alpha={alpha:7.3f}: Train={train_rmse:6.2f} | Test={test_rmse:6.2f} | "
          f"Features={n_nonzero}/{X_train_poly.shape[1]}")

df_lasso = pd.DataFrame(lasso_results)

# Elastic Net
print(f"\n2.3. ELASTIC NET (L1+L2)")
elastic_results = []
for alpha in alphas:
    elastic = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000, tol=1e-4)
    elastic.fit(X_train_poly, y_train)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, elastic.predict(X_train_poly)))
    test_rmse = np.sqrt(mean_squared_error(y_test, elastic.predict(X_test_poly)))
    
    elastic_results.append({
        'alpha': alpha,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'gap': test_rmse - train_rmse
    })
    print(f"Alpha={alpha:7.3f}: Train={train_rmse:6.2f} | Test={test_rmse:6.2f} | Gap={test_rmse - train_rmse:6.2f}")

df_elastic = pd.DataFrame(elastic_results)

# ----------------------------------------------------------------------------
# GRAFICE
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("Generare grafice")
print(f"{'='*70}")

# Grafic 1: Overfitting demonstration
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# panoul 1: grade 1-3 liniar
df_low = df_no_reg[df_no_reg['degree'] <= 3]
axes[0, 0].plot(df_low['degree'], df_low['train_rmse'], 'o-', label='Train', linewidth=2.5, markersize=10)
axes[0, 0].plot(df_low['degree'], df_low['test_rmse'], 's-', label='Test', linewidth=2.5, markersize=10)
axes[0, 0].set_xlabel('Grad Polinomial')
axes[0, 0].set_ylabel('RMSE (MPa)')
axes[0, 0].set_title('Overfitting Progresiv (Grade 1-3)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# panoul 2: toate gradele log
axes[0, 1].semilogy(df_no_reg['degree'], df_no_reg['train_rmse'], 'o-', label='Train', linewidth=2.5, markersize=10)
axes[0, 1].semilogy(df_no_reg['degree'], df_no_reg['test_rmse'], 's-', label='Test', linewidth=2.5, markersize=10)
axes[0, 1].set_xlabel('Grad Polinomial')
axes[0, 1].set_ylabel('RMSE (MPa) - LOG')
axes[0, 1].set_title('Explozie Overfitting (Grade 1-5)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# panoul 3: R² pentru grade 1-3
axes[1, 0].plot(df_low['degree'], df_low['train_r2'], 'o-', label='Train R²', linewidth=2.5, markersize=10)
axes[1, 0].plot(df_low['degree'], df_low['test_r2'], 's-', label='Test R²', linewidth=2.5, markersize=10)
axes[1, 0].set_xlabel('Grad Polinomial')
axes[1, 0].set_ylabel('R² Score')
axes[1, 0].set_title('Acuratete vs Complexitate')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# panoul 4: gap
colors = ['green' if g < 1 else 'orange' if g < 5 else 'red' for g in df_no_reg['gap']]
axes[1, 1].bar(df_no_reg['degree'], df_no_reg['gap'], color=colors, alpha=0.7)
axes[1, 1].set_xlabel('Grad Polinomial')
axes[1, 1].set_ylabel('Gap (MPa)')
axes[1, 1].set_title('Masura Overfitting-ului')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('fig1_overfitting.png', dpi=300)
plt.show()

print("Grafic 1 salvat: fig1_overfitting.png")

# Grafic 2: Comparatie regularizare
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RMSE
axes[0].plot(np.log10(df_ridge['alpha']), df_ridge['test_rmse'], 'o-', label='Ridge', linewidth=2)
axes[0].plot(np.log10(df_lasso['alpha']), df_lasso['test_rmse'], 's-', label='Lasso', linewidth=2)
axes[0].plot(np.log10(df_elastic['alpha']), df_elastic['test_rmse'], '^-', label='Elastic Net', linewidth=2)
axes[0].set_xlabel('log₁₀(Alpha)')
axes[0].set_ylabel('Test RMSE (MPa)')
axes[0].set_title('Performanta pe Test')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gap
axes[1].plot(np.log10(df_ridge['alpha']), df_ridge['gap'], 'o-', label='Ridge', linewidth=2)
axes[1].plot(np.log10(df_lasso['alpha']), df_lasso['gap'], 's-', label='Lasso', linewidth=2)
axes[1].plot(np.log10(df_elastic['alpha']), df_elastic['gap'], '^-', label='Elastic Net', linewidth=2)
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[1].set_xlabel('log₁₀(Alpha)')
axes[1].set_ylabel('Gap (MPa)')
axes[1].set_title('Reducerea Overfitting-ului')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig2_regularization.png', dpi=300)
plt.show()

print("Grafic 2 salvat: fig2_regularization.png")

# Grafic 3: Feature selection Lasso
plt.figure(figsize=(10, 6))
plt.plot(np.log10(df_lasso['alpha']), df_lasso['n_features'], 'o-', linewidth=3, markersize=10, color='crimson')
plt.axhline(y=len(feature_cols), color='green', linestyle='--', linewidth=2, label=f'Features originale ({len(feature_cols)})')
plt.xlabel('log₁₀(Alpha)')
plt.ylabel('Features Active')
plt.title('Feature Selection în Lasso')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig3_lasso_features.png', dpi=300)
plt.show()

print("Grafic 3 salvat: fig3_lasso_features.png")

# ----------------------------------------------------------------------------
# CROSS-VALIDATION PENTRU ALPHA OPTIM
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("Cross-Validation pentru alpha optim")
print(f"{'='*70}")

alphas_cv = np.logspace(-4, 3, 50)

# Ridge CV
ridge_cv = RidgeCV(alphas=alphas_cv, scoring='neg_mean_squared_error', cv=5)
ridge_cv.fit(X_train_poly, y_train)
best_ridge_alpha = ridge_cv.alpha_
print(f"Ridge alpha optim: {best_ridge_alpha:.4f}")

# Lasso CV
lasso_cv = LassoCV(alphas=alphas_cv, cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(X_train_poly, y_train)
best_lasso_alpha = lasso_cv.alpha_
print(f"Lasso alpha optim: {best_lasso_alpha:.4f}")

# Elastic Net CV
elastic_cv = ElasticNetCV(alphas=alphas_cv, l1_ratio=0.5, cv=5, max_iter=10000, random_state=42)
elastic_cv.fit(X_train_poly, y_train)
best_elastic_alpha = elastic_cv.alpha_
print(f"Elastic Net alpha optim: {best_elastic_alpha:.4f}")

# ----------------------------------------------------------------------------
# EVALUARE FINALA
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("Evaluare finala pe test set")
print(f"{'='*70}")

models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=best_ridge_alpha),
    'Lasso': Lasso(alpha=best_lasso_alpha, max_iter=10000),
    'Elastic Net': ElasticNet(alpha=best_elastic_alpha, l1_ratio=0.5, max_iter=10000)
}

results = []

for name, model in models.items():
    model.fit(X_train_poly, y_train)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train_poly)))
    test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test_poly)))
    train_r2 = r2_score(y_train, model.predict(X_train_poly))
    test_r2 = r2_score(y_test, model.predict(X_test_poly))
    
    results.append({
        'Model': name,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Gap': test_rmse - train_rmse
    })
    
    print(f"\n{name}:")
    print(f"  Train RMSE: {train_rmse:6.2f} | Test RMSE: {test_rmse:6.2f}")
    print(f"  Train R²: {train_r2:7.4f} | Test R²: {test_r2:7.4f}")
    print(f"  Gap: {test_rmse - train_rmse:6.2f}")

df_results = pd.DataFrame(results)

# Grafic 4: Comparatie finala
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

x_pos = np.arange(len(df_results))
width = 0.35

# RMSE
axes[0, 0].bar(x_pos - width/2, df_results['Train RMSE'], width, label='Train', alpha=0.8)
axes[0, 0].bar(x_pos + width/2, df_results['Test RMSE'], width, label='Test', alpha=0.8)
axes[0, 0].set_ylabel('RMSE (MPa)')
axes[0, 0].set_title('Root Mean Squared Error')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(df_results['Model'], rotation=45, ha='right')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')

# R²
axes[0, 1].bar(x_pos - width/2, df_results['Train R²'], width, label='Train', alpha=0.8)
axes[0, 1].bar(x_pos + width/2, df_results['Test R²'], width, label='Test', alpha=0.8)
axes[0, 1].set_ylabel('R² Score')
axes[0, 1].set_title('R² Score')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(df_results['Model'], rotation=45, ha='right')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# MAE (calculat acum)
train_mae = [mean_absolute_error(y_train, m.predict(X_train_poly)) for m in models.values()]
test_mae = [mean_absolute_error(y_test, m.predict(X_test_poly)) for m in models.values()]
axes[1, 0].bar(x_pos - width/2, train_mae, width, label='Train', alpha=0.8)
axes[1, 0].bar(x_pos + width/2, test_mae, width, label='Test', alpha=0.8)
axes[1, 0].set_ylabel('MAE (MPa)')
axes[1, 0].set_title('Mean Absolute Error')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(df_results['Model'], rotation=45, ha='right')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Gap
axes[1, 1].bar(x_pos, df_results['Gap'], alpha=0.8, color='coral')
axes[1, 1].axhline(y=0, color='black', linestyle='--')
axes[1, 1].set_ylabel('Gap (MPa)')
axes[1, 1].set_title('Indicator Overfitting')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(df_results['Model'], rotation=45, ha='right')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('fig4_final_comparison.png', dpi=300)
plt.show()

print("\nGrafic 4 salvat: fig4_final_comparison.png")

# Salvare rezultate
df_results.to_csv('tabel_rezultate.csv', index=False)
print("\nTabel salvat: tabel_rezultate.csv")

# ----------------------------------------------------------------------------
# SUMAR
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("SUMAR REZULTATE")
print(f"{'='*70}")

best_model = df_results.loc[df_results['Test RMSE'].idxmin(), 'Model']
best_rmse = df_results['Test RMSE'].min()
best_r2 = df_results.loc[df_results['Test RMSE'].idxmin(), 'Test R²']

print(f"\nCel mai bun model: {best_model}")
print(f"  Test RMSE: {best_rmse:.2f} MPa")
print(f"  Test R²: {best_r2:.4f}")

print(f"\nOverfitting maxim detectat:")
print(f"  Grad 5: Gap = {df_no_reg.iloc[-1]['gap']:.2f} MPa")

print(f"\nFisiere generate:")
print("  - fig1_overfitting.png")
print("  - fig2_regularization.png")
print("  - fig3_lasso_features.png")
print("  - fig4_final_comparison.png")
print("  - tabel_rezultate.csv")

print(f"\n{'='*70}")
print("EXECUTIE FINALIZATA")
print(f"{'='*70}")