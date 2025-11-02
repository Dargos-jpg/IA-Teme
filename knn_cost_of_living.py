# ============================================================================
# STUDIU DE CAZ COMPARATIV - ALGORITMUL k-NEAREST NEIGHBORS (kNN)
# Tema 3: Cautarea eficienta a vecinilor apropiati
# Dataset: Cost of Living Index 2024
# ============================================================================

# ----------------------------------------------------------------------------
# PASUL 1: INSTALARE SI IMPORT BIBLIOTECI
# ----------------------------------------------------------------------------
# daca rulezi local si nu ai bibliotecile, decomenteaza linia de jos:
# !pip install pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')

# setari pentru grafice profesionale
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

print("="*70)
print("Biblioteci importate cu succes")
print("="*70)

# ----------------------------------------------------------------------------
# PASUL 2: INCARCAREA DATASETULUI
# ----------------------------------------------------------------------------
# pentru Google Colab: incarca fisierul folosind butonul de upload din stanga
# sau ruleaza: from google.colab import files; uploaded = files.upload()

# incarca datasetul
try:
    df = pd.read_csv('Cost_of_Living_Index_by_Country_2024.csv')
    print("\nDataset incarcat")
except FileNotFoundError:
    print("\nEroare: fisierul nu a fost gasit")
    print("Incarca fisierul CSV in Google Colab")
    raise

# afiseaza informatii despre dataset
print(f"\n{'='*70}")
print("Informatii dataset")
print(f"{'='*70}")
print(f"Dimensiune: {df.shape[0]} tari, {df.shape[1]} coloane")
print(f"\nPrimele 3 randuri:")
print(df.head(3))
print(f"\nColoane:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i}. {col}")

# ----------------------------------------------------------------------------
# PASUL 3: PREGATIREA DATELOR
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("Pregatirea datelor")
print(f"{'='*70}")

# coloanele folosite pentru clasificare
feature_columns = [
    'Cost of Living Index',
    'Rent Index',
    'Cost of Living Plus Rent Index',
    'Groceries Index',
    'Restaurant Price Index',
    'Local Purchasing Power Index'
]

# verifica daca coloanele exista
missing_cols = [col for col in feature_columns if col not in df.columns]
if missing_cols:
    print(f"Atentie: coloanele {missing_cols} lipsesc")

# curata datele
df_clean = df[['Country'] + feature_columns].copy()
df_clean = df_clean.dropna()

print(f"Date curatate: {df_clean.shape[0]} tari")

# creaza categorii pentru clasificare
df_clean['Category'] = pd.qcut(
    df_clean['Cost of Living Index'], 
    q=3, 
    labels=['Ieftin', 'Moderat', 'Scump']
)

print(f"\nDistributia categoriilor:")
category_counts = df_clean['Category'].value_counts().sort_index()
for category, count in category_counts.items():
    percentage = (count / len(df_clean)) * 100
    print(f"   {category}: {count} tari ({percentage:.1f}%)")

print(f"\nExemple tari per categorie:")
for category in ['Ieftin', 'Moderat', 'Scump']:
    countries = df_clean[df_clean['Category'] == category]['Country'].head(3).tolist()
    print(f"   {category}: {', '.join(countries)}")

# separa features si target
X = df_clean[feature_columns].values
y = df_clean['Category'].values

print(f"\nDimensiuni: X = {X.shape}, y = {y.shape}")

# ----------------------------------------------------------------------------
# PASUL 4: NORMALIZAREA DATELOR
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("Normalizarea datelor")
print(f"{'='*70}")

# split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training: {X_train.shape[0]} tari ({(X_train.shape[0]/len(X))*100:.0f}%)")
print(f"Test: {X_test.shape[0]} tari ({(X_test.shape[0]/len(X))*100:.0f}%)")

# normalizare StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nNormalizare aplicata")
print(f"Exemplu - prima observatie:")
print(f"   Original: {X_train[0][:3]}")
print(f"   Normalizat: {X_train_scaled[0][:3]}")

# ----------------------------------------------------------------------------
# PASUL 5: EXPERIMENTUL 1 - COMPARATIE METODE
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("Experiment 1: Comparatie metode pentru diferite k")
print(f"{'='*70}")

k_values = [3, 5, 7, 10, 15, 20]
methods = ['brute', 'kd_tree', 'ball_tree']
method_names = ['Brute Force', 'KD-Tree', 'Ball Tree']

results = {
    'k': [],
    'method': [],
    'method_name': [],
    'time_ms': [],
    'accuracy': []
}

print(f"\nTestare: {len(k_values)} valori k x {len(methods)} metode\n")

for k in k_values:
    print(f"k = {k}")
    for method, name in zip(methods, method_names):
        knn = KNeighborsClassifier(n_neighbors=k, algorithm=method, n_jobs=-1)
        
        start_time = time.time()
        knn.fit(X_train_scaled, y_train)
        predictions = knn.predict(X_test_scaled)
        end_time = time.time()
        
        acc = accuracy_score(y_test, predictions)
        elapsed_ms = (end_time - start_time) * 1000
        
        results['k'].append(k)
        results['method'].append(method)
        results['method_name'].append(name)
        results['time_ms'].append(elapsed_ms)
        results['accuracy'].append(acc)
        
        print(f"   {name:15s} -> {elapsed_ms:7.2f} ms | {acc*100:5.1f}%")
    print()

results_df = pd.DataFrame(results)
print("Experiment 1 finalizat")

# ----------------------------------------------------------------------------
# PASUL 6: GENERARE GRAFICE
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("Generare grafice")
print(f"{'='*70}")

# grafic 1: timp vs k
plt.figure(figsize=(12, 7))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
markers = ['o', 's', 'D']

for i, method in enumerate(method_names):
    data = results_df[results_df['method_name'] == method]
    plt.plot(data['k'], data['time_ms'], 
             marker=markers[i], linewidth=2.5, markersize=10,
             label=method, color=colors[i])

plt.xlabel('k (numar vecini)', fontsize=14, fontweight='bold')
plt.ylabel('Timp executie (ms)', fontsize=14, fontweight='bold')
plt.title('Figura 1: Timp executie vs k\n' + 
          f'Dataset: {len(X_train_scaled)} tari, {X.shape[1]} features',
          fontsize=15, fontweight='bold', pad=20)
plt.legend(fontsize=12, frameon=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('figura1_timp_vs_k.png', dpi=300, bbox_inches='tight')
plt.show()

# grafic 2: acuratete vs k
plt.figure(figsize=(12, 7))

for i, method in enumerate(method_names):
    data = results_df[results_df['method_name'] == method]
    plt.plot(data['k'], data['accuracy']*100, 
             marker=markers[i], linewidth=2.5, markersize=10,
             label=method, color=colors[i])

plt.xlabel('k (numar vecini)', fontsize=14, fontweight='bold')
plt.ylabel('Acuratete (%)', fontsize=14, fontweight='bold')
plt.title('Figura 2: Acuratete vs k\n' +
          f'Test set: {len(X_test_scaled)} tari',
          fontsize=15, fontweight='bold', pad=20)
plt.legend(fontsize=12, frameon=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim([0, 105])
plt.tight_layout()
plt.savefig('figura2_acuratete_vs_k.png', dpi=300, bbox_inches='tight')
plt.show()

print("Grafice salvate: figura1, figura2")

# ----------------------------------------------------------------------------
# PASUL 7: EXPERIMENTUL 2 - SCALABILITATE
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("Experiment 2: Scalabilitate")
print(f"{'='*70}")

n_total = len(X_train_scaled)
if n_total >= 100:
    n_values = [int(n_total*0.5), n_total]
else:
    n_values = [max(30, int(n_total*0.5)), n_total]

k_fixed = 5

scalability_results = {
    'n': [],
    'method': [],
    'method_name': [],
    'time_ms': [],
}

print(f"Testare cu k={k_fixed}, n = {n_values}\n")

for n in n_values:
    print(f"n = {n}")
    X_subset = X_train_scaled[:n]
    y_subset = y_train[:n]
    
    for method, name in zip(methods, method_names):
        knn = KNeighborsClassifier(n_neighbors=k_fixed, algorithm=method, n_jobs=-1)
        
        start_time = time.time()
        knn.fit(X_subset, y_subset)
        predictions = knn.predict(X_test_scaled)
        end_time = time.time()
        
        elapsed_ms = (end_time - start_time) * 1000
        
        scalability_results['n'].append(n)
        scalability_results['method'].append(method)
        scalability_results['method_name'].append(name)
        scalability_results['time_ms'].append(elapsed_ms)
        
        print(f"   {name:15s} -> {elapsed_ms:7.2f} ms")
    print()

scalability_df = pd.DataFrame(scalability_results)
print("Experiment 2 finalizat")

# grafic 3: scalabilitate
print(f"\n{'='*70}")
print("Generare grafic scalabilitate")
print(f"{'='*70}")

plt.figure(figsize=(12, 7))

for i, method in enumerate(method_names):
    data = scalability_df[scalability_df['method_name'] == method]
    plt.plot(data['n'], data['time_ms'], 
             marker=markers[i], linewidth=3, markersize=12,
             label=method, color=colors[i])

plt.xlabel('n (marime training set)', fontsize=14, fontweight='bold')
plt.ylabel('Timp executie (ms)', fontsize=14, fontweight='bold')
plt.title(f'Figura 3: Scalabilitate (k={k_fixed}, {X.shape[1]} features)',
          fontsize=15, fontweight='bold', pad=20)
plt.legend(fontsize=12, frameon=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('figura3_scalabilitate.png', dpi=300, bbox_inches='tight')
plt.show()

print("Grafic salvat: figura3")

# ----------------------------------------------------------------------------
# PASUL 8: TABEL REZULTATE
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("Generare tabel rezultate")
print(f"{'='*70}")

k_target = 5
summary = results_df[results_df['k'] == k_target][['method_name', 'time_ms', 'accuracy']].copy()

brute_time = summary[summary['method_name'] == 'Brute Force']['time_ms'].values[0]
summary['speedup'] = brute_time / summary['time_ms']

print(f"\n{'='*80}")
print(f"Tabel: Performanta metodelor (k={k_target})")
print(f"{'='*80}")
print(f"{'Metoda':<20} {'Timp (ms)':<15} {'Acuratete (%)':<18} {'Speedup':<12}")
print(f"{'-'*80}")

for _, row in summary.iterrows():
    print(f"{row['method_name']:<20} {row['time_ms']:<15.2f} {row['accuracy']*100:<18.1f} {row['speedup']:<12.2f}x")

print(f"{'='*80}")

summary.to_csv('tabel1_rezultate.csv', index=False)
print("\nTabel salvat: tabel1_rezultate.csv")

avg_times = results_df.groupby('method_name')['time_ms'].agg(['mean', 'std', 'min', 'max'])
print(f"\nTimp mediu pe metoda:")
print(avg_times)

# ----------------------------------------------------------------------------
# PASUL 9: GRAFIC COMBINAT
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("Generare grafic combinat (4 subgrafice)")
print(f"{'='*70}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Figura 4: Analiza comparativa metode kNN\n' +
             f'Dataset: Cost of Living 2024 ({len(X)} tari, {X.shape[1]} features)', 
             fontsize=17, fontweight='bold', y=0.995)

# subplot 1
ax1 = axes[0, 0]
for i, method in enumerate(method_names):
    data = results_df[results_df['method_name'] == method]
    ax1.plot(data['k'], data['time_ms'], marker=markers[i], linewidth=2.5, 
             markersize=9, label=method, color=colors[i])
ax1.set_xlabel('k', fontsize=12, fontweight='bold')
ax1.set_ylabel('Timp (ms)', fontsize=12, fontweight='bold')
ax1.set_title('a) Timp vs k', fontsize=13, fontweight='bold', pad=10)
ax1.legend(fontsize=10, frameon=True)
ax1.grid(True, alpha=0.3, linestyle='--')

# subplot 2
ax2 = axes[0, 1]
for i, method in enumerate(method_names):
    data = results_df[results_df['method_name'] == method]
    ax2.plot(data['k'], data['accuracy']*100, marker=markers[i], linewidth=2.5,
             markersize=9, label=method, color=colors[i])
ax2.set_xlabel('k', fontsize=12, fontweight='bold')
ax2.set_ylabel('Acuratete (%)', fontsize=12, fontweight='bold')
ax2.set_title('b) Acuratete vs k', fontsize=13, fontweight='bold', pad=10)
ax2.legend(fontsize=10, frameon=True)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim([0, 105])

# subplot 3
ax3 = axes[1, 0]
for i, method in enumerate(method_names):
    data = scalability_df[scalability_df['method_name'] == method]
    ax3.plot(data['n'], data['time_ms'], marker=markers[i], linewidth=2.5,
             markersize=9, label=method, color=colors[i])
ax3.set_xlabel('n', fontsize=12, fontweight='bold')
ax3.set_ylabel('Timp (ms)', fontsize=12, fontweight='bold')
ax3.set_title(f'c) Scalabilitate (k={k_fixed})', fontsize=13, fontweight='bold', pad=10)
ax3.legend(fontsize=10, frameon=True)
ax3.grid(True, alpha=0.3, linestyle='--')

# subplot 4
ax4 = axes[1, 1]
k5_data = results_df[results_df['k'] == 5]
x_pos = np.arange(len(method_names))
times = k5_data['time_ms'].values
bars = ax4.bar(x_pos, times, color=colors, edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Metoda', fontsize=12, fontweight='bold')
ax4.set_ylabel('Timp (ms)', fontsize=12, fontweight='bold')
ax4.set_title(f'd) Comparatie (k=5)', fontsize=13, fontweight='bold', pad=10)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(method_names, rotation=0)
ax4.grid(True, alpha=0.3, axis='y', linestyle='--')

for bar, time_val in zip(bars, times):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{time_val:.1f} ms', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('figura4_analiza_completa.png', dpi=300, bbox_inches='tight')
plt.show()

print("Grafic salvat: figura4")

# ----------------------------------------------------------------------------
# PASUL 10: ANALIZA k OPTIM
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("Analiza: k optim")
print(f"{'='*70}")

print("\nk optim per metoda:\n")

for method in method_names:
    method_data = results_df[results_df['method_name'] == method]
    best_idx = method_data['accuracy'].idxmax()
    best_row = method_data.loc[best_idx]
    
    print(f"{method}:")
    print(f"   k optim: {int(best_row['k'])}")
    print(f"   Acuratete: {best_row['accuracy']*100:.2f}%")
    print(f"   Timp: {best_row['time_ms']:.2f} ms\n")

# ----------------------------------------------------------------------------
# PASUL 11: SUMAR
# ----------------------------------------------------------------------------
print(f"{'='*80}")
print("Sumar")
print(f"{'='*80}")

fastest_method = results_df.groupby('method_name')['time_ms'].mean().idxmin()
fastest_time = results_df.groupby('method_name')['time_ms'].mean().min()
most_accurate = results_df.groupby('method_name')['accuracy'].mean().idxmax()
best_acc = results_df.groupby('method_name')['accuracy'].mean().max()

print(f"\nMetoda cea mai rapida: {fastest_method} ({fastest_time:.2f} ms)")
print(f"Metoda cea mai precisa: {most_accurate} ({best_acc*100:.2f}%)")

print(f"\nInformatii dataset:")
print(f"   Total: {len(X)} tari")
print(f"   Features: {X.shape[1]}")
print(f"   Training: {len(X_train_scaled)} ({(len(X_train_scaled)/len(X))*100:.0f}%)")
print(f"   Test: {len(X_test_scaled)} ({(len(X_test_scaled)/len(X))*100:.0f}%)")

print(f"\nFisiere generate:")
print("   - figura1_timp_vs_k.png")
print("   - figura2_acuratete_vs_k.png")
print("   - figura3_scalabilitate.png")
print("   - figura4_analiza_completa.png")
print("   - tabel1_rezultate.csv")

print(f"\n{'='*80}")
print("Executie finalizata")
print(f"{'='*80}")