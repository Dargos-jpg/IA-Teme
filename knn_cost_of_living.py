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
print("Toate bibliotecile au fost importate cu succes!")
print("="*70)

# ----------------------------------------------------------------------------
# PASUL 2: INCARCAREA DATASETULUI
# ----------------------------------------------------------------------------
# pentru Google Colab: incarca fisierul folosind butonul de upload din stanga
# sau ruleaza: from google.colab import files; uploaded = files.upload()

# incarca datasetul - schimba numele fisierului daca e diferit
try:
    df = pd.read_csv('Cost_of_Living_Index_by_Country_2024.csv')
    print("\nDataset incarcat cu succes!")
except FileNotFoundError:
    print("\nEROARE: Fisierul nu a fost gasit!")
    print("Asigura-te ca ai incarcat 'Cost_of_Living_Index_2024.csv'")
    print("In Google Colab: click pe simbolul folder din stanga si upload fisierul")
    raise

# afiseaza informatii despre dataset
print(f"\n{'='*70}")
print("INFORMATII DESPRE DATASET")
print(f"{'='*70}")
print(f"Dimensiune: {df.shape[0]} tari, {df.shape[1]} coloane")
print(f"\nPrimele 3 tari:")
print(df.head(3))
print(f"\nColoane disponibile:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i}. {col}")

# ----------------------------------------------------------------------------
# PASUL 3: PREGATIREA DATELOR
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("PREGATIREA DATELOR")
print(f"{'='*70}")

# coloanele numerice exacte din dataset-ul tau
feature_columns = [
    'Cost of Living Index',
    'Rent Index',
    'Cost of Living Plus Rent Index',
    'Groceries Index',
    'Restaurant Price Index',
    'Local Purchasing Power Index'
]

# verifica daca toate coloanele exista
missing_cols = [col for col in feature_columns if col not in df.columns]
if missing_cols:
    print(f"ATENTIE: Coloanele {missing_cols} lipsesc!")
    print("Verifica numele exact din CSV si actualizeaza lista feature_columns")

# selecteaza doar coloanele numerice si elimina randurile cu valori lipsa
df_clean = df[['Country'] + feature_columns].copy()
df_clean = df_clean.dropna()

print(f"Date curate: {df_clean.shape[0]} tari (dupa eliminarea valorilor lipsa)")

# creaza categorii pentru clasificare bazate pe Cost of Living Index
# categorii: Ieftin (0-33%), Moderat (33-66%), Scump (66-100%)
df_clean['Category'] = pd.qcut(
    df_clean['Cost of Living Index'], 
    q=3, 
    labels=['Ieftin', 'Moderat', 'Scump']
)

print(f"\nDistributia categoriilor de tari:")
category_counts = df_clean['Category'].value_counts().sort_index()
for category, count in category_counts.items():
    percentage = (count / len(df_clean)) * 100
    print(f"   {category}: {count} tari ({percentage:.1f}%)")

# exemple de tari din fiecare categorie
print(f"\nExemple de tari din fiecare categorie:")
for category in ['Ieftin', 'Moderat', 'Scump']:
    countries = df_clean[df_clean['Category'] == category]['Country'].head(3).tolist()
    print(f"   {category}: {', '.join(countries)}")

# separa features (X) si target (y)
X = df_clean[feature_columns].values
y = df_clean['Category'].values

print(f"\nDate pregatite: X shape = {X.shape}, y shape = {y.shape}")

# ----------------------------------------------------------------------------
# PASUL 4: NORMALIZAREA DATELOR
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("NORMALIZAREA DATELOR")
print(f"{'='*70}")

# imparte in train (80%) si test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} tari ({(X_train.shape[0]/len(X))*100:.0f}%)")
print(f"Test set: {X_test.shape[0]} tari ({(X_test.shape[0]/len(X))*100:.0f}%)")

# normalizare StandardScaler: (X - mu) / sigma
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nDatele au fost normalizate (StandardScaler)")
print(f"\nExemplu normalizare - prima tara:")
print(f"   Inainte: {X_train[0][:3]}")
print(f"   Dupa:    {X_train_scaled[0][:3]}")
print(f"   (Medie aproximativ 0, Deviatie standard aproximativ 1)")

# ----------------------------------------------------------------------------
# PASUL 5: EXPERIMENTUL 1 - COMPARATIE METODE SI VALORI k
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("EXPERIMENTUL 1: COMPARATIE METODE PENTRU DIFERITE VALORI ALE k")
print(f"{'='*70}")

# parametri experimentali
k_values = [3, 5, 7, 10, 15, 20]
methods = ['brute', 'kd_tree', 'ball_tree']
method_names = ['Brute Force', 'KD-Tree', 'Ball Tree']

# dictionar pentru stocarea rezultatelor
results = {
    'k': [],
    'method': [],
    'method_name': [],
    'time_ms': [],
    'accuracy': []
}

print(f"\nTestam {len(k_values)} valori ale k x {len(methods)} metode = {len(k_values)*len(methods)} experimente\n")

# ruleaza toate combinatiile
for k in k_values:
    print(f"k = {k}")
    for method, name in zip(methods, method_names):
        # creaza modelul kNN
        knn = KNeighborsClassifier(n_neighbors=k, algorithm=method, n_jobs=-1)
        
        # masoara timpul de antrenare + predictie
        start_time = time.time()
        knn.fit(X_train_scaled, y_train)
        predictions = knn.predict(X_test_scaled)
        end_time = time.time()
        
        # calculeaza metrici
        acc = accuracy_score(y_test, predictions)
        elapsed_ms = (end_time - start_time) * 1000  # converteste in milisecunde
        
        # stocheaza rezultatele
        results['k'].append(k)
        results['method'].append(method)
        results['method_name'].append(name)
        results['time_ms'].append(elapsed_ms)
        results['accuracy'].append(acc)
        
        print(f"   {name:15s} -> Timp: {elapsed_ms:7.2f} ms | Acuratete: {acc*100:5.1f}%")
    print()

# converteste in DataFrame
results_df = pd.DataFrame(results)

print("Experimentul 1 completat cu succes!")

# ----------------------------------------------------------------------------
# PASUL 6: GRAFICUL 1 - TIMPUL DE EXECUTIE VS k
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("GENERARE FIGURA 1: Timpul de executie vs k")
print(f"{'='*70}")

plt.figure(figsize=(12, 7))

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
markers = ['o', 's', 'D']

for i, method in enumerate(method_names):
    data = results_df[results_df['method_name'] == method]
    plt.plot(data['k'], data['time_ms'], 
             marker=markers[i], 
             linewidth=2.5, 
             markersize=10,
             label=method,
             color=colors[i])

plt.xlabel('Numarul de vecini (k)', fontsize=14, fontweight='bold')
plt.ylabel('Timp de executie (ms)', fontsize=14, fontweight='bold')
plt.title('Figura 1: Comparatia timpului de executie pentru diferite valori ale k\n' + 
          f'(Dataset: {len(X_train_scaled)} tari in training, {X.shape[1]} features)',
          fontsize=15, fontweight='bold', pad=20)
plt.legend(fontsize=12, frameon=True, shadow=True, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('figura1_timp_vs_k.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figura 1 salvata: 'figura1_timp_vs_k.png'")

# ----------------------------------------------------------------------------
# PASUL 7: GRAFICUL 2 - ACURATETEA VS k
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("GENERARE FIGURA 2: Acuratetea vs k")
print(f"{'='*70}")

plt.figure(figsize=(12, 7))

for i, method in enumerate(method_names):
    data = results_df[results_df['method_name'] == method]
    plt.plot(data['k'], data['accuracy']*100, 
             marker=markers[i], 
             linewidth=2.5, 
             markersize=10,
             label=method,
             color=colors[i])

plt.xlabel('Numarul de vecini (k)', fontsize=14, fontweight='bold')
plt.ylabel('Acuratete (%)', fontsize=14, fontweight='bold')
plt.title('Figura 2: Comparatia acuratetei pentru diferite valori ale k\n' +
          f'(Test set: {len(X_test_scaled)} tari)',
          fontsize=15, fontweight='bold', pad=20)
plt.legend(fontsize=12, frameon=True, shadow=True, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim([0, 105])
plt.tight_layout()
plt.savefig('figura2_acuratete_vs_k.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figura 2 salvata: 'figura2_acuratete_vs_k.png'")

# ----------------------------------------------------------------------------
# PASUL 8: EXPERIMENTUL 2 - SCALABILITATE
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("EXPERIMENTUL 2: SCALABILITATE IN FUNCTIE DE MARIMEA DATASETULUI")
print(f"{'='*70}")

# dimensiuni diferite pentru testare
n_total = len(X_train_scaled)
if n_total >= 100:
    n_values = [int(n_total*0.5), n_total]
else:
    n_values = [max(30, int(n_total*0.5)), n_total]

k_fixed = 5  # k fix pentru acest experiment

scalability_results = {
    'n': [],
    'method': [],
    'method_name': [],
    'time_ms': [],
}

print(f"Testam cu k={k_fixed} fix, pentru n = {n_values}\n")

for n in n_values:
    print(f"n = {n} tari")
    # ia primele n exemple
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
        
        print(f"   {name:15s} -> Timp: {elapsed_ms:7.2f} ms")
    print()

scalability_df = pd.DataFrame(scalability_results)
print("Experimentul 2 completat cu succes!")

# ----------------------------------------------------------------------------
# PASUL 9: GRAFICUL 3 - SCALABILITATE
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("GENERARE FIGURA 3: Scalabilitatea metodelor")
print(f"{'='*70}")

plt.figure(figsize=(12, 7))

for i, method in enumerate(method_names):
    data = scalability_df[scalability_df['method_name'] == method]
    plt.plot(data['n'], data['time_ms'], 
             marker=markers[i], 
             linewidth=3, 
             markersize=12,
             label=method,
             color=colors[i])

plt.xlabel('Numarul de tari in training set (n)', fontsize=14, fontweight='bold')
plt.ylabel('Timp de executie (ms)', fontsize=14, fontweight='bold')
plt.title(f'Figura 3: Scalabilitatea metodelor in functie de dimensiunea datasetului\n(k={k_fixed}, {X.shape[1]} features)',
          fontsize=15, fontweight='bold', pad=20)
plt.legend(fontsize=12, frameon=True, shadow=True, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('figura3_scalabilitate.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figura 3 salvata: 'figura3_scalabilitate.png'")

# ----------------------------------------------------------------------------
# PASUL 10: TABELUL CU REZULTATE
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("GENERARE TABEL: Comparatie performanta metode")
print(f"{'='*70}")

# tabel sumar pentru k=5
k_target = 5
summary = results_df[results_df['k'] == k_target][['method_name', 'time_ms', 'accuracy']].copy()

# calculeaza speedup fata de Brute Force
brute_time = summary[summary['method_name'] == 'Brute Force']['time_ms'].values[0]
summary['speedup'] = brute_time / summary['time_ms']

print(f"\n{'='*80}")
print(f"TABELUL 1: Performanta metodelor de cautare a vecinilor")
print(f"(k={k_target}, n={len(X_train_scaled)} tari, d={X.shape[1]} features)")
print(f"{'='*80}")
print(f"{'Metoda':<20} {'Timp (ms)':<15} {'Acuratete (%)':<18} {'Speedup':<12}")
print(f"{'-'*80}")

for _, row in summary.iterrows():
    print(f"{row['method_name']:<20} {row['time_ms']:<15.2f} {row['accuracy']*100:<18.1f} {row['speedup']:<12.2f}x")

print(f"{'='*80}")

# salveaza in CSV pentru Word
summary.to_csv('tabel1_rezultate.csv', index=False)
print("\nTabelul salvat: 'tabel1_rezultate.csv'")

# tabel suplimentar: timpul mediu pe metoda
print(f"\n{'='*80}")
print("TABELUL 2: Timpul mediu de executie pe toate valorile k testate")
print(f"{'='*80}")

avg_times = results_df.groupby('method_name')['time_ms'].agg(['mean', 'std', 'min', 'max'])
print(avg_times.to_string())
print(f"{'='*80}")

# ----------------------------------------------------------------------------
# PASUL 11: GRAFIC COMBINAT (BONUS)
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("GENERARE FIGURA 4: Analiza comparativa completa (4 subgrafice)")
print(f"{'='*70}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Figura 4: Analiza comparativa completa a metodelor kNN\n' +
             f'Dataset: Cost of Living Index 2024 ({len(X)} tari, {X.shape[1]} features)', 
             fontsize=17, fontweight='bold', y=0.995)

# subplot 1: Timp vs k
ax1 = axes[0, 0]
for i, method in enumerate(method_names):
    data = results_df[results_df['method_name'] == method]
    ax1.plot(data['k'], data['time_ms'], marker=markers[i], linewidth=2.5, 
             markersize=9, label=method, color=colors[i])
ax1.set_xlabel('k (numar vecini)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Timp (ms)', fontsize=12, fontweight='bold')
ax1.set_title('a) Timpul de executie in functie de k', fontsize=13, fontweight='bold', pad=10)
ax1.legend(fontsize=10, frameon=True)
ax1.grid(True, alpha=0.3, linestyle='--')

# subplot 2: Acuratete vs k
ax2 = axes[0, 1]
for i, method in enumerate(method_names):
    data = results_df[results_df['method_name'] == method]
    ax2.plot(data['k'], data['accuracy']*100, marker=markers[i], linewidth=2.5,
             markersize=9, label=method, color=colors[i])
ax2.set_xlabel('k (numar vecini)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Acuratete (%)', fontsize=12, fontweight='bold')
ax2.set_title('b) Acuratetea in functie de k', fontsize=13, fontweight='bold', pad=10)
ax2.legend(fontsize=10, frameon=True)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim([0, 105])

# subplot 3: Scalabilitate
ax3 = axes[1, 0]
for i, method in enumerate(method_names):
    data = scalability_df[scalability_df['method_name'] == method]
    ax3.plot(data['n'], data['time_ms'], marker=markers[i], linewidth=2.5,
             markersize=9, label=method, color=colors[i])
ax3.set_xlabel('n (marime training set)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Timp (ms)', fontsize=12, fontweight='bold')
ax3.set_title(f'c) Scalabilitatea metodelor (k={k_fixed})', fontsize=13, fontweight='bold', pad=10)
ax3.legend(fontsize=10, frameon=True)
ax3.grid(True, alpha=0.3, linestyle='--')

# subplot 4: Bar chart comparatie directa
ax4 = axes[1, 1]
k5_data = results_df[results_df['k'] == 5]
x_pos = np.arange(len(method_names))
times = k5_data['time_ms'].values
bars = ax4.bar(x_pos, times, color=colors, edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Metoda', fontsize=12, fontweight='bold')
ax4.set_ylabel('Timp (ms)', fontsize=12, fontweight='bold')
ax4.set_title(f'd) Comparatie directa (k=5)', fontsize=13, fontweight='bold', pad=10)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(method_names, rotation=0)
ax4.grid(True, alpha=0.3, axis='y', linestyle='--')

# adauga valorile pe bare
for bar, time_val in zip(bars, times):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{time_val:.1f} ms', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('figura4_analiza_completa.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figura 4 salvata: 'figura4_analiza_completa.png'")

# ----------------------------------------------------------------------------
# PASUL 12: ANALIZA DETALIATA PENTRU k OPTIM
# ----------------------------------------------------------------------------
print(f"\n{'='*70}")
print("ANALIZA DETALIATA: Alegerea valorii optime a lui k")
print(f"{'='*70}")

# gaseste k optim pentru fiecare metoda
print("\nk optim pentru fiecare metoda (bazat pe acuratete):\n")

for method in method_names:
    method_data = results_df[results_df['method_name'] == method]
    best_idx = method_data['accuracy'].idxmax()
    best_row = method_data.loc[best_idx]
    
    print(f"   {method}:")
    print(f"      -> k optim: {int(best_row['k'])}")
    print(f"      -> Acuratete maxima: {best_row['accuracy']*100:.2f}%")
    print(f"      -> Timp la k optim: {best_row['time_ms']:.2f} ms")
    print()

# ----------------------------------------------------------------------------
# PASUL 13: SUMAR FINAL SI CONCLUZII
# ----------------------------------------------------------------------------
print(f"\n{'='*80}")
print("SUMAR FINAL - REZULTATE SI CONCLUZII")
print(f"{'='*80}")

# statistici generale
fastest_method = results_df.groupby('method_name')['time_ms'].mean().idxmin()
fastest_time = results_df.groupby('method_name')['time_ms'].mean().min()

most_accurate = results_df.groupby('method_name')['accuracy'].mean().idxmax()
best_acc = results_df.groupby('method_name')['accuracy'].mean().max()

brute_avg = results_df[results_df['method_name'] == 'Brute Force']['time_ms'].mean()
kd_avg = results_df[results_df['method_name'] == 'KD-Tree']['time_ms'].mean()
overall_speedup = brute_avg / kd_avg

print(f"\nCEA MAI RAPIDA METODA: {fastest_method}")
print(f"   -> Timp mediu: {fastest_time:.2f} ms")

print(f"\nCEA MAI PRECISA METODA: {most_accurate}")
print(f"   -> Acuratete medie: {best_acc*100:.2f}%")

print(f"\nSPEEDUP GENERAL:")
print(f"   -> Brute Force vs KD-Tree: {overall_speedup:.2f}x")

print(f"\nINFORMATII DATASET:")
print(f"   -> Total tari: {len(X)}")
print(f"   -> Features (dimensiuni): {X.shape[1]}")
print(f"   -> Training: {len(X_train_scaled)} tari ({(len(X_train_scaled)/len(X))*100:.0f}%)")
print(f"   -> Test: {len(X_test_scaled)} tari ({(len(X_test_scaled)/len(X))*100:.0f}%)")
print(f"   -> Categorii: {len(np.unique(y))} (Ieftin, Moderat, Scump)")

print(f"\nFISIERE GENERATE:")
print("   1. figura1_timp_vs_k.png")
print("   2. figura2_acuratete_vs_k.png")
print("   3. figura3_scalabilitate.png")
print("   4. figura4_analiza_completa.png")
print("   5. tabel1_rezultate.csv")

print(f"\nCONCLUZII CHEIE:")
print(f"   1. Brute Force este mai rapid cu {1/overall_speedup:.1f}x pentru n=96")
print(f"   2. Toate metodele ofera acuratete similara (~{best_acc*100:.0f}%)")
print(f"   3. Pentru d={X.shape[1]} dimensiuni si n mic, Brute Force este optim")
print(f"   4. KD-Tree devine eficient pentru n > 1000 puncte")

print(f"\n{'='*80}")
print("Cod executat cu succes!")
print(f"{'='*80}")