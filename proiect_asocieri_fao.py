"""
Proiect: Analiza Algoritmilor de Asociere (Apriori vs FP-Growth vs FPMax)
Dataset: FAO GIFT Bangladesh 2017-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, fpmax, association_rules
import time
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# incarcare date
subjects_df = pd.read_csv('subject_user.csv')
consumption_df = pd.read_csv('consumption_user.csv')

print(f"Incarcat {subjects_df.shape[0]} subiecti si {consumption_df.shape[0]} inregistrari consum")

# statistici rapide despre populatie
print(f"\nSubiecti: {subjects_df['SUBJECT'].nunique()}")
print(f"Gen - Male: {len(subjects_df[subjects_df['SEX']==1])}, Female: {len(subjects_df[subjects_df['SEX']==2])}")
print(f"Varsta medie: {subjects_df['AGE_YEAR'].mean():.1f} ani")
print(f"Rural: {len(subjects_df[subjects_df['AREA_TYPE']==1])}, Urban: {len(subjects_df[subjects_df['AREA_TYPE']==2])}")

# top regiuni
print("\nTop 5 regiuni:")
for region, count in subjects_df['ADM1_NAME'].value_counts().head(5).items():
    print(f"  {region}: {count}")

# prelucrare alimente
consumption_with_names = consumption_df[consumption_df['INGREDIENT_ENG'].notna()].copy()
food_freq = consumption_with_names['INGREDIENT_ENG'].value_counts()

print(f"\nTotal alimente unice: {len(food_freq)}")
print("\nTop 10 alimente (cu tot cu sare/apa):")
for i, (food, count) in enumerate(food_freq.head(10).items(), 1):
    print(f"{i}. {food[:45]} - {count}")

# filtrare sare/apa/zahar
items_remove = ['water', 'salt', 'sugar', 'tea']
consumption_filtered = consumption_with_names[
    ~consumption_with_names['INGREDIENT_ENG'].str.lower().str.strip().isin(items_remove) &
    ~consumption_with_names['INGREDIENT_ENG'].str.lower().str.contains('water|salt|sugar', case=False, na=False, regex=True)
].copy()

print(f"\nDupa filtrare: {len(consumption_filtered)} inregistrari ({len(consumption_with_names)-len(consumption_filtered)} eliminate)")

food_freq_filtered = consumption_filtered['INGREDIENT_ENG'].value_counts()
print(f"Alimente ramase: {len(food_freq_filtered)}")

print("\nTop 10 dupa filtrare:")
for i, (food, count) in enumerate(food_freq_filtered.head(10).items(), 1):
    print(f"{i}. {food[:45]} - {count}")

# creare tranzactii
transactions = consumption_filtered.groupby('SUBJECT')['INGREDIENT_ENG'].apply(
    lambda x: list(set(x.dropna()))
).values.tolist()
transactions = [t for t in transactions if len(t) > 0]

print(f"\nTranzactii: {len(transactions)}")
trans_lens = [len(t) for t in transactions]
print(f"Media alimente/subiect: {np.mean(trans_lens):.2f}, Mediana: {np.median(trans_lens):.0f}")

# one-hot encoding
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(f"Matrice: {df.shape[0]} x {df.shape[1]}")

# comparatie apriori vs fpgrowth la support 0.2
min_sup = 0.2
print(f"\n--- Testare algoritmi la support={min_sup} ---")

t1 = time.time()
itemsets_apr = apriori(df, min_support=min_sup, use_colnames=True)
time_apr = time.time() - t1
print(f"Apriori: {time_apr:.4f}s, {len(itemsets_apr)} itemsets")

t1 = time.time()
itemsets_fpg = fpgrowth(df, min_support=min_sup, use_colnames=True)
time_fpg = time.time() - t1
print(f"FP-Growth: {time_fpg:.4f}s, {len(itemsets_fpg)} itemsets")

if time_apr < time_fpg:
    print(f"Castigator: Apriori (cu {((time_fpg-time_apr)/time_fpg)*100:.1f}% mai rapid)")
else:
    print(f"Castigator: FP-Growth (cu {((time_apr-time_fpg)/time_apr)*100:.1f}% mai rapid)")

# simulare pe mai multe niveluri
supports = [0.2, 0.15, 0.1, 0.05]
results = []

print("\n--- Simulare scalabilitate ---")
for sup in supports:
    print(f"\nSupport {sup}:")
    
    t = time.time()
    apr_items = apriori(df, min_support=sup, use_colnames=True)
    apr_t = time.time() - t
    
    t = time.time()
    fpg_items = fpgrowth(df, min_support=sup, use_colnames=True)
    fpg_t = time.time() - t
    
    t = time.time()
    fpmax_items = fpmax(df, min_support=sup, use_colnames=True)
    fpmax_t = time.time() - t
    
    print(f"  Apriori: {apr_t:.4f}s ({len(apr_items)} items)")
    print(f"  FP-Growth: {fpg_t:.4f}s ({len(fpg_items)} items)")
    print(f"  FPMax: {fpmax_t:.4f}s ({len(fpmax_items)} items)")
    
    results.append({
        'support': sup,
        'apr_time': apr_t,
        'fpg_time': fpg_t,
        'fpmax_time': fpmax_t,
        'apr_items': len(apr_items),
        'fpg_items': len(fpg_items),
        'fpmax_items': len(fpmax_items)
    })
    
    # salvez itemsets maximale pentru support 0.05
    if sup == 0.05:
        maximal_items = fpmax_items.copy()

perf_df = pd.DataFrame(results)

# afisare reducere redundanta
print("\nReducere redundanta prin FPMax:")
for _, row in perf_df.iterrows():
    if row['fpmax_items'] > 0:
        red = ((row['fpg_items'] - row['fpmax_items']) / row['fpg_items']) * 100
        print(f"  Support {row['support']:.2f}: {row['fpg_items']} -> {row['fpmax_items']} (-{red:.1f}%)")

# vizualizare 1: scalabilitate log
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(perf_df))
w = 0.25

ax.bar(x - w, perf_df['apr_time'], w, label='Apriori', color='#3498db', alpha=0.8)
ax.bar(x, perf_df['fpg_time'], w, label='FP-Growth', color='#e74c3c', alpha=0.8)
ax.bar(x + w, perf_df['fpmax_time'], w, label='FPMax', color='#2ecc71', alpha=0.8)

ax.set_yscale('log')
ax.set_xlabel('Minimum Support')
ax.set_ylabel('Timp (s, scara log)')
ax.set_title('Scalabilitate Algoritmi: Apriori vs FP-Growth vs FPMax')
ax.set_xticks(x)
ax.set_xticklabels([f'{s:.2f}' for s in perf_df['support']])
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('scalabilitate_log.png', dpi=300)
plt.close()
print("\nGrafic scalabilitate salvat")

# vizualizare 2: demografie
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# gen
sex_dist = subjects_df['SEX'].value_counts().sort_index()
axes[0].pie(sex_dist.values, labels=['Male', 'Female'], autopct='%1.1f%%',
            colors=['#3498db', '#e74c3c'], startangle=90)
axes[0].set_title('Distributie Gen')

# zona
area_dist = subjects_df['AREA_TYPE'].value_counts().sort_index()
axes[1].pie(area_dist.values, labels=['Rural', 'Urban'], autopct='%1.1f%%',
            colors=['#2ecc71', '#f39c12'], startangle=90)
axes[1].set_title('Distributie Zona')

# regiuni
region_dist = subjects_df['ADM1_NAME'].value_counts().head(5)
axes[2].pie(region_dist.values, labels=region_dist.index, autopct='%1.1f%%', startangle=90)
axes[2].set_title('Top 5 Regiuni')

plt.tight_layout()
plt.savefig('demografie_pie.png', dpi=300)
plt.close()
print("Grafic demografie salvat")

# vizualizare 3: redundanta
if len(maximal_items) > 0:
    fig, ax = plt.subplots(figsize=(9, 6))
    
    row_005 = perf_df[perf_df['support'] == 0.05].iloc[0]
    categories = ['FP-Growth\n(Complete)', 'FPMax\n(Maximale)']
    values = [row_005['fpg_items'], row_005['fpmax_items']]
    
    ax.bar(categories, values, color=['#e74c3c', '#2ecc71'], width=0.5)
    ax.set_ylabel('Numar Itemsets')
    ax.set_title('Reducere Redundanta prin FPMax (support=0.05)')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('redundanta_itemsets.png', dpi=300)
    plt.close()
    print("Grafic redundanta salvat")

# analiza itemsets maximale
if len(maximal_items) > 0:
    maximal_sorted = maximal_items.sort_values('support', ascending=False)
    
    print(f"\n--- Itemsets maximale (support 0.05) ---")
    print(f"Total: {len(maximal_sorted)}")
    
    complete = perf_df[perf_df['support'] == 0.05]['fpg_items'].values[0]
    red = ((complete - len(maximal_sorted)) / complete) * 100
    print(f"Reducere: {complete} -> {len(maximal_sorted)} (-{red:.1f}%)")
    
    print("\nTop 10 itemsets maximale:")
    for i, (idx, row) in enumerate(maximal_sorted.head(10).iterrows(), 1):
        items = list(row['itemsets'])
        if len(items) <= 3:
            items_str = ', '.join(items)
        else:
            items_str = ', '.join(items[:3]) + f' + {len(items)-3} altele'
        print(f"{i}. [{len(items)} alimente] {items_str[:60]} (support={row['support']:.4f})")
    
    item_sizes = [len(items) for items in maximal_sorted['itemsets']]
    print(f"\nStatistici: medie {np.mean(item_sizes):.1f}, mediana {np.median(item_sizes):.0f}, interval {min(item_sizes)}-{max(item_sizes)}")
    
    maximal_sorted.to_csv('maximal_itemsets.csv', index=False)

# generare reguli
min_conf = 0.6
print(f"\n--- Generare reguli (confidence >= {min_conf}) ---")

if len(itemsets_fpg) > 0:
    rules = association_rules(itemsets_fpg, metric="confidence", min_threshold=min_conf)
    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
    
    print(f"Reguli gasite: {len(rules)}")
    print(f"Lift mediu: {rules['lift'].mean():.2f}, Confidence medie: {rules['confidence'].mean():.2f}")
    
    print("\nTop 10 reguli:")
    for i, row in rules.head(10).iterrows():
        ant = ', '.join(list(row['antecedents']))[:50]
        cons = ', '.join(list(row['consequents']))[:50]
        print(f"{i+1}. {ant} -> {cons}")
        print(f"   Support={row['support']:.4f}, Conf={row['confidence']:.4f}, Lift={row['lift']:.4f}")
    
    rules.to_csv('association_rules.csv', index=False)
    
    # vizualizare 4: distributie reguli
    fig, ax = plt.subplots(figsize=(11, 7))
    
    scatter = ax.scatter(rules['support'], rules['confidence'], 
                        c=rules['lift'], s=60, alpha=0.6, cmap='viridis')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Lift')
    
    ax.set_xlabel('Support')
    ax.set_ylabel('Confidence')
    ax.set_title(f'Distributie Reguli (n={len(rules):,})')
    ax.axhline(y=min_conf, color='red', linestyle='--', alpha=0.5, label=f'Min Conf={min_conf}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('distributie_reguli.png', dpi=300)
    plt.close()
    print("Grafic distributie reguli salvat")

# vizualizare 5: top alimente
fig, ax = plt.subplots(figsize=(13, 7))
top_foods = food_freq_filtered.head(15)

ax.barh(range(len(top_foods)), top_foods.values, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top_foods))))
ax.set_yticks(range(len(top_foods)))
ax.set_yticklabels([f[:40] for f in top_foods.index])
ax.set_xlabel('Frecventa')
ax.set_ylabel('Aliment')
ax.set_title('Top 15 Alimente (fara sare/apa/zahar)')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('top_15_foods_frequency.png', dpi=300)
plt.close()
print("Grafic top alimente salvat")

# sumar final
print("\n--- SUMAR ---")
print(f"Subiecti: {len(transactions)}")
print(f"Inregistrari: {len(consumption_df):,} (filtrate: {len(consumption_filtered):,})")
print(f"Alimente unice: {df.shape[1]}")
print(f"Apriori vs FP-Growth la sup=0.2: {time_apr:.4f}s vs {time_fpg:.4f}s")
if len(maximal_items) > 0:
    print(f"Itemsets maximale: {len(maximal_items)} (reducere {red:.1f}%)")
if len(rules) > 0:
    print(f"Reguli: {len(rules)}, Lift max: {rules.iloc[0]['lift']:.4f}")

print("\nFisiere generate:")
print("  - scalabilitate_log.png")
print("  - demografie_pie.png")
print("  - redundanta_itemsets.png")
print("  - distributie_reguli.png")
print("  - top_15_foods_frequency.png")
print("  - association_rules.csv")
print("  - maximal_itemsets.csv")