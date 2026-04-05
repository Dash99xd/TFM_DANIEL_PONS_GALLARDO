"""
TFM - Análisis de Datos de Hepatitis C
======================================

Código completo de análisis de datos para el Trabajo Fin de Máster.
Este repositorio contiene todas las implementaciones de:
- Análisis exploratorio de datos
- Visualizaciones
- Modelos de Machine Learning
- Optimización de hiperparámetros
- Evaluación de modelos

Dataset: hepatitisC.csv (UCI Machine Learning Repository)
Autor: [Tu nombre]
Repositorio: [URL del repositorio GitHub/GitLab]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            confusion_matrix, classification_report, roc_auc_score, 
                            roc_curve, matthews_corrcoef)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# =============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# =============================================================================


def cargar_datos(ruta_csv='hepatitisC.csv'):
    """
    Carga el dataset de hepatitis C desde CSV.
    
    Returns:
        pd.DataFrame: Datos cargados
    """
    df = pd.read_csv(ruta_csv)
    print(f"Dataset cargado: {df.shape[0]} muestras, {df.shape[1]} variables")
    print(f"\nColumnas: {list(df.columns)}")
    return df


def preparar_datos(df):
    """
    Prepara los datos para el análisis.
    
    Args:
        df: DataFrame con datos brutos
        
    Returns:
        X, y, scaler: Features, target y scaler ajustado
    """
    # Eliminar columna de índice si existe
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Codificar variable objetivo
    le = LabelEncoder()
    df['Category_encoded'] = le.fit_transform(df['Category'])
    
    # Crear clasificación binaria: 0=Sano (0=Blood Donor), 1=Enfermo (resto)
    df['Binary_target'] = (df['Category'] != '0=Blood Donor').astype(int)
    
    # Seleccionar features numéricas
    features = ['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
    X = df[features].copy()
    y = df['Binary_target']
    
    # Imputar valores faltantes con la mediana
    X = X.fillna(X.median())
    
    # Estandarizar
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    print(f"\nPreparación completada:")
    print(f"  Features: {list(X.columns)}")
    print(f"  Muestras: {len(X)}")
    print(f"  Distribución clase: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X_scaled, y, scaler, le, df


# =============================================================================
# 2. ANÁLISIS EXPLORATORIO Y VISUALIZACIONES
# =============================================================================

def figura_1_distribucion_categorias(df, save_path='figura1_distribucion.png'):
    """
    Figura 1: Distribución de pacientes por categoría diagnóstica.
    """
    plt.figure(figsize=(10, 6))
    
    counts = df['Category'].value_counts()
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6']
    
    bars = plt.bar(range(len(counts)), counts.values, color=colors, edgecolor='black')
    plt.xticks(range(len(counts)), counts.index, rotation=45, ha='right')
    plt.ylabel('Número de Pacientes', fontsize=11)
    plt.xlabel('Categoría Diagnóstica', fontsize=11)
    plt.title('1. Distribución de Pacientes por Categoría Diagnóstica', fontsize=12, fontweight='bold')
    
    # Añadir valores encima de barras
    for i, (bar, val) in enumerate(zip(bars, counts.values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(val), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Figura 1 guardada: {save_path}")


def figura_2_distribucion_marcadores(df, save_path='figura2_marcadores.png'):
    """
    Figura 2: Distribución de marcadores hepáticos por categoría (boxplot).
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    marcadores = ['ALT', 'AST', 'ALP', 'GGT', 'BIL', 'CHE']
    colores_cat = {'0=Blood Donor': '#2ecc71', '0s=suspect Blood Donor': '#f1c40f',
                    '1=Hepatitis': '#e67e22', '2=Fibrosis': '#e74c3c', '3=Cirrhosis': '#9b59b6'}

    for i, marcador in enumerate(marcadores):
        ax = axes[i]
        sns.boxplot(x='Category', y=marcador, data=df, palette=colores_cat, ax=ax,
                    order=df['Category'].unique())
        ax.set_xlabel('')
        ax.set_ylabel(marcador, fontsize=10)
        ax.set_title(f'Distribución de {marcador}', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=8)

    plt.suptitle('2. Distribución de Marcadores Hepáticos por Categoría', 
                    fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Figura 2 guardada: {save_path}")


def figura_3_matriz_correlacion(X, save_path='figura3_correlacion.png'):
    """
    Figura 3: Matriz de correlación de Pearson.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    
    vars_corr = ['Age', 'ALB', 'ALP', 'AST', 'ALT', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
    corr_matrix = X[vars_corr].corr()
    
    # Máscara para triángulo superior
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, 
                cbar_kws={"shrink": .8, "label": "r de Pearson"},
                ax=ax, annot_kws={'size': 10, 'weight': 'bold'})
    
    ax.set_title('3. Correlaciones entre Variables Clínicas', 
                 fontsize=13, fontweight='bold', pad=15)
    
    labels = ['Edad\n(años)', 'Albúmina\n(g/L)', 'Fosfatasa Alcalina\n(U/L)', 
              'AST\n(U/L)', 'ALT\n(U/L)', 'Bilirrubina\n(μmol/L)', 
              'Colinesterasa\n(U/L)', 'Colesterol\n(mmol/L)', 'Creatinina\n(μmol/L)', 
              'GGT\n(U/L)', 'Proteínas Totales\n(g/L)']
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Figura 3 guardada: {save_path}")


def figura_4_pca(X, y, df, save_path='figura4_pca.png'):
    """
    Figura 4: Análisis de Componentes Principales (PCA) con 5 categorías diagnósticas.
    """
    # Aplicar PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Calcular varianza explicada
    var_exp = pca.explained_variance_ratio_
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Mapeo de categorías con colores
    categorias_map = {
        '0=Blood Donor': {'color': '#2ecc71', 'label': 'Donantes Sanos (n=526)'},
        '0s=suspect Blood Donor': {'color': '#f1c40f', 'label': 'Sospechosos (n=7)'},
        '1=Hepatitis': {'color': '#e67e22', 'label': 'Hepatitis Activa (n=20)'},
        '2=Fibrosis': {'color': '#e74c3c', 'label': 'Fibrosis (n=12)'},
        '3=Cirrhosis': {'color': '#9b59b6', 'label': 'Cirrosis (n=24)'}
    }
    
    # Plotear cada categoría
    for categoria, info in categorias_map.items():
        mask = df['Category'] == categoria
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  c=info['color'], label=info['label'],
                  alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)', fontsize=12)
    ax.set_title('4. Análisis de Componentes Principales (PCA)\nCon Categorías Diagnósticas', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Añadir varianza total explicada
    total_var = sum(var_exp) * 100
    ax.text(0.02, 0.98, f'Varianza explicada total: {total_var:.1f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Figura 4 guardada: {save_path}")
    print(f"  Varianza explicada PC1: {var_exp[0]*100:.1f}%")
    print(f"  Varianza explicada PC2: {var_exp[1]*100:.1f}%")


# =============================================================================
# 3. MODELOS DE MACHINE LEARNING
# =============================================================================

def entrenar_modelos_base(X_train, X_test, y_train, y_test):
    """
    Entrena y evalúa 5 modelos base.
    
    Returns:
        dict: Resultados de cada modelo
    """
    modelos = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    resultados = {}
    
    for nombre, modelo in modelos.items():
        print(f"\n{'='*50}")
        print(f"Entrenando: {nombre}")
        print('='*50)
        
        # Entrenar
        modelo.fit(X_train, y_train)
        
        # Predicciones
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]
        
        # Métricas
        resultados[nombre] = {
            'modelo': modelo,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'specificity': recall_score(y_test, y_pred, pos_label=0),  # Especificidad
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_prob),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        print(f"Accuracy:     {resultados[nombre]['accuracy']:.4f}")
        print(f"Precision:    {resultados[nombre]['precision']:.4f}")
        print(f"Recall:       {resultados[nombre]['recall']:.4f}")
        print(f"Specificity:  {resultados[nombre]['specificity']:.4f}")
        print(f"F1-Score:     {resultados[nombre]['f1']:.4f}")
        print(f"AUC-ROC:      {resultados[nombre]['auc']:.4f}")
        print(f"MCC:          {resultados[nombre]['mcc']:.4f}")
    
    return resultados


def aplicar_smote(X_train, y_train, k_neighbors=5):
    """
    Aplica SMOTE para balancear clases.
    """
    smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"\nSMOTE aplicado:")
    print(f"  Original: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Balanceado: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
    
    return X_resampled, y_resampled


def seleccion_caracteristicas_sfs(X, y, n_features=8):
    """
    Selección de características con SFS (Sequential Feature Selector).
    """
    modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    sfs = SequentialFeatureSelector(
        modelo_rf, 
        n_features_to_select=n_features,
        direction='forward',
        cv=5,
        scoring='roc_auc'
    )
    
    sfs.fit(X, y)
    
    features_selected = X.columns[sfs.get_support()].tolist()
    print(f"\nSFS - Características seleccionadas ({n_features}):")
    for i, feat in enumerate(features_selected, 1):
        print(f"  {i}. {feat}")
    
    X_selected = pd.DataFrame(sfs.transform(X), columns=features_selected)
    
    return X_selected, features_selected, sfs


# =============================================================================
# 4. OPTIMIZACIÓN DE HIPERPARÁMETROS
# =============================================================================

def optimizar_random_forest(X_train, y_train):
    """
    Grid Search CV para Random Forest.
    """
    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(
        rf, param_grid, 
        cv=StratifiedKFold(n_splits=5),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\n" + "="*60)
    print("OPTIMIZACIÓN RANDOM FOREST - MEJORES RESULTADOS")
    print("="*60)
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor AUC (CV): {grid_search.best_score_:.4f}")
    
    # Top 5 configuraciones
    results = pd.DataFrame(grid_search.cv_results_)
    results = results.sort_values('rank_test_score').head(5)
    
    print("\nTop 5 configuraciones:")
    for i, row in results.iterrows():
        print(f"  Rank {int(row['rank_test_score'])}: {row['params']} | AUC: {row['mean_test_score']:.4f} (+/- {row['std_test_score']*2:.4f})")
    
    return grid_search.best_estimator_, grid_search.cv_results_


def optimizar_gradient_boosting(X_train, y_train):
    """
    Grid Search CV para Gradient Boosting.
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2]
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    
    grid_search = GridSearchCV(
        gb, param_grid,
        cv=StratifiedKFold(n_splits=5),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\n" + "="*60)
    print("OPTIMIZACIÓN GRADIENT BOOSTING - MEJORES RESULTADOS")
    print("="*60)
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor AUC (CV): {grid_search.best_score_:.4f}")
    
    # Top 5 configuraciones
    results = pd.DataFrame(grid_search.cv_results_)
    results = results.sort_values('rank_test_score').head(5)
    
    print("\nTop 5 configuraciones:")
    for i, row in results.iterrows():
        print(f"  Rank {int(row['rank_test_score'])}: {row['params']} | AUC: {row['mean_test_score']:.4f} (+/- {row['std_test_score']*2:.4f})")
    
    return grid_search.best_estimator_, grid_search.cv_results_


def figura_comparacion_metricas(resultados, titulo='Comparación de Métricas', 
                                  save_path='figura5_comparacion.png'):
    """
    Figura 5: Comparación visual de métricas de rendimiento entre modelos.
    """
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    
    metricas = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc', 'mcc']
    titulos = ['Accuracy', 'Precision', 'Recall', 'Especificidad', 'F1-Score', 'AUC-ROC', 'MCC']
    
    modelos = list(resultados.keys())
    colores = plt.cm.Set3(np.linspace(0, 1, len(modelos)))
    
    for i, (metrica, tit) in enumerate(zip(metricas, titulos)):
        ax = axes[i]
        valores = [resultados[m][metrica] for m in modelos]
        
        bars = ax.barh(modelos, valores, color=colores, edgecolor='black', height=0.6)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Score', fontsize=10)
        ax.set_title(tit, fontsize=11, fontweight='bold')
        
        # Añadir valores
        for bar, val in zip(bars, valores):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=8)
    
    # Ocultar subplots vacíos
    for j in range(len(metricas), len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(f'5. {titulo}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Figura 5 guardada: {save_path}")


def figura_curvas_roc_detallada(resultados, y_test, titulo='Curvas ROC',
                                 save_path='figura6_roc.png'):
    """
    Figura 6: Curvas ROC comparativas de los cinco modelos (Zoom en valores altos).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    modelos = list(resultados.keys())
    colores = plt.cm.tab10(np.linspace(0, 1, len(modelos)))
    
    for i, (nombre, color, ax) in enumerate(zip(modelos, colores, axes)):
        y_prob = resultados[nombre]['y_prob']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = resultados[nombre]['auc']
        
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{nombre} (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random')
        ax.set_xlim([-0.05, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('1 - Especificidad', fontsize=9)
        ax.set_ylabel('Sensibilidad', fontsize=9)
        ax.set_title(nombre, fontsize=11, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Eliminar eje extra si existe
    if len(modelos) < 6:
        fig.delaxes(axes[-1])
    
    plt.suptitle(f'6. {titulo}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Figura 6 guardada: {save_path}")


def figura_smote_distribucion(y_train, y_train_smote, save_path='figuraA_smote.png'):
    """
    Figura A: Distribución de clases antes y después de aplicar SMOTE.
    """
    # Contar clases antes y después
    before_counts = np.bincount(y_train)
    after_counts = np.bincount(y_train_smote)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    labels = ['Sano', 'Enfermo']
    colors = ['#2ecc71', '#e74c3c']
    
    # Antes de SMOTE
    ax1 = axes[0]
    bars1 = ax1.bar(labels, before_counts, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Número de Muestras', fontsize=11)
    ax1.set_title('Antes de SMOTE', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(after_counts) * 1.1)
    
    # Añadir valores encima de las barras
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    total_before = sum(before_counts)
    ax1.text(0.5, 0.95, f'Total: {total_before} muestras\nRatio: {before_counts[0]}/{before_counts[1]}',
            transform=ax1.transAxes, fontsize=10, verticalalignment='top',
            ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Después de SMOTE
    ax2 = axes[1]
    bars2 = ax2.bar(labels, after_counts, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Número de Muestras', fontsize=11)
    ax2.set_title('Después de SMOTE', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(after_counts) * 1.1)
    
    # Añadir valores encima de las barras
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    total_after = sum(after_counts)
    ax2.text(0.5, 0.95, f'Total: {total_after} muestras\nRatio: {after_counts[0]}/{after_counts[1]} (Balanceado)',
            transform=ax2.transAxes, fontsize=10, verticalalignment='top',
            ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    fig.suptitle('Figura A. Distribución de clases antes y después de aplicar SMOTE\nEl conjunto de entrenamiento se balancea de ' +
                 f'{before_counts[0]} sanos / {before_counts[1]} enfermos a {after_counts[0]} de cada clase (n={total_after} total)',
                 fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Figura A guardada: {save_path}")


def figura_importancia_gini(modelo, feature_names, save_path='figura7_importancia.png'):
    """
    Figura 7: Importancia de variables según Gini (Random Forest sin SMOTE/SFS).
    """
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Barras horizontales
    y_pos = np.arange(len(feature_names))
    bars = ax.barh(y_pos, importancias[indices], color='darkgreen', 
                   edgecolor='black', height=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importancia de Gini', fontsize=11)
    ax.set_title('7. Importancia de Variables (Sin SMOTE/SFS)\nRandom Forest - Gini Criterion', 
                 fontsize=13, fontweight='bold')
    
    # Añadir porcentajes
    for i, (bar, idx) in enumerate(zip(bars, indices)):
        pct = importancias[idx] * 100
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
               f'{pct:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Figura 7 guardada: {save_path}")


def figura_comparacion_smote(resultados_sin, resultados_con, 
                              save_path='figura8_comparacion_smote.png'):
    """
    Figura 8: Comparación de rendimiento: Sin vs Con SMOTE/SFS.
    """
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    
    metricas = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc', 'mcc']
    titulos = ['Accuracy', 'Precision', 'Recall', 'Especificidad', 'F1-Score', 'AUC-ROC', 'MCC']
    
    modelos = list(resultados_sin.keys())
    x = np.arange(len(modelos))
    width = 0.35
    
    for i, (metrica, tit) in enumerate(zip(metricas, titulos)):
        ax = axes[i]
        
        vals_sin = [resultados_sin[m][metrica] for m in modelos]
        vals_con = [resultados_con[m][metrica] for m in modelos]
        
        bars1 = ax.bar(x - width/2, vals_sin, width, label='Sin SMOTE/SFS', 
                      color='lightcoral', edgecolor='black')
        bars2 = ax.bar(x + width/2, vals_con, width, label='Con SMOTE/SFS',
                      color='lightblue', edgecolor='black')
        
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title(tit, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m[:8] for m in modelos], rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
    
    # Ocultar subplots vacíos
    for j in range(len(metricas), len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('8. Comparación de Rendimiento: Sin vs Con SMOTE/SFS',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Figura 8 guardada: {save_path}")


def figura_roc_smote(resultados_sin, resultados_con, y_test,
                     save_path='figura9_roc_smote.png'):
    """
    Figura 9: Curvas ROC comparativas con y sin SMOTE/SFS.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colores = plt.cm.tab10(np.linspace(0, 1, len(resultados_sin)))
    
    # Sin SMOTE
    ax1 = axes[0]
    for nombre, color in zip(resultados_sin.keys(), colores):
        y_prob = resultados_sin[nombre]['y_prob']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = resultados_sin[nombre]['auc']
        ax1.plot(fpr, tpr, color=color, lw=2, label=f'{nombre} (AUC={auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set_xlim([-0.05, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_title('Without SMOTE/SFS', fontsize=12, fontweight='bold')
    ax1.set_xlabel('1 - Especificidad')
    ax1.set_ylabel('Sensibilidad')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Con SMOTE
    ax2 = axes[1]
    for nombre, color in zip(resultados_con.keys(), colores):
        y_prob = resultados_con[nombre]['y_prob']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = resultados_con[nombre]['auc']
        ax2.plot(fpr, tpr, color=color, lw=2, label=f'{nombre} (AUC={auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', lw=2)
    ax2.set_xlim([-0.05, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_title('With SMOTE/SFS', fontsize=12, fontweight='bold')
    ax2.set_xlabel('1 - Especificidad')
    ax2.set_ylabel('Sensibilidad')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('9. ROC Curves: SMOTE/SFS Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Figura 9 guardada: {save_path}")


def figura_importancia_sfs(modelo, feature_names, selected_features,
                           save_path='figura10_importancia_sfs.png'):
    """
    Figura 10: Importancia de variables seleccionadas por SFS (con SMOTE).
    """
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Solo las características seleccionadas
    bars = ax.bar(range(len(importancias)), importancias[indices],
                  color='teal', edgecolor='black')
    ax.set_xticks(range(len(importancias)))
    ax.set_xticklabels([feature_names[i] for i in indices], 
                      rotation=45, ha='right')
    ax.set_ylabel('Importancia (Gini)', fontsize=11)
    ax.set_title('10. Importancia de Variables Seleccionadas (SFS)\nCon SMOTE - 8 Variables',
                 fontsize=13, fontweight='bold')
    
    # Añadir valores
    for bar, idx in zip(bars, indices):
        pct = importancias[idx] * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Figura 10 guardada: {save_path}")


def figura_comparacion_final(resultados_sin, resultados_con,
                              save_path='figura11_comparacion_final.png'):
    """
    Figura 11: Comparación final entre enfoques (Sin vs Con SMOTE/SFS).
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calcular promedios por métrica
    metricas = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc', 'mcc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'Especificidad', 'F1-Score', 'AUC-ROC', 'MCC']
    
    promedios_sin = [np.mean([resultados_sin[m][met] for m in resultados_sin.keys()]) 
                     for met in metricas]
    promedios_con = [np.mean([resultados_con[m][met] for m in resultados_con.keys()])
                     for met in metricas]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, promedios_sin, width, label='Without SMOTE/SFS',
                  color='coral', edgecolor='black')
    bars2 = ax.bar(x + width/2, promedios_con, width, label='With SMOTE/SFS',
                  color='steelblue', edgecolor='black')
    
    ax.set_ylabel('Average Score', fontsize=12)
    ax.set_title('11. Final Comparison: Without vs With SMOTE/SFS\n(Average of 5 Models)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    # Añadir valores
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Figura 11 guardada: {save_path}")

def figura_comparacion_modelos(resultados, save_path='figura_comparacion.png'):
    """
    Visualización comparativa de métricas entre modelos.
    """
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    
    metricas = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc', 'mcc']
    titulos = ['Accuracy', 'Precision', 'Recall', 'Especificidad', 'F1-Score', 'AUC-ROC', 'MCC']
    
    modelos = list(resultados.keys())
    
    for i, (metrica, titulo) in enumerate(zip(metricas, titulos)):
        ax = axes[i]
        valores = [resultados[m][metrica] for m in modelos]
        
        bars = ax.bar(modelos, valores, color='steelblue', edgecolor='black')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title(titulo, fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Añadir valores
        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Ocultar subplots vacíos
    for j in range(len(metricas), len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('Comparison of Metrics Between Models', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓ Figura comparación guardada: {save_path}")


def figura_curvas_roc(resultados, y_test, save_path='figura_roc.png'):
    """
    Curvas ROC comparativas.
    """
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(resultados)))
    
    for i, (nombre, color) in enumerate(zip(resultados.keys(), colors)):
        y_prob = resultados[nombre]['y_prob']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = resultados[nombre]['auc']
        plt.plot(fpr, tpr, color=color, lw=2, 
                label=f'{nombre} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Especificidad)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensibilidad)', fontsize=11)
    plt.title('Curvas ROC Comparativas', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Figura ROC guardada: {save_path}")


def figura_importancia_variables(modelo, feature_names, save_path='figura_importancia.png'):
    """
    Importancia de variables (Random Forest).
    """
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(importancias)), importancias[indices], 
                   color='forestgreen', edgecolor='black')
    plt.xticks(range(len(importancias)), 
               [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.ylabel('Importancia (Gini)', fontsize=11)
    plt.xlabel('Variables', fontsize=11)
    plt.title('Importancia de Variables - Random Forest', fontsize=13, fontweight='bold')
    
    # Añadir porcentajes
    for bar, idx in zip(bars, indices):
        pct = importancias[idx] * 100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Figura importancia guardada: {save_path}")


# =============================================================================
# 5. GENERACIÓN DE TABLAS Y LOGS
# =============================================================================

def tabla_descripcion_variables(X_original):
    """
    Tabla: Descripción detallada de las variables predictoras del estudio.
    X_original: DataFrame con datos sin escalar (valores originales del dataset)
    """
    variables_info = {
        'Age': {'Name': 'Edad', 'Unit': 'años', 'Description': 'Edad del paciente'},
        'ALB': {'Name': 'Albúmina', 'Unit': 'g/dL', 'Description': 'Proteína sintetizada por el hígado'},
        'ALP': {'Name': 'Fosfatasa Alcalina', 'Unit': 'U/L', 'Description': 'Enzima elevada en enfermedad hepática biliar'},
        'ALT': {'Name': 'Alanina Aminotransferasa', 'Unit': 'U/L', 'Description': 'Enzima hepática específica'},
        'AST': {'Name': 'Aspartato Aminotransferasa', 'Unit': 'U/L', 'Description': 'Enzima hepática y muscular'},
        'BIL': {'Name': 'Bilirrubina Total', 'Unit': 'μmol/L', 'Description': 'Pigmento biliar, marcador de ictericia'},
        'CHE': {'Name': 'Colinesterasa', 'Unit': 'U/L', 'Description': 'Enzima relacionada con función hepática'},
        'CHOL': {'Name': 'Colesterol Total', 'Unit': 'mmol/L', 'Description': 'Lípido sintetizado por el hígado'},
        'CREA': {'Name': 'Creatinina', 'Unit': 'μmol/L', 'Description': 'Marcador de función renal'},
        'GGT': {'Name': 'Gamma Glutamil Transferasa', 'Unit': 'U/L', 'Description': 'Enzima hepatobiliar'},
        'PROT': {'Name': 'Proteínas Totales', 'Unit': 'g/dL', 'Description': 'Proteínas séricas totales'}
    }
    
    print("\n" + "="*150)
    print("TABLA: Descripción Detallada de las Variables Predictoras del Estudio")
    print("="*150)
    print(f"{'No.':<5} {'Variable':<10} {'Nombre':<25} {'Unidad':<12} {'Descripción':<40} {'Rango en Dataset':<20}")
    print("-"*150)
    
    for i, col in enumerate(X_original.columns, 1):
        if col in variables_info:
            info = variables_info[col]
            min_val = X_original[col].min()
            max_val = X_original[col].max()
            rango = f"{min_val:.2f} - {max_val:.2f}"
            print(f"{i:<5} {col:<10} {info['Name']:<25} {info['Unit']:<12} {info['Description']:<40} {rango:<20}")
    
    return variables_info


def tabla_estadisticas_completas(X_original):
    """
    Tabla: Estadísticas descriptivas completas de las variables numéricas.
    X_original: DataFrame con datos sin escalar (valores originales del dataset)
    """
    stats = X_original.describe().T
    stats = stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    
    print("\n" + "="*140)
    print("TABLA: Estadísticas Descriptivas Completas de las Variables Numéricas")
    print("="*140)
    print(f"{'Variable':<12} {'N':<8} {'Media':<12} {'Desv. Est.':<12} {'Mínimo':<12} {'Q1':<12} {'Mediana':<12} {'Q3':<12} {'Máximo':<12}")
    print("-"*140)
    
    for var, row in stats.iterrows():
        print(f"{var:<12} {int(row['count']):<8} {row['mean']:<12.4f} {row['std']:<12.4f} {row['min']:<12.4f} {row['25%']:<12.4f} {row['50%']:<12.4f} {row['75%']:<12.4f} {row['max']:<12.4f}")
    
    return stats


def tabla_caracteristicas_sfs_orden(features_selected):
    """
    Tabla: Características seleccionadas por SFS ordenadas por orden de selección.
    """
    # Categorías funcionales de los marcadores
    categorias_funcionales = {
        'Age': 'Datos Demográficos',
        'ALB': 'Síntesis Proteica',
        'ALP': 'Colestasis',
        'ALT': 'Daño Hepatocelular',
        'AST': 'Daño Hepatocelular',
        'BIL': 'Función Biliar',
        'CHE': 'Síntesis Proteica',
        'CHOL': 'Metabolismo Lipídico',
        'CREA': 'Función Renal',
        'GGT': 'Colestasis',
        'PROT': 'Síntesis Proteica'
    }
    
    print("\n" + "="*100)
    print("TABLA: Características Seleccionadas por SFS Ordenadas por Orden de Selección")
    print("="*100)
    print(f"{'Orden':<8} {'Característica':<20} {'Categoría Funcional':<60}")
    print("-"*100)
    
    for i, feat in enumerate(features_selected, 1):
        cat_func = categorias_funcionales.get(feat, 'N/A')
        print(f"{i:<8} {feat:<20} {cat_func:<60}")
    
    return features_selected


def tabla_grid_search_rf(cv_results):
    """
    Tabla: Top 5 configuraciones de Random Forest ordenadas por AUC promedio en validación cruzada (5-fold).
    """
    results_df = pd.DataFrame(cv_results)
    results_df = results_df.sort_values('rank_test_score')
    top5 = results_df[['rank_test_score', 'params', 'mean_test_score', 'std_test_score']].head(5).copy()
    
    print("\n" + "="*160)
    print("TABLA: Top 5 Configuraciones de Random Forest (Grid Search)")
    print("="*160)
    print(f"{'Ranking':<10} {'Configuración':<18} {'n_estimators':<16} {'max_depth':<14} {'min_samples_split':<18} {'AUC-ROC (CV)':<20}")
    print("-"*160)
    
    for idx, (i, row) in enumerate(top5.iterrows(), 1):
        params = row['params']
        n_est = params.get('n_estimators', 'N/A')
        max_d = params.get('max_depth', 'N/A')
        min_samp = params.get('min_samples_split', 'N/A')
        config = f"RF_{int(n_est)}_{int(max_d)}_{int(min_samp)}"
        auc_str = f"{row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}"
        print(f"{int(row['rank_test_score']):<10} {config:<18} {str(n_est):<16} {str(max_d):<14} {str(min_samp):<18} {auc_str:<20}")
    
    return top5


def tabla_grid_search_gb(cv_results):
    """
    Tabla: Top 5 configuraciones de Gradient Boosting ordenadas por AUC promedio en validación cruzada (5-fold).
    """
    results_df = pd.DataFrame(cv_results)
    results_df = results_df.sort_values('rank_test_score')
    top5 = results_df[['rank_test_score', 'params', 'mean_test_score', 'std_test_score']].head(5).copy()
    
    print("\n" + "="*165)
    print("TABLA: Top 5 Configuraciones de Gradient Boosting (Grid Search)")
    print("="*165)
    print(f"{'Ranking':<10} {'Configuración':<18} {'n_estimators':<16} {'max_depth':<14} {'learning_rate':<16} {'AUC-ROC (CV)':<20}")
    print("-"*165)
    
    for idx, (i, row) in enumerate(top5.iterrows(), 1):
        params = row['params']
        n_est = params.get('n_estimators', 'N/A')
        max_d = params.get('max_depth', 'N/A')
        lr = params.get('learning_rate', 'N/A')
        config = f"GB_{int(n_est)}_{int(max_d)}_{lr}"
        auc_str = f"{row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}"
        print(f"{int(row['rank_test_score']):<10} {config:<18} {str(n_est):<16} {str(max_d):<14} {str(lr):<16} {auc_str:<20}")
    
    return top5


def tabla_multiclase(df, X, y):
    """
    Tabla: Métricas de clasificación multiclase por categoría diagnóstica.
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    X_train, X_test, y_train_mc, y_test_mc = train_test_split(
        X, df['Category_encoded'], test_size=0.2, random_state=42, 
        stratify=df['Category_encoded']
    )
    
    modelo_mc = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_mc.fit(X_train, y_train_mc)
    y_pred_mc = modelo_mc.predict(X_test)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test_mc, y_pred_mc, average=None
    )
    
    categorias = ['Blood Donor', 'Suspect Blood Donor', 'Hepatitis', 'Fibrosis', 'Cirrhosis']
    accuracy_global = accuracy_score(y_test_mc, y_pred_mc)
    
    print("\n" + "="*100)
    print("TABLA: Métricas de Clasificación Multiclase por Categoría Diagnóstica")
    print("="*100)
    print(f"{'Categoría':<30} {'Precisión':<15} {'Sensibilidad':<15} {'F1-Score':<15} {'Muestras':<10}")
    print("-"*100)
    
    for cat, p, r, f, s in zip(categorias, precision, recall, f1, support):
        print(f"{cat:<30} {p:<15.4f} {r:<15.4f} {f:<15.4f} {int(s):<10}")
    
    print("-"*100)
    print(f"{'Accuracy Global':<30} {accuracy_global:<15.4f}")
    
    return categorias, precision, recall, f1, support, accuracy_global


def tabla_estadisticas_categorias(df):
    """
    Tabla A.1: Valores medios ± desviación estándar de marcadores hepáticos por categoría diagnóstica.
    """
    # Solo marcadores clave (6 en lugar de 11)
    marcadores_clave = ['Age', 'ALT', 'AST', 'GGT', 'BIL', 'ALB']
    categorias = df['Category'].unique()
    
    print("\n" + "="*150)
    print("TABLA A.1: Valores Medios ± Desviación Estándar de Marcadores Hepáticos por Categoría Diagnóstica")
    print("="*150)
    
    # Preparar datos por categoría
    results = []
    for cat in categorias:
        datos_cat = df[df['Category'] == cat]
        
        # Contar muestras (n)
        n_muestras = len(datos_cat)
        
        # Calcular medias y desviaciones
        row = {'Categoría': cat, 'n': n_muestras}
        for marcador in marcadores_clave:
            valores = datos_cat[marcador].dropna()
            if len(valores) > 0:
                media = valores.mean()
                std = valores.std()
                row[marcador] = f"{media:.1f} ± {std:.1f}"
            else:
                row[marcador] = "N/A"
        
        results.append(row)
    
    # Imprimir encabezado
    encabezado = f"{'Categoría':<25} {'n':<6}"
    for marc in marcadores_clave:
        encabezado += f" {marc:<18}"
    print(encabezado)
    print("-"*150)
    
    # Imprimir datos
    for row in results:
        linea = f"{row['Categoría']:<25} {row['n']:<6}"
        for marc in marcadores_clave:
            linea += f" {row[marc]:<18}"
        print(linea)
    
    return results, categorias


def tabla_matrices_confusion(resultados_base, resultados_smote, y_test):
    """
    Tabla A.2: Matriz de confusión detallada para cada modelo (TN=Verdaderos Negativos, FP=Falsos Positivos, FN=Falsos Negativos, TP=Verdaderos Positivos).
    """
    print("\n" + "="*120)
    print("TABLA A.2: Matriz de Confusión Detallada para Cada Modelo (Modelos Base - Sin SMOTE/SFS)")
    print("="*120)
    
    # Encabezado
    print(f"{'Modelo':<30} {'TN':<10} {'FP':<10} {'FN':<10} {'TP':<10}")
    print("-"*120)
    
    modelos = list(resultados_base.keys())
    
    for modelo in modelos:
        y_pred = resultados_base[modelo]['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        print(f"{modelo:<30} {int(tn):<10} {int(fp):<10} {int(fn):<10} {int(tp):<10}")
    
    return


# =============================================================================
# 6. EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Ejecución completa del análisis.
    """
    print("="*70)
    print("TFM - ANÁLISIS DE HEPATITIS C CON MACHINE LEARNING")
    print("="*70)
    
    # 1. Cargar y preparar datos
    print("\n" + "="*70)
    print("1. CARGA Y PREPARACIÓN DE DATOS")
    print("="*70)
    df = cargar_datos('hepatitisC.csv')
    X, y, scaler, label_encoder, df = preparar_datos(df)
    
    # 2. Generar visualizaciones exploratorias
    print("\n" + "="*70)
    print("2. VISUALIZACIONES EXPLORATORIAS")
    print("="*70)
    figura_1_distribucion_categorias(df)
    figura_2_distribucion_marcadores(df)
    figura_3_matriz_correlacion(X)
    figura_4_pca(X, y, df)
    
    # 3. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nDivisión train/test: {len(X_train)} / {len(X_test)}")
    
    # 4. Modelos base (SIN SMOTE/SFS)
    print("\n" + "="*70)
    print("4. MODELOS BASE (SIN SMOTE/SFS)")
    print("="*70)
    resultados_base = entrenar_modelos_base(X_train, X_test, y_train, y_test)
    figura_comparacion_modelos(resultados_base, 'figura_comparacion_base.png')
    figura_curvas_roc(resultados_base, y_test, 'figura_roc_base.png')
    
    # 5. Optimización de hiperparámetros
    print("\n" + "="*70)
    print("5. OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print("="*70)
    best_rf, cv_results_rf = optimizar_random_forest(X_train, y_train)
    best_gb, cv_results_gb = optimizar_gradient_boosting(X_train, y_train)
    
    # 6. Modelos con SMOTE y SFS
    print("\n" + "="*70)
    print("6. MODELOS CON SMOTE Y SFS")
    print("="*70)
    
    # Aplicar SMOTE
    X_train_smote, y_train_smote = aplicar_smote(X_train, y_train)
    
    # Generar figura A: Distribución SMOTE
    figura_smote_distribucion(y_train, y_train_smote, 'figuraA_smote.png')
    
    # Selección de características
    X_train_sfs, features_sel, sfs_selector = seleccion_caracteristicas_sfs(
        X_train_smote, y_train_smote, n_features=8
    )
    X_test_sfs = pd.DataFrame(
        sfs_selector.transform(X_test), 
        columns=features_sel
    )
    
    # Re-entrenar con SMOTE + SFS
    resultados_smote = entrenar_modelos_base(X_train_sfs, X_test_sfs, 
                                              y_train_smote, y_test)
    figura_comparacion_modelos(resultados_smote, 'figura_comparacion_smote.png')
    
    # 8. Generar todas las figuras adicionales
    print("\n" + "="*70)
    print("8. FIGURAS ADICIONALES DEL TFM")
    print("="*70)
    
    # Figuras del estudio SIN SMOTE/SFS
    figura_comparacion_metricas(resultados_base, 'Comparación de Métricas - Sin SMOTE/SFS',
                                'figura5_comparacion_base.png')
    figura_curvas_roc_detallada(resultados_base, y_test, 'Curvas ROC - Sin SMOTE/SFS',
                                'figura6_roc_base.png')
    figura_importancia_gini(best_rf, X.columns, 'figura7_importancia_base.png')
    
    # Figuras del estudio CON SMOTE/SFS
    figura_comparacion_metricas(resultados_smote, 'Comparación de Métricas - Con SMOTE/SFS',
                                'figura8_comparacion_smote.png')
    figura_roc_smote(resultados_base, resultados_smote, y_test, 'figura9_roc_smote.png')
    
    # Re-entrenar RF con features seleccionadas para importancia
    best_rf_sfs = RandomForestClassifier(**best_rf.get_params())
    best_rf_sfs.fit(X_train_sfs, y_train_smote)
    figura_importancia_sfs(best_rf_sfs, features_sel, features_sel,
                          'figura10_importancia_sfs.png')
    
    # Comparación final
    figura_comparacion_final(resultados_base, resultados_smote, 'figura11_comparacion_final.png')
    
    # 9. GENERACIÓN DE TABLAS Y LOGS DE DATOS
    print("\n" + "="*70)
    print("9. GENERACIÓN DE TABLAS Y LOGS DE DATOS")
    print("="*70)
    
    # Reconstruir X original (sin escalar) para las tablas
    features = ['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
    X_original = df[features].fillna(df[features].median())
    
    # Tabla: Descripción de variables
    tabla_descripcion_variables(X_original)
    
    # Tabla: Estadísticas descriptivas completas
    tabla_estadisticas_completas(X_original)
    
    # Tabla: Características seleccionadas por SFS
    tabla_caracteristicas_sfs_orden(features_sel)
    
    # Tabla: Grid Search Random Forest (Top 5)
    tabla_grid_search_rf(cv_results_rf)
    
    # Tabla: Grid Search Gradient Boosting (Top 5)
    tabla_grid_search_gb(cv_results_gb)
    
    # Tabla: Análisis multiclase
    tabla_multiclase(df, X, y)
    
    # Tabla A.1: Estadísticas por categoría
    tabla_estadisticas_categorias(df)
    
    # Tabla A.2: Matrices de confusión
    tabla_matrices_confusion(resultados_base, resultados_smote, y_test)
    
    print("\n" + "="*70)
    print("ANÁLISIS COMPLETADO - 11 FIGURAS + 8 TABLAS GENERADAS EN LOGS")
    print("="*70)
    
    print("\nFiguras generadas:")
    print("  [Exploratorias]")
    print("    - figura1_distribucion.png: Distribución por categoría")
    print("    - figura2_marcadores.png: Distribución de marcadores hepáticos")
    print("    - figura3_correlacion.png: Matriz de correlación de Pearson")
    print("    - figura4_pca.png: Análisis de Componentes Principales")
    print("\n  [Estudio SIN SMOTE/SFS]")
    print("    - figura5_comparacion_base.png: Comparación de métricas")
    print("    - figura6_roc_base.png: Curvas ROC")
    print("    - figura7_importancia_base.png: Importancia de variables (Gini)")
    print("\n  [Estudio CON SMOTE/SFS]")
    print("    - figura8_comparacion_smote.png: Comparación de métricas")
    print("    - figura9_roc_smote.png: Curvas ROC comparativas")
    print("    - figura10_importancia_sfs.png: Importancia con SFS (8 variables)")
    print("\n  [Comparación Final]")
    print("    - figura11_comparacion_final.png: Sin vs Con SMOTE/SFS")
    
    print("\nTablas generadas en logs:")
    print("  1. Descripción detallada de variables predictoras del estudio")
    print("  2. Estadísticas descriptivas completas de variables numéricas")
    print("  3. Características seleccionadas por SFS (orden de selección)")
    print("  4. Grid Search Random Forest - Top 5 configuraciones")
    print("  5. Grid Search Gradient Boosting - Top 5 configuraciones")
    print("  6. Métricas de clasificación multiclase por categoría diagnóstica")
    print("  7. Tabla A.1 - Estadísticas por categoría diagnóstica")
    print("  8. Tabla A.2 - Matriz de confusión detallada por modelo")


if __name__ == "__main__":
    main()
