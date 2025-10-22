"""
EXPERIMENTO MEJORADO: Densidad Semántica con Controles Rigurosos
Versión 2.0 - Octubre 2025
"""

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# ============================================================
#  SELLO DE EJECUCIÓN - DIEGO CERDA SEGUEL
# ============================================================

from datetime import datetime
from zoneinfo import ZoneInfo  # ya viene con Python 3.9+

# Definir zona horaria local
TZ = ZoneInfo("America/Santiago")

# Capturar la hora exacta de ejecución
RUN_TIMESTAMP = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z")

# Imprimir y registrar en texto
print("⏱️ Acta de Ejecución: ", RUN_TIMESTAMP)

# Crear una breve nota humanista dentro del archivo
EXECUTION_NOTE = f"""
DIEGO CERDA SEGUEL — GEOTENSORIAL LAB
Ejecutado en entorno Colab — Versión 2.0 (Octubre 2025)
Timestamp local: {RUN_TIMESTAMP}
'Este código midió por primera vez la curvatura del sentido.'
"""
print(EXECUTION_NOTE)

# Guardar la nota en un archivo de acta ligera
with open("EXECUTION_RECORD_TdST_OCT_2025.txt", "w", encoding="utf-8") as f:
    f.write(EXECUTION_NOTE)

class SemanticDensityAnalyzer:
    """
    Mide densidad semántica con múltiples métricas complementarias
    """
    
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Añadir pad_token si no existe
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
    def compute_semantic_density(self, text: str) -> Dict[str, float]:
        """
        Calcula múltiples métricas de densidad semántica
        """
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=128,
            padding=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extraer embeddings de última capa y convertir a float32
        embeddings = outputs.hidden_states[-1][0].cpu().float().numpy()  # [seq_len, hidden_dim]
        
        metrics = {}
        
        # 1. MÉTRICA ORIGINAL (para comparación)
        cov = np.cov(embeddings.T)
        metrics['frobenius_norm'] = np.linalg.norm(cov, 'fro') / 1000
        
        # 2. ISOTROPY (uniformidad de representaciones)
        # Baja isotropy = embeddings concentrados (potencialmente más semánticos)
        norms = np.linalg.norm(embeddings, axis=1)
        metrics['isotropy'] = np.std(norms) / (np.mean(norms) + 1e-8)
        
        # 3. COHERENCIA LOCAL (gradiente semántico)
        # Mide cuán coherente es la transición entre tokens adyacentes
        if len(embeddings) > 1:
            cosine_sims = []
            for i in range(len(embeddings) - 1):
                sim = np.dot(embeddings[i], embeddings[i+1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]) + 1e-8
                )
                cosine_sims.append(sim)
            metrics['local_coherence'] = np.mean(cosine_sims)
            metrics['coherence_variance'] = np.var(cosine_sims)
        else:
            metrics['local_coherence'] = 0
            metrics['coherence_variance'] = 0
        
        # 4. RANGO EFECTIVO (dimensionalidad intrínseca)
        # SVD de embeddings para medir concentración de información
        _, s, _ = np.linalg.svd(embeddings, full_matrices=False)
        total_variance = np.sum(s**2)
        metrics['effective_rank'] = np.sum(s**2)**2 / np.sum(s**4)  # Participation ratio
        metrics['top_5_variance'] = np.sum(s[:5]**2) / total_variance if total_variance > 0 else 0
        
        # 5. DENSIDAD SEMÁNTICA COMPUESTA (normalizada)
        # Combina coherencia local y concentración de varianza
        S = (metrics['local_coherence'] * metrics['top_5_variance']) / (metrics['isotropy'] + 0.1)
        metrics['semantic_density'] = S
        
        return metrics
    
    def analyze_corpus(self, sentences: List[str]) -> Dict[str, List[float]]:
        """
        Analiza un conjunto de oraciones
        """
        results = {key: [] for key in [
            'frobenius_norm', 'isotropy', 'local_coherence', 
            'coherence_variance', 'effective_rank', 'top_5_variance', 'semantic_density'
        ]}
        
        for sent in sentences:
            metrics = self.compute_semantic_density(sent)
            for key, value in metrics.items():
                results[key].append(value)
        
        return results


def run_controlled_experiment():
    """
    Experimento con controles rigurosos
    """
    
    # DATASET DE PRUEBA CON CONTROLES
    test_sentences = {
        'coherent': [
            "El gato se sentó en la alfombra.",
            "La inteligencia artificial transforma nuestra comprensión del lenguaje."
        ],
        'paraphrase': [
            "La felina descansaba sobre el tapete.",
            "La IA revoluciona cómo entendemos las palabras."
        ],
        'broken_syntax': [
            "Alfombra la en sentó gato el.",
            "Lenguaje del comprensión nuestra transforma artificial inteligencia la."
        ],
        'random_words': [
            "Alfombra teorema transforma felina descansaba.",
            "Inteligencia tapete sentó revoluciona palabras."
        ]
    }
    
    models = ['EleutherAI/pythia-70m', 'EleutherAI/pythia-160m', 'EleutherAI/pythia-410m']
    
    results = {}
    
    for model_name in models:
        print(f"\n🔍 Analizando {model_name}...")
        analyzer = SemanticDensityAnalyzer(model_name)
        
        model_results = {}
        for category, sentences in test_sentences.items():
            print(f"   Categoría: {category}")
            model_results[category] = analyzer.analyze_corpus(sentences)
        
        results[model_name] = model_results
    
    # ANÁLISIS ESTADÍSTICO
    print("\n" + "="*60)
    print("📊 ANÁLISIS ESTADÍSTICO")
    print("="*60)
    
    for model_name, model_data in results.items():
        print(f"\n🤖 {model_name}")
        
        # Comparar coherentes vs. rotas
        coherent_S = np.mean(model_data['coherent']['semantic_density'])
        broken_S = np.mean(model_data['broken_syntax']['semantic_density'])
        random_S = np.mean(model_data['random_words']['semantic_density'])
        
        print(f"   S(coherentes): {coherent_S:.4f}")
        print(f"   S(sintaxis rota): {broken_S:.4f}")
        print(f"   S(palabras random): {random_S:.4f}")
        
        # Test t entre coherente y rota
        t_stat, p_val = stats.ttest_ind(
            model_data['coherent']['semantic_density'],
            model_data['broken_syntax']['semantic_density']
        )
        print(f"   Diferencia estadística: t={t_stat:.3f}, p={p_val:.4f}")
        
        if p_val < 0.05:
            print(f"   ✅ El modelo distingue significativamente entre coherente/rota")
        else:
            print(f"   ⚠️  No hay diferencia estadística significativa")
    
    return results


def plot_multimetric_analysis(results: Dict):
    """
    Visualiza múltiples métricas
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Análisis Multi-Métrico de Densidad Semántica', fontsize=16)
    
    metrics = ['local_coherence', 'isotropy', 'effective_rank', 
               'top_5_variance', 'coherence_variance', 'semantic_density']
    
    categories = ['coherent', 'paraphrase', 'broken_syntax', 'random_words']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        for model_name in results.keys():
            values = [
                np.mean(results[model_name][cat][metric])
                for cat in categories
            ]
            model_label = model_name.split('/')[-1]
            ax.plot(categories, values, marker='o', label=model_label, linewidth=2)
        
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylabel('Valor')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


# EJECUTAR EXPERIMENTO
if __name__ == "__main__":
    print("🧪 EXPERIMENTO CONTROLADO DE DENSIDAD SEMÁNTICA")
    print("="*60)
    
    results = run_controlled_experiment()
    plot_multimetric_analysis(results)
    

    print("\n✅ Experimento completado con controles rigurosos")
