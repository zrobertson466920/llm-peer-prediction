import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from scipy import stats

def detect_task_type(dataset_name):
    """Detect task type from dataset name."""
    if 'translation' in dataset_name.lower():
        return 'translation'
    elif 'summarization' in dataset_name.lower() or 'cnn' in dataset_name.lower():
        return 'summarization'
    elif 'review' in dataset_name.lower() or 'iclr' in dataset_name.lower():
        return 'peer_review'
    return 'translation'  # default

def load_task_config(task_type):
    """Load task-specific configuration."""
    configs = {
        'translation': {
            'categories': {
                'Strategic': ['All Positive', 'All Negative', 'Misleading', 'Contradictory'],
                'Low Effort': ['Low Effort', 'Exaggerate', 'Understate', 'Sarcastic'],
                'Style': ['Informal', 'Simplify', 'Poetic', 'Humorous', 'Dramatic', 'Historical', 'Futuristic', 'Persuasive', 'Emotional', 'Subjective', 'Metaphorical', 'Comparative', 'Hypothetical', 'Philosophical', 'Quantitative', 'Creative'],
                'Faithful': ['Original', 'Formal', 'Technical', 'Academic', 'Objective', 'Cultural']
            }
        },
        'summarization': {
            'categories': {
                'Strategic': [
                    'Fact Manipulation',
                    'Selective Omission', 
                    'Misleading Emphasis',
                    'Sentiment Flip',
                    'Conspiracy Theory',
                    'Contradictory',
                    'Context Removal',
                    'False Attribution',
                    'Agenda Push',
                    'Cherry Pick'
                ],
                'Low Effort': [
                    'Low Effort',
                    'Ultra Concise',
                    'Template Response',
                    'Surface Skim',
                    'Minimal Detail'
                ],
                'Style': [
                    'Academic Style',
                    'Sensationalist',
                    'Bureaucratic',
                    'Casual Conversational',
                    'Historical Perspective',
                    'Poetic',
                    'Technical Jargon',
                    'Euphemistic',
                    'Overly Technical',
                    'Minimalist'
                ],
                'Faithful': [
                    'Faithful',
                    'Objective',
                    'Comprehensive',
                    'Neutral Tone'
                ]
            }
        },
        'peer_review': {
            'categories': {
                'Strategic': [
                    'Method Shift',
                    'Question Shift', 
                    'Contribution Misrepresent',
                    'Result Manipulation',
                    'Assumption Attack',
                    'Dismissive Expert',
                    'Agenda Push',
                    'Benchmark Obsessed'
                ],
                'Low Effort': [
                    'Low Effort',
                    'Generic',
                    'Surface Skim',
                    'Template Fill',
                    'Checklist Review'
                ],
                'Style': [
                    'Balanced Critique',
                    'Overly Technical',
                    'Harsh Critique',
                    'Overly Positive',
                    'Theory Focus',
                    'Implementation Obsessed',
                    'Comparison Fixated',
                    'Pedantic Details',
                    'Scope Creep',
                    'Statistical Nitpick',
                    'Future Work Focus',
                    'Writing Critique'
                ],
                'Faithful': [
                    'Reference',
                    'Faithful',
                    'Objective Analysis',
                    'Thorough Evaluation'
                ]
            }
        }
    }
    return configs.get(task_type, configs['translation'])

class MatrixLoader:
    """Load and organize matrices from peer prediction evaluation results."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.matrices = {}
        self.metadata = {}

    def load_all_matrices(self) -> Dict[str, Dict]:
        """Load matrices from all available mechanisms."""

        # Load aggregated results
        self._load_mi_gppm()
        self._load_tvd_mi()
        self._load_judge_with_context()
        self._load_judge_without_context()

        # Load individual example matrices if needed
        self._load_individual_examples()

        return self.matrices

    def _load_json_file(self, pattern: str) -> Optional[Dict]:
        """Load the first JSON file matching the pattern."""
        files = list(self.results_dir.glob(pattern))
        if files:
            with open(files[0], 'r') as f:
                return json.load(f)
        return None

    def _load_mi_gppm(self):
        """Load MI/GPPM matrices from aggregated results."""
        data = self._load_json_file("*_mi_gppm.json")
        if data:
            # Store the raw data for inspection
            self.matrices['mi_gppm'] = data
            self.matrices['mi'] = data  # MI uses the same data
            self.matrices['gppm'] = data  # GPPM uses the same data

            # Extract metadata
            metadata = {
                'task_type': data.get('task_type'),
                'num_examples': data.get('num_examples'),
                'condition_keys': data.get('condition_keys', [])
            }
            self.metadata['mi_gppm'] = metadata
            self.metadata['mi'] = metadata
            self.metadata['gppm'] = metadata

    def _load_tvd_mi(self):
        """Load TVD-MI matrices from aggregated results."""
        data = self._load_json_file("*_tvd_mi.json")
        if data:
            # Store the raw data
            self.matrices['tvd_mi'] = data

            # Extract metadata
            self.metadata['tvd_mi'] = {
                'task_type': data.get('task_type'),
                'num_examples': data.get('num_examples'),
                'condition_keys': data.get('condition_keys', [])
            }

    def _load_judge_with_context(self):
        """Load judge with context matrices."""
        data = self._load_json_file("*_judge_with_context.json")
        if data:
            # Store the raw data
            self.matrices['judge_with_context'] = data

            # Extract metadata
            self.metadata['judge_with_context'] = {
                'task_type': data.get('task_type'),
                'num_examples': data.get('num_examples'),
                'condition_keys': data.get('condition_keys', [])
            }

    def _load_judge_without_context(self):
        """Load judge without context matrices."""
        data = self._load_json_file("*_judge_without_context.json")
        if data:
            # Store the raw data
            self.matrices['judge_without_context'] = data

            # Extract metadata
            self.metadata['judge_without_context'] = {
                'task_type': data.get('task_type'),
                'num_examples': data.get('num_examples'),
                'condition_keys': data.get('condition_keys', [])
            }

    def _load_individual_examples(self):
        """Load individual example matrices from archive directories."""

        # MI/GPPM individual examples
        mi_dir = self.results_dir / "log_individual_examples"
        if mi_dir.exists():
            individual_data = self._load_individual_mi_examples(mi_dir)
            self.matrices['mi_gppm_individual'] = individual_data
            self.matrices['mi_individual'] = individual_data
            self.matrices['gppm_individual'] = individual_data

        # TVD-MI individual examples
        tvd_dir = self.results_dir / "tvd_mi_individual_examples"
        if tvd_dir.exists():
            self.matrices['tvd_mi_individual'] = self._load_individual_tvd_examples(tvd_dir)

        # Judge individual examples
        judge_context_dir = self.results_dir / "llm_context_individual_examples"
        if judge_context_dir.exists():
            self.matrices['judge_with_context_individual'] = self._load_individual_judge_examples(judge_context_dir)

        judge_no_context_dir = self.results_dir / "llm_without_context_individual_examples"
        if judge_no_context_dir.exists():
            self.matrices['judge_without_context_individual'] = self._load_individual_judge_examples(judge_no_context_dir)

    def _load_individual_mi_examples(self, directory: Path) -> Dict:
        """Load MI/GPPM individual example matrices."""
        examples = {}
        for file in sorted(directory.glob("peer_prediction_example_*.json")):
            with open(file, 'r') as f:
                data = json.load(f)
                idx = data['example_idx']
                examples[idx] = {
                    'logp_base': np.array(data['logp_base']),
                    'logp_cond': np.array(data['logp_cond']),
                    'difference_matrix': np.array(data['difference_matrix']),
                    'row_avgs': np.array(data['row_avgs']),
                    'col_avgs': np.array(data['col_avgs']),
                    'combined_avgs': np.array(data['combined_avgs']),
                    'gppm': np.array(data['gppm']),
                    'gppm_normalized': np.array(data['gppm_normalized']),
                    'condition_keys': data['condition_keys']
                }
        return examples

    def _load_individual_tvd_examples(self, directory: Path) -> Dict:
        """Load TVD-MI individual example matrices."""
        examples = {}
        for file in sorted(directory.glob("tvd_mi_example_*.json")):
            with open(file, 'r') as f:
                data = json.load(f)
                idx = data['example_idx']
                examples[idx] = {
                    'tvd_mi_matrix': np.array(data['tvd_mi_matrix']),
                    'tvd_mi_scores': np.array(data['tvd_mi_scores']),
                    'tvd_mi_bidirectional': np.array(data['tvd_mi_bidirectional']),
                    'condition_keys': data['condition_keys']
                }
        return examples

    def _load_individual_judge_examples(self, directory: Path) -> Dict:
        """Load judge individual example matrices."""
        examples = {}
        for file in sorted(directory.glob("judge_with_context_example_*.json")):
            with open(file, 'r') as f:
                data = json.load(f)
                idx = data['example_idx']
                examples[idx] = {
                    'win_matrix': np.array(data['win_matrix']),
                    'win_rates': np.array(data['win_rates']),
                    'condition_keys': data['condition_keys'],
                    'with_context': data.get('with_context', True)
                }
        return examples
    
    def get_symmetric_score(self, example_data: Dict, mechanism: str, idx1: int, idx2: int) -> Optional[float]:
        """Get symmetric pairwise score for a given mechanism.
        
        Args:
            example_data: Individual example data containing matrices
            mechanism: The mechanism to use
            idx1, idx2: Indices of conditions to compare
            
        Returns:
            Symmetric score or None if not available
        """
        if mechanism in ['mi_gppm', 'mi']:
            matrix = example_data.get('difference_matrix')
            if matrix is not None and matrix.shape[0] > max(idx1, idx2) and matrix.shape[1] > max(idx1, idx2):
                # Average the directional scores for symmetry
                return 0.5 * (matrix[idx1, idx2] + matrix[idx2, idx1])
                
        elif mechanism == 'gppm':
            matrix = example_data.get('logp_cond')
            if matrix is not None and matrix.shape[0] > max(idx1, idx2) and matrix.shape[1] > max(idx1, idx2):
                # Average the directional scores for symmetry
                return 0.5 * (matrix[idx1, idx2] + matrix[idx2, idx1])
                
        elif mechanism == 'tvd_mi':
            matrix = example_data.get('tvd_mi_matrix')
            if matrix is not None and matrix.shape[0] > max(idx1, idx2) and matrix.shape[1] > max(idx1, idx2):
                # TVD-MI might already be symmetric, but symmetrize to be safe
                return 0.5 * (matrix[idx1, idx2] + matrix[idx2, idx1])
                
        elif mechanism in ['judge_with_context','judge_without_context']:
            W = example_data.get('win_matrix')
            if W is None: return None
            conds = example_data['condition_keys']
            n = W.shape[0]

            if 'Reference' in conds:
                r = conds.index('Reference')
                # orientation-invariant quality in [0,1]
                q = 0.5 * (W[:, r].astype(float) + (1.0 - W[r, :].astype(float)))
            else:
                # row-mean proxy (no modeling)
                Wm = W.astype(float).copy()
                np.fill_diagonal(Wm, np.nan)
                q = np.nanmean(Wm, axis=1)

            qa, qb = q[idx1], q[idx2]
            if not (np.isfinite(qa) and np.isfinite(qb)): return None

            # 2) slightly smoother “both must be good” aggregator
            return float(min(qa, qb))
    
    def calculate_per_item_auc(self, mechanism: str, faithful_conditions: List[str] = None,
                              problematic_conditions: List[str] = None, task_type: str = None,
                              balance_classes: bool = True) -> np.ndarray:
        """Calculate AUC for each item separately to avoid pooling bias.
        
        Returns:
            Array of per-item AUC scores
        """
        # If task_type is provided, use it to get appropriate conditions
        if task_type and faithful_conditions is None and problematic_conditions is None:
            task_config = load_task_config(task_type)
            categories = task_config['categories']
            
            # Faithful conditions include Faithful and Style categories
            faithful_conditions = categories.get('Faithful', []) + categories.get('Style', [])
            
            # Problematic conditions include Strategic and Low Effort categories
            problematic_conditions = categories.get('Strategic', []) + categories.get('Low Effort', [])
        
        # Filter to available conditions
        if mechanism not in self.metadata:
            return np.array([])
            
        # Calculate AUC for each item (respect per-example condition order)
        individual_key = f"{mechanism}_individual"
        if individual_key not in self.matrices:
            return np.array([])
            
        per_item_aucs = []
        
        for example_idx, example_data in self.matrices[individual_key].items():
            ex_keys = example_data.get('condition_keys', [])
            faithful_indices    = [ex_keys.index(c) for c in faithful_conditions if c in ex_keys]
            problematic_indices = [ex_keys.index(c) for c in problematic_conditions if c in ex_keys]
            if not faithful_indices or not problematic_indices:
                continue
            y_true = []
            y_scores = []
            
            # Collect faithful-faithful pairs (positive class) - unordered pairs
            for i in range(len(faithful_indices)):
                for j in range(i + 1, len(faithful_indices)):
                    idx1, idx2 = faithful_indices[i], faithful_indices[j]
                    score = self.get_symmetric_score(example_data, mechanism, idx1, idx2)
                    if score is not None and np.isfinite(score):
                        y_true.append(1)
                        y_scores.append(score)
            
            # Collect faithful-problematic pairs (negative class) - unordered pairs
            for faithful_idx in faithful_indices:
                for problematic_idx in problematic_indices:
                    score = self.get_symmetric_score(example_data, mechanism, faithful_idx, problematic_idx)
                    if score is not None and np.isfinite(score):
                        y_true.append(0)
                        y_scores.append(score)
            
            # Balance classes if requested
            if balance_classes and len(y_true) > 0:
                positive_indices = [i for i, y in enumerate(y_true) if y == 1]
                negative_indices = [i for i, y in enumerate(y_true) if y == 0]
                
                if positive_indices and negative_indices:
                    # Downsample to the minority class
                    min_size = min(len(positive_indices), len(negative_indices))
                    
                    # Use fixed seed for reproducibility within each item
                    rng = np.random.RandomState(example_idx)
                    
                    if len(positive_indices) > min_size:
                        positive_indices = rng.choice(positive_indices, min_size, replace=False).tolist()
                    if len(negative_indices) > min_size:
                        negative_indices = rng.choice(negative_indices, min_size, replace=False).tolist()
                    
                    balanced_indices = positive_indices + negative_indices
                    y_true = [y_true[i] for i in balanced_indices]
                    y_scores = [y_scores[i] for i in balanced_indices]
            
            # Calculate AUC for this item if we have both classes
            if len(set(y_true)) == 2 and len(y_true) >= 4:  # Need at least some examples of each class
                try:
                    auc = roc_auc_score(y_true, y_scores)
                    per_item_aucs.append(auc)
                except:
                    pass  # Skip items where AUC calculation fails
                    
        return np.array(per_item_aucs)
    
    def bootstrap_confidence_interval(self, aucs: np.ndarray, n_bootstrap: int = 1000, 
                                    confidence_level: float = 0.95) -> Tuple[float, float, float]:
        """Calculate bootstrap confidence interval for AUC scores.
        
        Args:
            aucs: Array of per-item AUC scores
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            (mean_auc, lower_ci, upper_ci)
        """
        if len(aucs) == 0:
            return np.nan, np.nan, np.nan
            
        bootstrap_means = []
        n_items = len(aucs)
        
        for _ in range(n_bootstrap):
            # Resample items with replacement
            bootstrap_indices = np.random.choice(n_items, size=n_items, replace=True)
            bootstrap_sample = aucs[bootstrap_indices]
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        mean_auc = np.mean(aucs)
        lower_ci = np.percentile(bootstrap_means, lower_percentile)
        upper_ci = np.percentile(bootstrap_means, upper_percentile)
        
        return mean_auc, lower_ci, upper_ci
    
    def plot_all_roc_curves(self, mechanisms: List[str], save_path: Optional[str] = None, task_type: str = None, title: str = None):
        """Plot ROC curves for all mechanisms together in one plot."""
        # Get task-specific conditions
        task_config = load_task_config(task_type)
        categories = task_config['categories']
        
        faithful_conditions = categories.get('Faithful', []) + categories.get('Style', [])
        problematic_conditions = categories.get('Strategic', []) + categories.get('Low Effort', [])
        
        plt.figure(figsize=(10, 8))

        # Colors for different mechanisms
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for idx, mechanism in enumerate(mechanisms):
            # Get per-item AUCs with balanced classes
            per_item_aucs = self.calculate_per_item_auc(mechanism, faithful_conditions, 
                                                       problematic_conditions, task_type,
                                                       balance_classes=True)
            if len(per_item_aucs) == 0:
                print(f"Could not calculate per-item AUCs for {mechanism}")
                continue
                
            mean_auc, lower_ci, upper_ci = self.bootstrap_confidence_interval(per_item_aucs)
            
            # Aggregate data for ROC curve visualization
            all_y_true = []
            all_y_scores = []
            
            individual_key = f"{mechanism}_individual"
            if individual_key not in self.matrices:
                continue
                
            for example_idx, example_data in self.matrices[individual_key].items():
                ex_keys = example_data.get('condition_keys', [])
                faithful_indices = [ex_keys.index(c) for c in faithful_conditions if c in ex_keys]
                problematic_indices = [ex_keys.index(c) for c in problematic_conditions if c in ex_keys]
                
                if not faithful_indices or not problematic_indices:
                    continue
                    
                # Faithful-faithful pairs (positive class)
                for i in range(len(faithful_indices)):
                    for j in range(i + 1, len(faithful_indices)):
                        idx1, idx2 = faithful_indices[i], faithful_indices[j]
                        score = self.get_symmetric_score(example_data, mechanism, idx1, idx2)
                        if score is not None and np.isfinite(score):
                            all_y_true.append(1)
                            all_y_scores.append(score)
                
                # Faithful-problematic pairs (negative class)
                for faithful_idx in faithful_indices:
                    for problematic_idx in problematic_indices:
                        score = self.get_symmetric_score(example_data, mechanism, 
                                                        faithful_idx, problematic_idx)
                        if score is not None and np.isfinite(score):
                            all_y_true.append(0)
                            all_y_scores.append(score)
            
            if len(set(all_y_true)) != 2:
                print(f"Insufficient data for ROC curve for {mechanism}")
                continue
                
            fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)
            
            # Plot with mechanism-specific color
            color = colors[idx % len(colors)]
            plt.plot(fpr, tpr, lw=2, color=color,
                    label=f'{mechanism}: AUC = {mean_auc:.3f} [{lower_ci:.3f}, {upper_ci:.3f}]')
        
        # Plot diagonal reference line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves Comparison: {title} Task\nFaithful+Style vs Strategic+LowEffort', 
                 fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()

    # ---------- Judge helpers (no extra modeling) ----------

    def _judge_quality_vector(self, example_data):
        """Orientation-invariant per-response 'quality' vs Reference in [0,1]."""
        W = example_data['win_matrix'].astype(float)
        conds = example_data['condition_keys']
        if 'Reference' in conds:
            r = conds.index('Reference')
            # symmetry: prob(row beats ref) + prob(ref loses to row)
            q = 0.5 * (W[:, r] + (1.0 - W[r, :]))
        else:
            # fallback: row-mean (still orientation-free)
            Wm = W.copy()
            np.fill_diagonal(Wm, np.nan)
            q = np.nanmean(Wm, axis=1)
        return q, conds

def get_dataset_name(folder_name: str) -> Tuple[str, str]:
    """Extract dataset name and domain from folder name."""
    # Mapping of folder patterns to dataset names
    folder_lower = folder_name.lower()
    
    # Translation datasets
    if 'mt14' in folder_lower or 'wmt14' in folder_lower:
        return 'MT14', 'Translation'
    elif 'opus' in folder_lower:
        return 'OPUS', 'Translation'
    
    # Summarization datasets
    elif 'billsum' in folder_lower:
        return 'BillSum', 'Summarization'
    elif 'cnn' in folder_lower or 'dailymail' in folder_lower:
        return 'CNN/DM', 'Summarization'
    elif 'multinews' in folder_lower or 'multi_news' in folder_lower:
        return 'MultiNews', 'Summarization'
    elif 'pubmed' in folder_lower:
        return 'PubMed', 'Summarization'
    elif 'reddit' in folder_lower or 'tifu' in folder_lower:
        return 'Reddit TIFU', 'Summarization'
    elif 'samsum' in folder_lower:
        return 'SAMSum', 'Summarization'
    elif 'xsum' in folder_lower:
        return 'XSum', 'Summarization'
    
    # Peer review datasets
    elif 'iclr' in folder_lower:
        return 'ICLR', 'Peer Review'
    
    # Attack types (for second table)
    elif 'case_flip' in folder_lower:
        return 'Case Flip', 'Attack'
    elif 'format' in folder_lower:
        return 'Format', 'Attack'
    elif 'padding' in folder_lower:
        return 'Padding', 'Attack'
    elif 'pattern' in folder_lower:
        return 'Pattern', 'Attack'
    
    # Default
    return folder_name, 'Unknown'

def format_auc_with_ci(mean_auc: float, lower_ci: float, upper_ci: float) -> str:
    """Format AUC with confidence interval for LaTeX table."""
    if np.isnan(mean_auc):
        return "—"
    
    # Calculate half-width of CI
    half_width = (upper_ci - lower_ci) / 2
    
    return f"{mean_auc:.3f} ± {half_width:.3f}"

def generate_latex_tables(results_dict: Dict[str, Dict[str, Tuple[float, float, float]]]) -> Tuple[str, str]:
    """Generate the two LaTeX tables from results."""
    
    # Table 1: Main results by domain
    table1_lines = []
    table1_lines.append("\\begin{table}[t]")
    table1_lines.append("\\centering")
    table1_lines.append("\\caption{AUC scores for distinguishing Faithful-Faithful from Faithful-Problematic agent pairs across domains. Information-theoretic mechanisms (especially TVD-MI) consistently outperform LLM judges, even when judges have access to source material. Values show macro-averaged AUC ± 95\\% CI half-width.}")
    table1_lines.append("\\label{tab:auc_results}")
    table1_lines.append("\\begin{tabular}{lccccc}")
    table1_lines.append("\\toprule")
    table1_lines.append("\\textbf{Domain} & \\textbf{n} & \\textbf{MI (DoE)} & \\textbf{GPPM} & \\textbf{TVD-MI} & \\textbf{Judge w/ context} \\\\")
    table1_lines.append("\\midrule")
    
    # Group results by domain
    domains = {}
    attack_results = {}
    
    for folder_name, data in results_dict.items():
        dataset_name, domain = get_dataset_name(folder_name)
        
        if domain == 'Attack':
            attack_results[dataset_name] = data
        else:
            if domain not in domains:
                domains[domain] = []
            domains[domain].append((dataset_name, data))
    
    # Sort domains and datasets within each domain
    domain_order = ['Translation', 'Summarization', 'Peer Review']
    
    for domain in domain_order:
        if domain not in domains:
            continue
            
        table1_lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{domain}}}}} \\\\")
        
        # Sort datasets within domain
        datasets = sorted(domains[domain], key=lambda x: x[0])
        
        for dataset_name, data in datasets:
            n = data.get('n', '—')
            mi_auc = format_auc_with_ci(*data.get('mi', (np.nan, np.nan, np.nan)))
            gppm_auc = format_auc_with_ci(*data.get('gppm', (np.nan, np.nan, np.nan)))
            tvd_mi_auc = format_auc_with_ci(*data.get('tvd_mi', (np.nan, np.nan, np.nan)))
            judge_auc = format_auc_with_ci(*data.get('judge_with_context', (np.nan, np.nan, np.nan)))
            
            # Find best score (excluding judge)
            scores = []
            if 'mi' in data and not np.isnan(data['mi'][0]):
                scores.append(('mi', data['mi'][0]))
            if 'gppm' in data and not np.isnan(data['gppm'][0]):
                scores.append(('gppm', data['gppm'][0]))
            if 'tvd_mi' in data and not np.isnan(data['tvd_mi'][0]):
                scores.append(('tvd_mi', data['tvd_mi'][0]))
            
            best_mechanism = max(scores, key=lambda x: x[1])[0] if scores else None
            
            # Bold the best score
            if best_mechanism == 'mi':
                mi_auc = f"\\textbf{{{mi_auc}}}"
            elif best_mechanism == 'gppm':
                gppm_auc = f"\\textbf{{{gppm_auc}}}"
            elif best_mechanism == 'tvd_mi':
                tvd_mi_auc = f"\\textbf{{{tvd_mi_auc}}}"
            
            table1_lines.append(f"{dataset_name} & {n} & {mi_auc} & {gppm_auc} & {tvd_mi_auc} & {judge_auc} \\\\")
        
        if domain != domain_order[-1]:  # Don't add midrule after last domain
            table1_lines.append("\\midrule")
    
    table1_lines.append("\\bottomrule")
    table1_lines.append("\\end{tabular}")
    table1_lines.append("\\end{table}")
    
    # Table 2: Attack results
    table2_lines = []
    table2_lines.append("\\begin{center}")
    table2_lines.append("\\begin{tabular}{lcccc}")
    table2_lines.append("\\toprule")
    table2_lines.append("\\textbf{Attack} & \\textbf{MI} & \\textbf{GPPM} & \\textbf{TVD-MI} & \\textbf{Judge} \\\\")
    table2_lines.append("\\midrule")
    
    # Sort attacks by name
    attack_order = ['Case Flip', 'Format', 'Padding', 'Pattern']
    
    for attack_name in attack_order:
        if attack_name not in attack_results:
            continue
            
        data = attack_results[attack_name]
        
        # For attacks, we just show the mean AUC without CI
        mi_auc = f"{data.get('mi', (np.nan,))[0]:.3f}" if 'mi' in data and not np.isnan(data['mi'][0]) else "—"
        gppm_auc = f"{data.get('gppm', (np.nan,))[0]:.3f}" if 'gppm' in data and not np.isnan(data['gppm'][0]) else "—"
        tvd_mi_auc = f"{data.get('tvd_mi', (np.nan,))[0]:.3f}" if 'tvd_mi' in data and not np.isnan(data['tvd_mi'][0]) else "—"
        judge_auc = f"{data.get('judge_with_context', (np.nan,))[0]:.3f}" if 'judge_with_context' in data and not np.isnan(data['judge_with_context'][0]) else "—"
        
        # Find best score
        scores = []
        if 'mi' in data and not np.isnan(data['mi'][0]):
            scores.append(('mi', data['mi'][0]))
        if 'gppm' in data and not np.isnan(data['gppm'][0]):
            scores.append(('gppm', data['gppm'][0]))
        if 'tvd_mi' in data and not np.isnan(data['tvd_mi'][0]):
            scores.append(('tvd_mi', data['tvd_mi'][0]))
        if 'judge_with_context' in data and not np.isnan(data['judge_with_context'][0]):
            scores.append(('judge', data['judge_with_context'][0]))
        
        best_mechanism = max(scores, key=lambda x: x[1])[0] if scores else None
        
        # Bold the best score
        if best_mechanism == 'mi':
            mi_auc = f"\\textbf{{{mi_auc}}}"
        elif best_mechanism == 'gppm':
            gppm_auc = f"\\textbf{{{gppm_auc}}}"
        elif best_mechanism == 'tvd_mi':
            tvd_mi_auc = f"\\textbf{{{tvd_mi_auc}}}"
        elif best_mechanism == 'judge':
            judge_auc = f"\\textbf{{{judge_auc}}}"
        
        table2_lines.append(f"{attack_name} & {mi_auc} & {gppm_auc} & {tvd_mi_auc} & {judge_auc} \\\\")
    
    table2_lines.append("\\bottomrule")
    table2_lines.append("\\end{tabular}")
    table2_lines.append("\\end{center}")
    
    return '\n'.join(table1_lines), '\n'.join(table2_lines)

def process_folder_for_tables(folder_path: str) -> Dict[str, Tuple[float, float, float]]:
    """Process a single folder and return AUC results for table generation."""
    results = {}
    
    try:
        # Load all matrices
        loader = MatrixLoader(folder_path)
        matrices = loader.load_all_matrices()
        
        if not matrices:
            return results
        
        # Detect task type
        detected_task_type = None
        folder_path_obj = Path(folder_path)
        for file in folder_path_obj.glob("*.json"):
            detected_task_type = detect_task_type(file.name)
            break
        
        if detected_task_type is None:
            detected_task_type = 'translation'
        
        # Calculate balanced AUC for each mechanism and get n from first successful mechanism
        mechanisms = ['mi', 'gppm', 'tvd_mi', 'judge_with_context']
        n_examples = None

        for mechanism in mechanisms:
            per_item_aucs = loader.calculate_per_item_auc(mechanism, task_type=detected_task_type, balance_classes=True)

            if len(per_item_aucs) > 0:
                if n_examples is None:  # Get n from first mechanism with results
                    n_examples = len(per_item_aucs)
                mean_auc, lower_ci, upper_ci = loader.bootstrap_confidence_interval(per_item_aucs)
                results[mechanism] = (mean_auc, lower_ci, upper_ci)

        results['n'] = n_examples if n_examples else '—'
    
    except Exception as e:
        print(f"Error processing {folder_path}: {str(e)}")
    
    return results

def main():
    """Process all folders in eval_results and generate LaTeX tables."""
    
    # Get all subdirectories in eval_results
    eval_results_dir = Path("eval_results")
    if not eval_results_dir.exists():
        print(f"Error: {eval_results_dir} directory not found")
        return

    # Collect all sub directories
    folders = [f for f in eval_results_dir.iterdir() if f.is_dir()]
    folders.sort()  # Sort for consistent output
    
    if not folders:
        print("No subdirectories found in eval_results")
        return
    
    # Process each folder and collect results
    all_results = {}
    
    # Attack folders to process for second table
    attack_folders = [
        'summarization_gpt_4o_mini_20250708_012535_case_flip_transformed_results',
        'summarization_gpt_4o_mini_20250708_012535_format_transformed_results',
        'summarization_gpt_4o_mini_20250708_012535_padding_transformed_results',
        'summarization_gpt_4o_mini_20250708_012535_pattern_transformed_results'
    ]
    
    for folder in folders:
        folder_name = folder.name
        
        # Skip attack folders for the main table
        is_attack_folder = any(attack in folder_name for attack in ['case_flip', 'format', 'padding', 'pattern'])
        
        print(f"Processing {folder_name}...")
        results = process_folder_for_tables(str(folder))
        
        if results:
            all_results[folder_name] = results
    
    # Generate LaTeX tables
    table1, table2 = generate_latex_tables(all_results)
    
    # Save tables to files
    with open("tables/auc_table_main_results.tex", 'w') as f:
        f.write(table1)
    
    with open("tables/auc_table_attack_results.tex", 'w') as f:
        f.write(table2)
    
    # Print tables
    print("\n" + "="*80)
    print("TABLE 1: Main Results")
    print("="*80)
    print(table1)
    
    print("\n" + "="*80)
    print("TABLE 2: Attack Results")
    print("="*80)
    print(table2)
    
    print(f"\nTables saved to table1_main_results.tex and table2_attack_results.tex")
    
    # Also generate ROC curves for each folder
    print("\nGenerating ROC curves...")
    for folder in folders:
        loader = MatrixLoader(str(folder))
        matrices = loader.load_all_matrices()
        
        if matrices:
            # Detect task type
            detected_task_type = None
            for file in folder.glob("*.json"):
                detected_task_type = detect_task_type(file.name)
                break
            
            # Plot all mechanisms together
            mechanisms = ['mi', 'gppm', 'tvd_mi', 'judge_with_context']
            # Build save path
            save_dir = "roc_curves"
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(save_dir, f"roc_curves_{folder.name}.png")
            
            try:
                # Generate title based on folder name
                folder_name_parts = folder.name.split('_')

                # Check if it's an attack folder
                is_attack = any(attack in folder.name for attack in ['case_flip', 'format', 'padding', 'pattern'])

                if is_attack:
                    # For attack folders, use "Reddit TIFU (Attack Name)"
                    if 'case_flip' in folder.name:
                        plot_title = "Reddit TIFU (Case Flip)"
                    elif 'format' in folder.name:
                        plot_title = "Reddit TIFU (Format)"
                    elif 'padding' in folder.name:
                        plot_title = "Reddit TIFU (Padding)"
                    elif 'pattern' in folder.name:
                        plot_title = "Reddit TIFU (Pattern)"
                    else:
                        plot_title = "Reddit TIFU (Unknown Attack)"
                else:
                    # For regular folders, use first two words
                    if len(folder_name_parts) >= 2:
                        first_word = folder_name_parts[0].capitalize()
                        second_word = folder_name_parts[1].capitalize()
                        plot_title = f"{first_word} {second_word}"
                    else:
                        plot_title = folder.name

                loader.plot_all_roc_curves(
                    mechanisms,
                    save_path=save_path,
                    task_type=detected_task_type,
                    title=plot_title
                )
            except Exception as e:
                print(f"  Error generating ROC curve for {folder.name}: {str(e)}")

if __name__ == "__main__":
    main()
