import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from scipy import stats

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

    def get_condition_scores(self, mechanism: str, condition: str) -> np.ndarray:
        """Get scores for a specific condition across all examples."""
        if mechanism in self.matrices:
            data = self.matrices[mechanism]

            # Handle different data structures
            if isinstance(data, dict):
                # Try different possible keys for scores
                for key in ['mi_scores', 'tvd_mi_scores', 'tvd_mi_bidirectional', 'win_rates', 
                           'gppm_scores', 'gppm_normalized_scores']:
                    if key in data:
                        scores_data = data[key]
                        if isinstance(scores_data, dict) and condition in scores_data:
                            return np.array(scores_data[condition])
                        elif isinstance(scores_data, list):
                            # If it's a list, we need condition_keys to map
                            condition_keys = data.get('condition_keys', [])
                            if condition in condition_keys:
                                idx = condition_keys.index(condition)
                                return np.array([scores_data[idx]])
        return np.array([])

    def get_pairwise_matrix(self, mechanism: str, example_idx: int) -> Optional[np.ndarray]:
        """Get the pairwise comparison matrix for a specific example."""
        individual_key = f"{mechanism}_individual"
        if individual_key in self.matrices and example_idx in self.matrices[individual_key]:
            example_data = self.matrices[individual_key][example_idx]

            # Return the appropriate matrix based on mechanism
            if mechanism in ['mi_gppm', 'mi']:
                return example_data.get('difference_matrix')
            elif mechanism == 'gppm':
                return example_data.get('logp_cond')
            elif mechanism == 'tvd_mi':
                return example_data.get('tvd_mi_matrix')
            elif mechanism in ['judge_with_context', 'judge_without_context']:
                return example_data.get('win_matrix')

        return None

    def get_ground_truth_labels(self, mechanism: str) -> Optional[np.ndarray]:
        """Extract ground truth labels for AUC calculation."""
        if mechanism not in self.matrices:
            return None
            
        data = self.matrices[mechanism]
        
        # Try to find ground truth labels in the data
        if 'ground_truth' in data:
            return np.array(data['ground_truth'])
        elif 'labels' in data:
            return np.array(data['labels'])
        elif 'true_labels' in data:
            return np.array(data['true_labels'])
        
        # For peer prediction, we might need to infer from condition names
        # Assuming binary classification where some conditions are "positive" class
        condition_keys = data.get('condition_keys', [])
        num_examples = data.get('num_examples', 0)
        
        if condition_keys and num_examples > 0:
            # This is a placeholder - you'll need to define which conditions are positive/negative
            print(f"Warning: No explicit ground truth found for {mechanism}")
            print(f"Available conditions: {condition_keys}")
            return None
            
        return None
    
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
                              problematic_conditions: List[str] = None) -> np.ndarray:
        """Calculate AUC for each item separately to avoid pooling bias.
        
        Returns:
            Array of per-item AUC scores
        """
        
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
    
    def calculate_auc_with_ci(self, mechanism: str, faithful_conditions: List[str] = None,
                             problematic_conditions: List[str] = None) -> Tuple[float, float, float]:
        """Calculate macro-averaged AUC with bootstrap confidence intervals.
        
        Returns:
            (mean_auc, lower_ci, upper_ci)
        """
        per_item_aucs = self.calculate_per_item_auc(mechanism, faithful_conditions, problematic_conditions)
        
        if len(per_item_aucs) == 0:
            return np.nan, np.nan, np.nan
            
        return self.bootstrap_confidence_interval(per_item_aucs)
    
    def plot_roc_curve_with_ci(self, mechanism: str, faithful_conditions: List[str] = None,
                               problematic_conditions: List[str] = None, save_path: Optional[str] = None):
        """Plot ROC curve with confidence band from per-item analysis."""

        # Get per-item AUCs for the title
        per_item_aucs = self.calculate_per_item_auc(mechanism, faithful_conditions, problematic_conditions)
        if len(per_item_aucs) == 0:
            print(f"Could not calculate per-item AUCs for {mechanism}")
            return

        mean_auc, lower_ci, upper_ci = self.bootstrap_confidence_interval(per_item_aucs)

        # For visualization, we'll aggregate data across all items
        # This is just for the curve shape, not for the reported AUC
        all_y_true = []
        all_y_scores = []

        individual_key = f"{mechanism}_individual"
        if individual_key not in self.matrices:
            return

        condition_keys = self.metadata[mechanism].get('condition_keys', [])
        faithful_indices = [condition_keys.index(c) for c in faithful_conditions if c in condition_keys]
        problematic_indices = [condition_keys.index(c) for c in problematic_conditions if c in condition_keys]
        
        for example_idx, example_data in self.matrices[individual_key].items():
            ex_keys = example_data.get('condition_keys', [])
            faithful_indices    = [ex_keys.index(c) for c in faithful_conditions if c in ex_keys]
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
                    score = self.get_symmetric_score(example_data, mechanism, faithful_idx, problematic_idx)
                    if score is not None and np.isfinite(score):
                        all_y_true.append(0)
                        all_y_scores.append(score)
        
        if len(set(all_y_true)) != 2:
            print(f"Insufficient data for ROC curve")
            return
            
        fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)
        pooled_auc = roc_auc_score(all_y_true, all_y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2,
                 label=f'Pooled ROC (AUC = {pooled_auc:.3f}); Macro AUC = {mean_auc:.3f} [{lower_ci:.3f}, {upper_ci:.3f}]')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {mechanism}\nFaithful+Style vs Strategic+LowEffort\n(Macro-averaged over {len(per_item_aucs)} items)')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def validate_auc_requirements(self):
        """Check if we have the necessary data for AUC calculation."""
        print("\nValidating AUC Requirements:")
        print("-" * 50)
        
        for mechanism in self.matrices.keys():
            if '_individual' in mechanism:
                continue
                
            print(f"\n{mechanism}:")
            
            # Check for ground truth
            gt = self.get_ground_truth_labels(mechanism)
            if gt is not None:
                print(f"  ✓ Ground truth labels found: {len(gt)} examples")
            else:
                print(f"  ✗ No explicit ground truth labels found")
            
            # Check for conditions and scores
            if mechanism in self.metadata:
                conditions = self.metadata[mechanism].get('condition_keys', [])
                print(f"  Conditions available: {conditions}")
                
                # Check if we can get scores for each condition
                for condition in conditions:
                    scores = self.get_condition_scores(mechanism, condition)
                    if len(scores) > 0:
                        print(f"    ✓ {condition}: {len(scores)} scores available")
                    else:
                        print(f"    ✗ {condition}: No scores found")

    def summarize(self):
        """Print a summary of loaded data."""
        print("Loaded Matrices Summary:")
        print("-" * 50)

        for mechanism, data in self.matrices.items():
            if '_individual' not in mechanism:
                print(f"\n{mechanism}:")
                if mechanism in self.metadata:
                    meta = self.metadata[mechanism]
                    print(f"  Task type: {meta.get('task_type')}")
                    print(f"  Num examples: {meta.get('num_examples')}")
                    print(f"  Conditions: {', '.join(meta.get('condition_keys', []))}")

                # Show data structure
                if isinstance(data, dict):
                    print(f"  Data keys: {list(data.keys())}")

                    # Show score ranges for different data structures
                    score_keys = [k for k in data.keys() if 'scores' in k or 'rates' in k]
                    for score_key in score_keys:
                        if score_key in data and data[score_key]:
                            scores_data = data[score_key]
                            print(f"  {score_key}:")

                            if isinstance(scores_data, dict):
                                # Dictionary of condition -> scores
                                for condition, scores in scores_data.items():
                                    if scores:
                                        mean_score = np.mean(scores)
                                        print(f"    {condition}: mean={mean_score:.3f}, n={len(scores)}")
                            elif isinstance(scores_data, list):
                                # List of scores with condition_keys
                                condition_keys = data.get('condition_keys', [])
                                if len(condition_keys) == len(scores_data):
                                    for condition, score in zip(condition_keys, scores_data):
                                        print(f"    {condition}: {score}")
                                else:
                                    print(f"    List of {len(scores_data)} scores (no condition mapping)")
            else:
                # Individual examples
                print(f"\n{mechanism}: {len(data)} examples loaded")

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

    def calculate_judge_response_auc(self, mechanism: str,
                                    good_labels=None, bad_labels=None):
        """
        Response-level AUC: score = orientation-invariant win prob vs Reference.
        Macro-average over items (like your pair AUC).
        """

        key = f"{mechanism}_individual"
        if key not in self.matrices: 
            return np.array([])

        per_item_aucs = []
        for _, ex in self.matrices[key].items():
            q, conds = self._judge_quality_vector(ex)
            good_idx = [conds.index(c) for c in good_labels if c in conds]
            bad_idx  = [conds.index(c) for c in bad_labels  if c in conds]
            if not good_idx or not bad_idx:
                continue
            y = np.array([1]*len(good_idx) + [0]*len(bad_idx))
            s = np.concatenate([q[good_idx], q[bad_idx]])
            # require both classes present and some variability
            if len(set(y)) == 2 and np.isfinite(s).all() and np.std(s) > 0:
                try:
                    per_item_aucs.append(roc_auc_score(y, s))
                except Exception:
                    pass
        return np.array(per_item_aucs)

    def calculate_judge_pairwise_accuracy(self, mechanism: str,
                                        good_labels=None, bad_labels=None,
                                        max_pairs_per_item: int = 2000,
                                        rng: Optional[np.random.Generator] = None):
        """
        Pairwise accuracy on (GOOD,BAD): prob judge says GOOD > BAD.
        Uses orientation-invariant pair prob: 0.5*(W[g,b] + 1 - W[b,g]).
        Returns per-item accuracies (macro-ready).
        """
        if rng is None:
            rng = np.random.default_rng(0)

        key = f"{mechanism}_individual"
        if key not in self.matrices:
            return np.array([])

        accs = []
        for _, ex in self.matrices[key].items():
            W = ex['win_matrix'].astype(float)
            conds = ex['condition_keys']
            g_idx = [conds.index(c) for c in good_labels if c in conds]
            b_idx = [conds.index(c) for c in bad_labels  if c in conds]
            if not g_idx or not b_idx:
                continue

            # all pairs or a capped random subset
            pairs = [(g, b) for g in g_idx for b in b_idx]
            if len(pairs) > max_pairs_per_item:
                pairs = [pairs[i] for i in rng.choice(len(pairs), size=max_pairs_per_item, replace=False)]

            # orientation-invariant win prob that GOOD beats BAD
            wins = []
            for g, b in pairs:
                p = 0.5*(W[g, b] + (1.0 - W[b, g]))
                if np.isfinite(p):
                    wins.append(1.0 if p > 0.5 else 0.0)
            if wins:
                accs.append(float(np.mean(wins)))
        return np.array(accs)

def process_single_folder(folder_path: str, output_lines: List[str]):
    """Process a single evaluation results folder and append results to output_lines."""
    folder_name = Path(folder_path).name

    try:
        # Load all matrices
        loader = MatrixLoader(folder_path)
        matrices = loader.load_all_matrices()

        # Skip if no data loaded
        if not matrices:
            output_lines.append(f"\n{folder_name}: No data found\n")
            return

        # Determine task type from metadata
        task_type = None
        for mechanism in ['mi', 'gppm', 'tvd_mi', 'judge_with_context']:
            if mechanism in loader.metadata:
                task_type = loader.metadata[mechanism].get('task_type', 'unknown')
                break

        output_lines.append(f"\n{'='*80}")
        output_lines.append(f"Folder: {folder_name}")
        output_lines.append(f"Task Type: {task_type}")
        output_lines.append(f"{'='*80}\n")

        # Mechanisms to consider
        pair_mechanisms = ['mi', 'gppm', 'tvd_mi']

        # Calculate AUC for each mechanism
        output_lines.append("AUC Results (macro-averaged with 95% CI):")
        output_lines.append("-" * 60)

        for mechanism in pair_mechanisms:
            per_item_aucs = loader.calculate_per_item_auc(mechanism)

            if len(per_item_aucs) > 0:
                mean_auc, lower_ci, upper_ci = loader.bootstrap_confidence_interval(per_item_aucs)
                output_lines.append(f"{mechanism:20s}: AUC = {mean_auc:.3f} [{lower_ci:.3f}, {upper_ci:.3f}] (n={len(per_item_aucs)} items)")
            else:
                output_lines.append(f"{mechanism:20s}: Could not calculate AUC")

        # Also check judge with context using the similarity mapping
        per_item_aucs = loader.calculate_per_item_auc('judge_with_context')
        if len(per_item_aucs) > 0:
            mean_auc, lower_ci, upper_ci = loader.bootstrap_confidence_interval(per_item_aucs)
            output_lines.append(f"{'judge_with_context':20s}: AUC = {mean_auc:.3f} [{lower_ci:.3f}, {upper_ci:.3f}] (n={len(per_item_aucs)} items)")

        output_lines.append("")  # Empty line after each folder

    except Exception as e:
        output_lines.append(f"\n{folder_name}: Error processing - {str(e)}\n")

def plot_all_mechanism_rocs(loader: MatrixLoader, save_path: Optional[str] = None,
                           faithful_conditions: List[str] = None,
                           problematic_conditions: List[str] = None):
    """Plot ROC curves for all mechanisms on the same plot."""

    plt.figure(figsize=(10, 8))

    # Define mechanisms to plot with colors and styles
    mechanisms = [
        ('tvd_mi', 'TVD-MI', 'blue', '-'),
        ('mi', 'MI', 'green', '-'),
        ('gppm', 'GPPM', 'red', '-'),
        ('judge_with_context', 'Judge w/ Context', 'purple', '--'),
        ('judge_without_context', 'Judge w/o Context', 'orange', '--')
    ]

    # Track which mechanisms were successfully plotted
    plotted_mechanisms = []

    for mechanism, label, color, linestyle in mechanisms:
        # Skip if mechanism not available
        if mechanism not in loader.matrices:
            continue

        # Get per-item AUCs
        per_item_aucs = loader.calculate_per_item_auc(mechanism, faithful_conditions, problematic_conditions)
        if len(per_item_aucs) == 0:
            print(f"Could not calculate per-item AUCs for {mechanism}")
            continue

        mean_auc, lower_ci, upper_ci = loader.bootstrap_confidence_interval(per_item_aucs)

        # For visualization, aggregate data across all items
        all_y_true = []
        all_y_scores = []

        individual_key = f"{mechanism}_individual"
        if individual_key not in loader.matrices:
            continue

        for example_idx, example_data in loader.matrices[individual_key].items():
            ex_keys = example_data.get('condition_keys', [])
            faithful_indices = [ex_keys.index(c) for c in faithful_conditions if c in ex_keys]
            problematic_indices = [ex_keys.index(c) for c in problematic_conditions if c in ex_keys]

            if not faithful_indices or not problematic_indices:
                continue

            # Faithful-faithful pairs (positive class)
            for i in range(len(faithful_indices)):
                for j in range(i + 1, len(faithful_indices)):
                    idx1, idx2 = faithful_indices[i], faithful_indices[j]
                    score = loader.get_symmetric_score(example_data, mechanism, idx1, idx2)
                    if score is not None and np.isfinite(score):
                        all_y_true.append(1)
                        all_y_scores.append(score)

            # Faithful-problematic pairs (negative class)
            for faithful_idx in faithful_indices:
                for problematic_idx in problematic_indices:
                    score = loader.get_symmetric_score(example_data, mechanism, faithful_idx, problematic_idx)
                    if score is not None and np.isfinite(score):
                        all_y_true.append(0)
                        all_y_scores.append(score)

        if len(set(all_y_true)) != 2:
            print(f"Insufficient data for ROC curve for {mechanism}")
            continue

        fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)

        # Plot with label showing macro AUC and CI
        plt.plot(fpr, tpr, color=color, linestyle=linestyle, lw=2,
                label=f'{label}: AUC = {mean_auc:.3f} [{lower_ci:.3f}, {upper_ci:.3f}]')

        plotted_mechanisms.append(label)

    # Plot baseline
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle=':', alpha=0.5, label='Random (AUC = 0.500)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves: All Mechanisms\nFaithful+Style vs Strategic+LowEffort', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add text with number of items
    if plotted_mechanisms:
        first_mechanism = mechanisms[0][0]
        individual_key = f"{first_mechanism}_individual"
        if individual_key in loader.matrices:
            n_items = len(loader.matrices[individual_key])
            plt.text(0.02, 0.98, f'n = {n_items} items', transform=plt.gca().transAxes,
                    verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Process all folders in eval_results and save AUC summary with combined ROC curves."""

    # Get all subdirectories in eval_results
    eval_results_dir = Path("eval_results")
    if not eval_results_dir.exists():
        print(f"Error: {eval_results_dir} directory not found")
        return

    # Collect all subdirectories
    folders = [f for f in eval_results_dir.iterdir() if f.is_dir()]
    folders.sort()  # Sort for consistent output

    if not folders:
        print("No subdirectories found in eval_results")
        return

    # Create output directory for ROC plots
    roc_output_dir = Path("roc_curves")
    roc_output_dir.mkdir(exist_ok=True)

    # Process each folder and collect results
    output_lines = []
    output_lines.append("Peer Prediction Mechanism AUC Summary")
    output_lines.append("=" * 80)
    output_lines.append(f"Generated: {np.datetime64('now')}")
    output_lines.append(f"Total folders: {len(folders)}")
    output_lines.append("")

    for folder in [folders[5]]:  # Process all folders, not just the last one
        print(f"Processing {folder.name}...")

        # Process folder for AUC summary
        process_single_folder(str(folder), output_lines)

        # Generate combined ROC curve for all mechanisms
        try:
            loader = MatrixLoader(str(folder))
            matrices = loader.load_all_matrices()

            if matrices:
                print(f"  Generating combined ROC curve for {folder.name}...")

                # Define conditions
                #faithful_conditions = ['Faithful', 'Objective', 'Comprehensive', 'Neutral Tone',
                #                     'Academic Style', 'Sensationalist', 'Bureaucratic', 
                #                     'Casual Conversational', 'Historical Perspective', 'Poetic',
                #                     'Technical Jargon', 'Euphemistic', 'Overly Technical', 'Minimalist']

                #problematic_conditions = ['Fact Manipulation', 'Selective Omission', 'Misleading Emphasis',
                #                        'Sentiment Flip', 'Conspiracy Theory', 'Contradictory',
                #                        'Context Removal', 'False Attribution', 'Agenda Push', 'Cherry Pick',
                #                        'Low Effort', 'Ultra Concise', 'Template Response',
                #                        'Surface Skim', 'Minimal Detail']

                problematic_conditions = [
                    'Method Shift',
                    'Question Shift', 
                    'Contribution Misrepresent',
                    'Result Manipulation',
                    'Assumption Attack',
                    'Dismissive Expert',
                    'Agenda Push',
                    'Benchmark Obsessed'
                    'Low Effort',
                    'Generic',
                    'Surface Skim',
                    'Template Fill',
                    'Checklist Review'
                ]
                faithful_conditions= [
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
                    'Reference',
                    'Faithful',
                    'Objective Analysis',
                    'Thorough Evaluation'
                ]

                # Generate combined ROC curve
                save_path = roc_output_dir / f"{folder.name}_all_mechanisms_roc.png"
                plot_all_mechanism_rocs(
                    loader,
                    save_path=str(save_path),
                    faithful_conditions=faithful_conditions,
                    problematic_conditions=problematic_conditions
                )
                output_lines.append(f"  Combined ROC curve saved: {save_path}")

        except Exception as e:
            print(f"  Error generating ROC curve for {folder.name}: {str(e)}")
            output_lines.append(f"  Combined ROC curve error: {str(e)}")

    # Save to file
    output_file = "auc_summary.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"\nSummary saved to {output_file}")
    print(f"ROC curves saved to {roc_output_dir}/")

    # Also print to console
    print("\n" + "="*80)
    print("AUC SUMMARY")
    print("="*80)
    print('\n'.join(output_lines))
    
if __name__ == "__main__":
    main()
