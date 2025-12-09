"""SVM and other classifier-based jumper classification using PCA features."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Any
from matplotlib.colors import LinearSegmentedColormap

# Unified color palette - car-inspired gradient scheme
PRIMARY_RED = '#C41E3A'      # Deep red
PRIMARY_ORANGE = '#FF6B35'   # Vibrant orange
PRIMARY_YELLOW = '#FFD23F'   # Golden yellow
PRIMARY_BLUE = '#0066CC'     # Electric blue

# Create gradient colormap
GRADIENT_COLORS = [PRIMARY_RED, PRIMARY_ORANGE, PRIMARY_YELLOW, PRIMARY_BLUE]
GRADIENT_CMAP = LinearSegmentedColormap.from_list('red_orange_yellow_blue', GRADIENT_COLORS, N=100)

# Set consistent font styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder


def classify_jumpers(
    principal_components: np.ndarray,
    participant_names: List[str],
    n_components: int = 4,
) -> Tuple[Dict[str, Any], str, Any]:
    """
    Classify jumps by person (file) using PCA features with various classifiers.
    
    Each file of data represents a unique person. This function classifies
    which person (file) a jump belongs to based on PCA features extracted
    from the jump segments.
    
    Args:
        principal_components: Projected data onto PCs (each row is a jump)
        participant_names: List of participant names (file identifiers), one per jump
        n_components: Number of principal components to use as features
        
    Returns:
        Tuple of (results_dict, best_classifier_name, best_classifier_object)
    """
    print(f"\n{'='*60}")
    print(f"PERSON IDENTIFICATION: CLASSIFYING JUMPS BY PERSON (FILE)")
    print(f"Each file = one unique person")
    print(f"{'='*60}")
    
    # Use first n_components as features
    X = np.real(principal_components[:, :n_components])
    
    # Encode participant names to numeric labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(participant_names)
    class_names = label_encoder.classes_
    
    print(f"\nUsing first {n_components} principal components as features")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of classes (participants): {len(class_names)}")
    print(f"Classes: {', '.join(class_names)}")
    print(f"Samples per class:")
    for name, count in Counter(participant_names).items():
        print(f"  - {name}: {count}")
    
    # Split data for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Comprehensive kernel testing - test many SVM kernel configurations
    classifiers = {}
    
    # Linear kernel with different C values
    for C in [0.1, 1.0, 10.0, 100.0]:
        classifiers[f"Linear SVM (C={C})"] = SVC(kernel='linear', C=C, random_state=42, probability=True)
    
    # RBF kernel with different gamma and C combinations
    for gamma in ['scale', 'auto', 0.001, 0.01, 0.1, 1.0, 10.0]:
        for C in [0.1, 1.0, 10.0, 100.0]:
            gamma_str = str(gamma) if isinstance(gamma, (int, float)) else gamma
            classifiers[f"RBF SVM (gamma={gamma_str}, C={C})"] = SVC(
                kernel='rbf', gamma=gamma, C=C, random_state=42, probability=True
            )
    
    # Polynomial kernel with different degrees and C values
    for degree in [2, 3, 4, 5]:
        for C in [0.1, 1.0, 10.0, 100.0]:
            classifiers[f"Polynomial SVM (degree={degree}, C={C})"] = SVC(
                kernel='poly', degree=degree, C=C, random_state=42, probability=True
            )
    
    # Sigmoid kernel
    for gamma in ['scale', 'auto', 0.001, 0.01, 0.1]:
        for C in [0.1, 1.0, 10.0]:
            gamma_str = str(gamma) if isinstance(gamma, (int, float)) else gamma
            classifiers[f"Sigmoid SVM (gamma={gamma_str}, C={C})"] = SVC(
                kernel='sigmoid', gamma=gamma, C=C, random_state=42, probability=True
            )
    
    # Add non-SVM classifiers for comparison
    classifiers["Random Forest"] = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    classifiers["Logistic Regression"] = LogisticRegression(max_iter=2000, random_state=42)
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"TESTING {len(classifiers)} CLASSIFIER CONFIGURATIONS")
    print("CROSS-VALIDATION RESULTS (5-fold)")
    print(f"{'='*60}")
    
    for idx, (name, clf) in enumerate(classifiers.items(), 1):
        try:
            # Cross-validation
            cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
            
            # Train on full training set
            clf.fit(X_train, y_train)
            
            # Test set predictions
            y_pred = clf.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation predictions on ALL data for final accuracy
            cv_all = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            clf_fresh = type(clf)(**clf.get_params())
            y_pred_cv_all = cross_val_predict(clf_fresh, X, y, cv=cv_all)
            cv_all_accuracy = accuracy_score(y, y_pred_cv_all)
            
            results[name] = {
                'classifier': clf,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': test_accuracy,
                'cv_all_accuracy': cv_all_accuracy,  # Final accuracy on all 145 jumps
                'predictions': y_pred,
            }
            
            print(f"[{idx}/{len(classifiers)}] {name}:")
            print(f"  CV Accuracy (train): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  Final CV Accuracy (all {len(y)} jumps): {cv_all_accuracy:.4f}")
        except Exception as e:
            print(f"[{idx}/{len(classifiers)}] {name}: FAILED - {str(e)}")
            continue
    
    # Find best classifier by final CV accuracy on all data
    best_name = max(results.keys(), key=lambda k: results[k]['cv_all_accuracy'])
    best_clf = results[best_name]['classifier']
    
    # Sort results by final CV accuracy for summary
    sorted_results = sorted(results.items(), key=lambda x: x[1]['cv_all_accuracy'], reverse=True)
    
    print(f"\n{'='*60}")
    print(f"BEST CLASSIFIER: {best_name}")
    print(f"CV Accuracy (train): {results[best_name]['cv_mean']:.4f} (+/- {results[best_name]['cv_std'] * 2:.4f})")
    print(f"Test Accuracy: {results[best_name]['test_accuracy']:.4f}")
    print(f"Final CV Accuracy (all {len(y)} jumps): {results[best_name]['cv_all_accuracy']:.4f}")
    print(f"{'='*60}")
    
    # Print top 10 classifiers by final accuracy
    print(f"\n{'='*60}")
    print("TOP 10 CLASSIFIERS BY FINAL ACCURACY (All 145 Jumps)")
    print(f"{'='*60}")
    print(f"{'Rank':<6} {'Classifier':<50} {'Final Accuracy':<15} {'CV (Train)':<15} {'Test':<10}")
    print("-" * 100)
    for rank, (name, result) in enumerate(sorted_results[:10], 1):
        print(f"{rank:<6} {name:<50} {result['cv_all_accuracy']:.4f}          {result['cv_mean']:.4f}        {result['test_accuracy']:.4f}")
    print(f"{'='*60}")
    
    # Get cross-validation predictions for ALL data (all 145 jumps)
    # Create a fresh instance of the best classifier for cross-validation
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION PREDICTIONS ON ALL DATA (145 jumps)")
    print(f"{'='*60}")
    cv_all = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Create a fresh classifier instance (same type as best_clf)
    best_clf_fresh = type(best_clf)(**best_clf.get_params())
    y_pred_cv_all = cross_val_predict(best_clf_fresh, X, y, cv=cv_all)
    cv_all_accuracy = accuracy_score(y, y_pred_cv_all)
    print(f"Cross-validation accuracy on all {len(y)} samples: {cv_all_accuracy:.4f}")
    
    # Confusion matrix for all data (CV predictions)
    cm_all = confusion_matrix(y, y_pred_cv_all)
    print(f"\nConfusion Matrix - All {len(y)} Jumps (Cross-Validation Predictions):")
    print("Rows = True labels, Columns = Predicted labels")
    print(f"{'':<15}", end="")
    for name in class_names:
        print(f"{name[:10]:>12}", end="")
    print()
    for i, name in enumerate(class_names):
        print(f"{name[:14]:<15}", end="")
        for j in range(len(class_names)):
            print(f"{cm_all[i, j]:>12}", end="")
        print()
    print(f"\nTotal samples in confusion matrix: {cm_all.sum()} (should equal {len(y)})")
    
    # Detailed classification report for all data (CV predictions)
    print(f"\nDetailed Classification Report - All Data ({best_name}, Cross-Validation):")
    print(classification_report(y, y_pred_cv_all, target_names=class_names))
    
    # Test set confusion matrix (for comparison)
    cm_test = confusion_matrix(y_test, results[best_name]['predictions'])
    print(f"\n{'='*60}")
    print(f"Test Set Confusion Matrix ({best_name}) - {len(y_test)} samples:")
    print("Rows = True labels, Columns = Predicted labels")
    print(f"{'':<15}", end="")
    for name in class_names:
        print(f"{name[:10]:>12}", end="")
    print()
    for i, name in enumerate(class_names):
        print(f"{name[:14]:<15}", end="")
        for j in range(len(class_names)):
            print(f"{cm_test[i, j]:>12}", end="")
        print()
    
    # Use the all-data confusion matrix for plotting
    cm = cm_all
    
    # Plot confusion matrix (using all-data CV predictions)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=GRADIENT_CMAP)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           title=f'Confusion Matrix - {best_name}\n(All {len(y)} Jumps, Cross-Validation Predictions)',
           ylabel='True Participant',
           xlabel='Predicted Participant')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    # Save figure
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    save_path = project_root / "results" / "plots" / "svm_confusion.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved SVM confusion matrix to {save_path}")
    
    plt.show()
    
    # Feature importance (if applicable)
    if hasattr(best_clf, 'feature_importances_'):
        print(f"\nFeature Importances ({best_name}):")
        importances = best_clf.feature_importances_
        for i, imp in enumerate(importances):
            print(f"  PC{i+1}: {imp:.4f}")
    elif hasattr(best_clf, 'coef_'):
        print(f"\nCoefficients ({best_name}):")
        coef = best_clf.coef_
        for i in range(n_components):
            print(f"  PC{i+1}: {np.mean(np.abs(coef[:, i])):.4f} (avg abs coefficient)")
    
    return results, best_name, best_clf


def main() -> None:
    """Main function to run classification - can be called standalone."""
    # Import PCA functions to get the data
    import sys
    from pathlib import Path
    # Add current directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    from PCA_jumps import extract_jump_segments, perform_pca, plot_average_jump_profile
    
    print("="*60)
    print("SVM CLASSIFICATION FOR PERSON IDENTIFICATION")
    print("Goal: Classify which person (file) each jump belongs to")
    print("Each file of data = one unique person")
    print("="*60)
    
    print("\nExtracting jump segments from all participants...")
    jump_matrix, participant_names = extract_jump_segments(window_size=150)
    
    print("\nPlotting average jump profile across all people...")
    plot_average_jump_profile(jump_matrix, participant_names, window_size=150)
    
    print("\nPerforming PCA...")
    principal_components, eigenvalues, explained_variance = perform_pca(jump_matrix)
    
    print("\nRunning Classification Analysis...")
    results, best_name, best_clf = classify_jumpers(
        principal_components, 
        participant_names, 
        n_components=4
    )
    
    print("\nClassification analysis complete!")
    print(f"\nBest classifier: {best_name}")


if __name__ == "__main__":
    main()



