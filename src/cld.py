import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.weightstats import ttest_ind
def multcomp_letters(pd_comp_matrix, letters=None, reversed=False):
    """
    Implementation of https://github.com/cran/multcompView/blob/master/R/multcompLetters.R
    """
    if letters is None:
        letters = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.")
    
    n = pd_comp_matrix.shape[0]
    lvls = pd_comp_matrix.index
    comp_matrix = pd_comp_matrix.to_numpy()

    np.fill_diagonal(comp_matrix, False)

    let_mat = np.ones((n, 1), dtype=bool)

    def absorb(A):
        k = A.shape[1]
        for i in range(k-1):
            for j in range(i+1, k):
                if np.all(A[A[:, j], i]):
                    A = np.delete(A, j, axis=1)
                    return absorb(A)
                elif np.all(A[A[:, i], j]):
                    A = np.delete(A, i, axis=1)
                    return absorb(A)
        return A

    distinct_pairs = np.argwhere(comp_matrix)
    for pair in distinct_pairs:
        i, j = pair
        ij_cols = np.logical_and(let_mat[i, :], let_mat[j, :])
        if np.any(ij_cols):
            a1 = let_mat[:, ij_cols]
            a1[i, :] = False
            let_mat[j, ij_cols] = False
            let_mat = np.hstack((let_mat, a1))
            let_mat = absorb(let_mat)

    def sort_cols(B):
        first_row = np.apply_along_axis(lambda x: np.argmax(x), 0, B)
        B = B[:, np.argsort(first_row)]
        first_row = np.apply_along_axis(lambda x: np.argmax(x), 0, B)
        reps = np.diff(first_row) == 0
        if np.any(reps):
            nrep = np.sum(reps)
            irep = np.where(reps)[0]
            for i in irep:
                j = first_row[i] + 1
                B[j:, i:i+nrep] = sort_cols(B[j:, i:i+nrep])
        return B

    let_mat = sort_cols(let_mat)
    if reversed:
        let_mat = let_mat[:, ::-1]

    def make_ltrs(kl, ltrs):
        if kl < len(ltrs):
            return ltrs[:kl]
        ltrecurse = [ltrs[-1] + l for l in ltrs[:-1]] + [ltrs[-1]]
        return ltrs[:-1] + make_ltrs(kl - len(ltrs) + 1, ltrecurse)

    k_ltrs = let_mat.shape[1]
    ltrs = make_ltrs(k_ltrs, letters)
    letters_dict = {lvls[i]: ''.join([ltrs[col] for col in range(k_ltrs) if let_mat[i, col]]) for i in range(n)}

    # monospaced letters
    # letters_dict = {lvl: ''.join([' ' * len(ltrs[col]) if not let_mat[i, col] else ltrs[col] for col in range(k_ltrs)]) for i, lvl in enumerate(lvls)}

    return letters_dict

# mc_results should be a simpletable
def create_comp_matrix_tukey_HSD(mc_results):
    comparisons = mc_results.summary().data[1:]
    levels = sorted(set([row[0] for row in comparisons] + [row[1] for row in comparisons]))
    n = len(levels)
    comp_matrix = np.zeros((n, n), dtype=bool)

    level_indices = {level: idx for idx, level in enumerate(levels)}

    for row in comparisons:
        group1, group2, reject = row[0], row[1], row[6]
        i, j = level_indices[group1], level_indices[group2]
        comp_matrix[i, j] = reject
        comp_matrix[j, i] = reject

    return pd.DataFrame(comp_matrix, index=levels, columns=levels)

def create_comp_matrix_allpair_t_test(mc_results):
    """
    Returns the non corrected p-value matrix. Could also return a boolean matrix by changing `row[3]` to `row[5]
    """
    comparisons = mc_results[0].data[1:]
    levels = sorted(set([row[0] for row in comparisons] + [row[1] for row in comparisons]))
    n = len(levels)
    comp_matrix = np.zeros((n, n))

    level_indices = {level: idx for idx, level in enumerate(levels)}

    for row in comparisons:
        print(row)
        group1, group2, reject = row[0], row[1], row[3]
        i, j = level_indices[group1], level_indices[group2]
        comp_matrix[i, j] = reject
        comp_matrix[j, i] = reject

    return pd.DataFrame(comp_matrix, index=levels, columns=levels)

if __name__ == "__main__":
    def test_tukeys_hsd():
        # Create synthetic dataset
        data = {
            'group': ['A'] * 10 + ['B'] * 10 + ['C'] * 10,
            'value': [1, 2, 1.5, 2.5, 1.2, 2.2, 1.1, 2.1, 1.3, 2.3, 5, 6, 5.5, 6.5, 5.2, 6.2, 5.1, 6.1, 5.3, 6.3, 10, 11, 10.5, 11.5, 10.2, 11.2, 10.1, 11.1, 10.3, 11.3]
        }
        df = pd.DataFrame(data)

        # Generate CLD
        all_data = df['value'].tolist()
        group_labels = df['group'].tolist()
        mc = MultiComparison(all_data, group_labels)
        tukey_result = mc.tukeyhsd()
        print(tukey_result.summary())
        comp_matrix = create_comp_matrix_tukey_HSD(tukey_result)

        result = multcomp_letters(comp_matrix)
        print("Letters:", result)
        
    def test_paired_t_test():
        # Create synthetic dataset
        data = {
            'group': ['A'] * 10 + ['B'] * 10 + ['C'] * 10,
            'value': [1, 2, 1.5, 2.5, 1.2, 2.2, 1.1, 2.1, 1.3, 2.3, 5, 6, 5.5, 6.5, 5.2, 6.2, 5.1, 6.1, 5.3, 6.3, 10, 11, 10.5, 11.5, 10.2, 11.2, 10.1, 11.1, 10.3, 11.3]
        }
        df = pd.DataFrame(data)

        # Generate CLD
        all_data = df['value'].tolist()
        group_labels = df['group'].tolist()
        mc = MultiComparison(all_data, group_labels)
        test_results = mc.allpairtest(ttest_ind)
        print(test_results[0])
        comp_matrix = create_comp_matrix_allpair_t_test(test_results)

        result = multcomp_letters(comp_matrix)
        print("Letters:", result)

        
        
    def test_multcomp_letters():
        threshold = 0.05
        
        p_matrix = np.array([
            [1.00, 0.01, 0.20],
            [0.01, 1.00, 0.03],
            [0.20, 0.03, 1.00]
        ])
        
        comp_matrix = p_matrix < threshold
        
        result = multcomp_letters(comp_matrix)
        print("Letters:", result)

        
        p_matrix = np.array([
            [1.00, 0.2, 0.002],
            [0.2, 1.00, 0.1],
            [0.02, 0.1, 1.00]
        ])
        
        comp_matrix = p_matrix < threshold

        result = multcomp_letters(comp_matrix)
        print("Letters:", result)

