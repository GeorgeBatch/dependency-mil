import os
from typing import List, Dict, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics as sklearn_metrics

# constants
DPI = 300
FIG_SIZE = (5, 3)


class NoPositiveLabelsError(Exception):
    pass

class NoNegativeLabelsError(Exception):
    pass


def ranking_auc(
    scores: Union[List[float], np.ndarray],
    labels: Union[List[Union[int, str]], np.ndarray],
    pos_label: Union[int, str],
    greater_is_better: bool = True,
    top_k: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Compute the ranking AUC of a ranking list of elements and return the values for plotting.

    Parameters:
    - scores: array-like, scores assigned to the elements
    - labels: array-like, true labels of the elements
    - pos_label: int or str, the label of the positive class
    - greater_is_better: bool, whether higher scores indicate better ranking (default: True)
    - top_k: int, top-k elements to consider for statistics (default: None)
    - verbose: bool, whether to print detailed information (default: False)

    Returns:
    - result: dict, containing x and y values for the plot, average y value, AUC values, and other statistics
    """
    # Ensure inputs are numpy arrays
    scores = np.array(scores)
    labels = np.array(labels)

    # cosine similarity is a distance, so we want to rank the closest ones higher
    if not greater_is_better:
        scores = [-score for score in scores]

    # Convert labels to a binary array
    positive_labels = (labels == pos_label).astype(int)

    # when scores are similarities or probabilities of the positive class, we rank them
    #     from highest to lowest (greater_is_better, descending order) => reverse=True
    # when scores are distances, we have already reversed the sign so the closest ones
    #     have highest negative distance. Again we rank them
    #     from highest to lowest (descending order) => reverse=True
    sorted_matches = sorted(
        [tup for tup in zip(positive_labels, scores)], key=lambda x: x[1], reverse=True
    )

    n_elements = len(labels)
    total_possible_matches = positive_labels.sum()
    if total_possible_matches == 0:
        raise NoPositiveLabelsError(f"No positive labels found for pos_label={pos_label}.")
    if total_possible_matches == n_elements:
        raise NoNegativeLabelsError(f"No negative labels found for pos_label={pos_label}.")
    total_negative_matches = n_elements - total_possible_matches
    cumulative_ranked_matches = np.array([tup[0] for tup in sorted_matches]).cumsum()
    cumulative_tpr_from_total = cumulative_ranked_matches / total_possible_matches

    if verbose:
        print("n_elements", n_elements)
        print("total_possible_matches\n", total_possible_matches, "\n")
        print("cumulative_ranked_matches\n", cumulative_ranked_matches, "\n")

    # --------------------------------------------------------------------------
    # Cumulative tpr from Cumulative Possible Matches
    # --------------------------------------------------------------------------
    cumulative_possible_matches = np.array([total_possible_matches] * n_elements)
    cumulative_possible_matches[:total_possible_matches] = np.arange(
        1, total_possible_matches + 1, 1
    )

    if verbose:
        print("cumulative_possible_matches\n", cumulative_possible_matches, "\n")
        print("ratio\n", cumulative_ranked_matches / cumulative_possible_matches, "\n")

    cumulative_tpr_from_cumulative_possible_matches = (
        cumulative_ranked_matches / cumulative_possible_matches
    )

    x = np.arange(1, n_elements + 1)
    y = cumulative_tpr_from_cumulative_possible_matches
    average_y = np.mean(y)
    auc = sklearn_metrics.auc(x=x, y=y) / (x[-1] - x[0])

    # --------------------------------------------------------------------------
    # Top-K stats
    # --------------------------------------------------------------------------
    # cumulative_false_ranked_matches = cumulative_possible_matches - cumulative_ranked_matches
    population_proportion = total_possible_matches / n_elements

    if verbose:
        print("population_proportion:", population_proportion)

    if top_k and verbose:
        # subtract 1 because of 0-indexing
        print(
            f"Matches in top-{top_k} ranked ROIs:", cumulative_ranked_matches[top_k - 1]
        )
        print(
            f"Coverage from total possible matches ({total_possible_matches}) in top-{top_k} ranked ROIs",
            cumulative_tpr_from_total[top_k - 1],
        )
        print(
            f"Coverage from cumulative possible matches ({cumulative_possible_matches[top_k-1]}) in top-{top_k} ranked ROIs",
            cumulative_tpr_from_cumulative_possible_matches[top_k - 1],
        )

    # --------------------------------------------------------------------------
    # Expected case: positive matches are uniformly distributed
    # --------------------------------------------------------------------------
    y_expected_case = np.concatenate(
        (
            [population_proportion for _ in range(total_possible_matches - 1)],
            np.linspace(
                start=population_proportion, stop=1, num=total_negative_matches + 1
            ),
        )
    )
    auc_expected_case = 0.5 + (
        (total_possible_matches - x[0]) * population_proportion / 2
    ) / (n_elements - x[0])
    assert (
        auc_expected_case
        - (sklearn_metrics.auc(x=x, y=y_expected_case) / (x[-1] - x[0]))
        < 1e-6
    ), "Expected Case AUC calculation is wrong"

    # --------------------------------------------------------------------------
    # Worst case: positive matches are at the end
    # --------------------------------------------------------------------------
    y_worst_case = np.concatenate(
        (
            [0 for _ in range(total_negative_matches - 1)],
            np.linspace(start=0, stop=1, num=total_possible_matches + 1),
        )
    )
    auc_worst_case = (total_possible_matches / 2) / (n_elements - 1)
    assert (
        auc_worst_case - (sklearn_metrics.auc(x=x, y=y_worst_case) / (x[-1] - x[0]))
        < 1e-6
    ), "Worst Case AUC calculation is wrong"

    # --------------------------------------------------------------------------

    result = {
        "x": x,
        # y-s
        "y": y,
        "average_y": average_y,
        "y_expected_case": y_expected_case,
        "y_worst_case": y_worst_case,
        # auc values
        "auc": auc,
        "auc_expected_case": auc_expected_case,
        "auc_worst_case": auc_worst_case,
        # other stats
        "population_proportion": population_proportion,
        "total_possible_matches": total_possible_matches,
        "n_elements": n_elements,
    }

    return result


def plot_ranking_curves(
    x: np.ndarray,
    y: np.ndarray,
    y_expected_case: np.ndarray,
    y_worst_case: np.ndarray,
    auc: float,
    auc_expected_case: float,
    auc_worst_case: float,
    plot_title: Optional[str] = None,
    show: bool = False,
    save: bool = False,
    save_dir: Optional[str] = None,
    save_name: Optional[str] = None,
    save_ext: Optional[str] = None,
) -> None:
    """
    Plot ranking curves including the ranked, expected case, and worst case scenarios.

    Parameters:
    - x: array-like, x-axis values
    - y: array-like, y-axis values for the ranked curve
    - y_expected_case: array-like, y-axis values for the expected case curve
    - y_worst_case: array-like, y-axis values for the worst case curve
    - auc: float, AUC value for the ranked curve
    - auc_expected_case: float, AUC value for the expected case curve
    - auc_worst_case: float, AUC value for the worst case curve
    - plot_title: str, title of the plot (default: None)
    - show: bool, whether to show the plot (default: False)
    - save: bool, whether to save the plot (default: False)
    - save_dir: str, directory to save the plot (required if save is True)
    - save_name: str, name of the saved plot (required if save is True)
    - save_ext: str, extension of the saved plot (required if save is True)
    """
    if save:
        assert save_dir is not None, "Please provide a directory to save the figure"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")
        assert save_name is not None, "Please provide a name for the saved figure"
        assert save_ext is not None, "Please provide an extension for the saved figure"

    # to fit into the miccai paper
    plt.rcParams["figure.figsize"] = FIG_SIZE

    # --------------------------------------------------------------------------
    # ranking curve
    plt.plot(
        x,
        y,
        color="darkorange",
        # color="navy", lw=2, linestyle="--",
        label=f"ranked: Normalized AUC = {auc.round(2)}",
    )

    # --------------------------------------------------------------------------
    # expected case ranking curve
    plt.plot(
        x,
        y_expected_case,
        lw=2,
        linestyle="--",
        color="navy",
        label=f"random: Normalized AUC = {np.round(auc_expected_case, 2)}",
    )

    # --------------------------------------------------------------------------
    # worst case ranking curve
    plt.plot(
        x,
        y_worst_case,
        lw=2,
        linestyle="--",
        color="black",
        label=f"worst: Normalized AUC = {np.round(auc_worst_case, 2)}",
    )

    # --------------------------------------------------------------------------
    # plot settings
    plt.ylim([-0.05, 1.05])
    plt.xlabel("top-n ranked examples")
    plt.ylabel("recall")
    if plot_title:
        plt.title(plot_title)

    plt.legend()
    # plt.legend(loc="lower right")
    # plt.legend(loc="upper left")

    plt.tight_layout()
    if save:
        plt.savefig(f"{save_dir}/{save_name}.{save_ext}", dpi=DPI)
    
    if show:
        plt.show()
    else:
        plt.close()


def unsupervised_example() -> None:
    # Sample data
    sample_scores = np.array([1, 2, 3, 4, 5, 6, 7]) # distance from the query example
    sample_labels = np.array([1, 0, 1, 1, 0, 0, 0])

    # Add a green star at score 0
    plt.scatter(0, 1, color="green", marker="*", s=300, label="Query Example")

    # Create a scatter plot
    # Create a scatter plot for green circles
    plt.scatter(sample_scores[sample_labels == 1], np.ones((sample_labels == 1).sum()), color="green", s=150, label="Positive examples")
    # Create a scatter plot for black circles
    plt.scatter(sample_scores[sample_labels == 0], np.zeros((sample_labels == 0).sum()), color="black", s=150, label="Negative examples")
    # Add labels and title
    plt.ylim(-0.5, 3)
    plt.yticks([0, 1], ["Negative", "Positive"])
    plt.xlabel("Distance from the query example")
    plt.ylabel("Class")
    plt.title("Unsupervised Ranking")
    plt.legend()
    # Show the plot
    # plt.show()
    plt.tight_layout()
    plt.savefig("./figures/ranking_curves/example/unsupervised_data.png")
    plt.clf()  # Clear the current figure

    result = ranking_auc(
        scores=sample_scores,
        labels=sample_labels,
        pos_label=1,
        greater_is_better=False # lower distance means higher rank
    )
    # print(result)

    plot_ranking_curves(
        x=result["x"],
        y=result["y"],
        y_expected_case=result["y_expected_case"],
        y_worst_case=result["y_worst_case"],
        auc=result["auc"],
        auc_expected_case=result["auc_expected_case"],
        auc_worst_case=result["auc_worst_case"],
        plot_title="Unsupervised Ranking",
        save=True,
        save_dir="./figures/ranking_curves/example",
        save_name="unsupervised_performance",
        save_ext="png",
    )
    plt.clf()  # Clear the current figure


def supervised_examples() -> None:
    # Sample data
    # probability of coming from the positive class
    sample_scores = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    cases_2_labels = {
        'Normal': [1, 0, 1, 0, 0, 1, 0, 0],
        'Worst': [0, 0, 0, 0, 0, 1, 1, 1], # positive examples have the lowest probabilities
        'Best': [1, 1, 1, 0, 0, 0, 0, 0],  # positive examples have the highest probabilities
        'Only Negatives': [0, 0, 0, 0, 0, 0, 0, 0], # no positive examples
        'Only Positives': [1, 1, 1, 1, 1, 1, 1, 1], # no negative examples
    }
    # convert the labels to numpy arrays
    cases_2_labels = {case: np.array(labels) for case, labels in cases_2_labels.items()}


    for case, sample_labels in cases_2_labels.items():

        # Create a scatter plot for blue circles
        plt.scatter(sample_scores[sample_labels == 1], np.ones(sample_labels.sum()), color="blue", s=150, label="Positive examples")
        # Create a scatter plot for black circles
        plt.scatter(sample_scores[sample_labels == 0], np.zeros((sample_labels == 0).sum()), color="black", s=150, label="Negative examples")
        # Add labels and title
        plt.ylim(-0.5, 3)
        plt.xlabel("Probability of positive class")
        plt.ylabel("Class")
        plt.yticks([0, 1], ["Negative", "Positive"])
        plt.title(f"Supervised Ranking: {case} case")
        plt.legend()
        # Show the plot
        # plt.show()
        plt.tight_layout()
        plt.savefig("./figures/ranking_curves/example/supervised_data_" + case.lower().replace(" ", "_") + ".png")
        plt.clf()  # Clear the current figure

        try:
            result = ranking_auc(
                scores=sample_scores,
                labels=sample_labels,
                pos_label=1,
                greater_is_better=True # higher probability means higher rank
            )
        except (NoPositiveLabelsError, NoNegativeLabelsError) as e:
            print(e)
            continue
        # print(result)

        plot_ranking_curves(
            x=result['x'],
            y=result['y'],
            y_expected_case=result["y_expected_case"],
            y_worst_case=result["y_worst_case"],
            auc=result['auc'],
            auc_expected_case=result["auc_expected_case"],
            auc_worst_case=result["auc_worst_case"],
            plot_title=f"Supervised Ranking: {case} case",
            save=True,
            save_dir="./figures/ranking_curves/example",
            save_name="supervised_performance_" + case.lower().replace(" ", "_"),
            save_ext="png",
        )
        plt.clf()  # Clear the current figure


if __name__ == "__main__":
    unsupervised_example()
    supervised_examples()