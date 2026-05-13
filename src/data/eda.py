"""
Exploratory data analysis: sanity checks + rating/vote distribution plot.
"""

import matplotlib.pyplot as plt
import seaborn as sns

from ..config.paths import FIG_EDA


def print_sanity_checks(df):
    """Print shape, info, missing-value summary, vote/sentiment distributions, date range."""
    print("Cleaned shape:", df.shape)
    print("\nInfo:")
    df.info()

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nVote summary:")
    print(df['vote'].describe())

    print("\nSentiment summary:")
    print(df['sentiment'].describe())

    print("\nDate range:")
    print(df['review_date'].min(), "to", df['review_date'].max())

    print("\nNumber of unique days:")
    print(df['review_day'].nunique())


def plot_rating_helpful_vote_distribution(df, save_path=None, show=True):
    """Two-panel EDA plot: overall-rating distribution + helpful-vote distribution.

    Saves to FIG_EDA/'rating_and_helpful_vote_distribution.png' by default.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.countplot(
        data=df,
        x='overall',
        hue='overall',
        ax=axes[0],
        palette='viridis',
        legend=False,
    )
    axes[0].set_title('Distribution of Overall Ratings')
    axes[0].set_xlabel('Rating')
    axes[0].set_ylabel('Count')

    sns.histplot(df[df['vote'] <= 10]['vote'], bins=11, ax=axes[1],
                 color='coral', discrete=True)
    axes[1].set_title('Distribution of Helpful Votes (Truncated at 10)')
    axes[1].set_xlabel('Helpful Votes')
    axes[1].set_ylabel('Count')

    plt.tight_layout()

    if save_path is None:
        save_path = FIG_EDA / 'rating_and_helpful_vote_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
