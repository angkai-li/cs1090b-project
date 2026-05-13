"""src.data - data loading, cleaning, EDA, time series, panels.

Typical usage:
    from src.data.load import load_or_clean  # high-level load+cache+clean
    from src.data.clean import clean_and_engineer
    from src.data.eda import print_sanity_checks, plot_rating_helpful_vote_distribution
    from src.data.timeseries import build_or_load as build_daily_rating_ts
    from src.data.panels import build_or_load_rating_weekly, plot_weekly_sentiment_one_product
    from src.data.nlp_balanced import build_or_load as build_balanced_nlp
"""
