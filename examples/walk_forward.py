"""Opt-in synthetic walk-forward example; replace inputs/factories with prepared research data."""
# CONFIGURATION LOGIC: Importing or running this file does not train until the caller enables this switch.
RUN_WALK_FORWARD = False

if RUN_WALK_FORWARD:
    # SETUP LOGIC: Install the training extra for these example estimators; the engine itself is library-neutral.
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from model_comparison_engine import TrainableModel, WalkForwardConfig, walk_forward_compare
    from model_comparison_engine.demo import make_demo

    # CONFIGURATION LOGIC: Synthetic targets are generated for software demonstration, not a real predictive claim.
    data = make_demo(n=3840, seed=2026)
    settings = WalkForwardConfig(min_train_dates=10, validation_dates=5, embargo_dates=1,
                                  n_splits=3, holdout_dates=5, holdout_embargo_dates=1)

    def reference_factory():
        # CONFIGURATION LOGIC: Categorical fields and estimator parameters are explicit user choices.
        encoder = ColumnTransformer([('numeric', 'passthrough', ['measure_1']),
            ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['segment'])])
        return make_pipeline(encoder, Ridge(alpha=1.))

    def candidate_factory():
        # CONFIGURATION LOGIC: A fresh full pipeline means its vocabulary and model fit only on that fold's past.
        encoder = ColumnTransformer([('numeric', 'passthrough', ['measure_1', 'measure_2']),
            ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['segment'])])
        return make_pipeline(encoder, RandomForestRegressor(n_estimators=40, min_samples_leaf=15,
                                                           random_state=2026, n_jobs=2))

    # ORCHESTRATION LOGIC: Only this explicit call performs model fitting; final reserved dates stay unscored.
    result = walk_forward_compare(data, target='actual', time_column='time', settings=settings,
        reference=TrainableModel('Synthetic linear', reference_factory, ['measure_1', 'segment']),
        candidate=TrainableModel('Synthetic forest', candidate_factory, ['measure_1', 'measure_2', 'segment']),
        id_column='row_id', entity_column='entity_id', timezone='UTC', unit='units')
    # FILE IO LOGIC: Exact folds, predictions and evidence are saved; no large table is printed.
    report_folder = result.export('reports/walk_forward_demo', slices=['segment'],
                                   interactions=[('segment', 'category')])
