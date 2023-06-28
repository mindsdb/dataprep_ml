from dataprep_ml.recommenders import RecommenderPreprocessor
import scipy as sp


def test_preprocessing_cf(recommender_interaction_data):
    """Tests helper function for preprocessing"""

    rec_preprocessor = RecommenderPreprocessor(
        interaction_data=recommender_interaction_data,
        user_id_column_name="userId",
        item_id_column_name="movieId",
    )

    preprocessed_data = rec_preprocessor.preprocess()

    # check ids are int64
    assert (
        preprocessed_data.interaction_df[
            [rec_preprocessor.user_id_column_name, rec_preprocessor.item_id_column_name]
        ]
        .dtypes[preprocessed_data.interaction_df.dtypes == "int64"]
        .all()
    )

    # check interaction are equal to 1 or -1 e.g. positive or negative
    assert (
        preprocessed_data.interaction_df["interaction"]
        .apply(lambda x: x == -1 or x == 1)
        .all()
    )

    # check interaction matrix is the expected shape
    assert preprocessed_data.interaction_matrix.shape == (503, 89)
    assert isinstance(preprocessed_data.interaction_matrix, sp.sparse.coo_matrix)

