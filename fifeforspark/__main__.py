from time import time
from warnings import warn
from fifeforspark import processors, utils, gbt_modelers


def parse_config() -> dict:
    """
    Parse configuration parameters specified in the command line.

    Returns:
        Configuration dictionary generated from the command line input
    """

    parser = utils.FIFEArgParser()
    args = parser.parse_args()
    config = {}
    config.update({k: v for k, v in vars(args).items() if v is not None})
    return config


def main():
    """
    Executable code of FIFE when run from the command line

    Returns:
        None
    """
    warn(
        """The current __main__.py implementation is not meant to be a full command-line
         interface for running FIFEForSpark without writing any code, but is instead only meant 
         to test and benchmark a pre-set FIFEForSpark pipeline. Further implementation will come
         in future versions."""
    )
    pdp_start = time()
    config = parse_config()
    spark_df = utils.create_example_data(n_persons=100, n_periods=12)
    data_processor = processors.PanelDataProcessor(config, spark_df, shuffle_parts=20)
    data_processor.build_processed_data()
    data_processor.data.show()
    pdp_end = time()
    print("PDP time:", round(pdp_end - pdp_start, 2), "seconds.")

    model_train_start = time()
    modeler_class = gbt_modelers.GBTSurvivalModeler
    modeler = modeler_class(config=config, data=data_processor.data)
    modeler.build_model()
    model_train_end = time()
    print(
        "Model training time:", round(model_train_end - model_train_start, 2), "seconds"
    )

    model_forecast_start = time()
    print(modeler.forecast())
    model_forecast_end = time()
    print(
        "Model forecast time:",
        round(model_forecast_end - model_forecast_start, 2),
        "seconds",
    )

    modeler.n_intervals = modeler.set_n_intervals()
    modeler.build_model(n_intervals=modeler.n_intervals)

    model_evaluate_start = time()
    print(modeler.evaluate())
    model_evaluate_end = time()
    print(
        "Model evaluate time:",
        round(model_evaluate_end - model_evaluate_start, 2),
        "seconds",
    )


if __name__ == "__main__":
    main()
