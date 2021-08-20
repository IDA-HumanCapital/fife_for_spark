import processors, utils


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

    """
    config = parse_config()
    spark_df = utils.create_example_data2(n_persons = 100, n_periods = 12)
    data_processor = processors.PanelDataProcessor(config, spark_df, shuffle_parts = 50)
    data_processor.build_processed_data()
    data_processor.data.show()

if __name__ == '__main__':
    main()
