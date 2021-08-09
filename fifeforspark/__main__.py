import processors, utils

def parse_config() -> dict:
    """Parse configuration parameters specified in the command line."""
    parser = utils.FIFEArgParser()
    args = parser.parse_args()
    config = {}
    config.update({k: v for k, v in vars(args).items() if v is not None})
    return config

def main():
    config = parse_config()
    spark_df = utils.create_example_data2(n_persons = 100, n_periods = 12)
    print(spark_df.dtypes)
    data_processor = processors.PanelDataProcessor(config, spark_df)
    data_processor.build_processed_data()
    data_processor.data.show()

if __name__ == '__main__':
    main()
