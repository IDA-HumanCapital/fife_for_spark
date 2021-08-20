import processors, utils, lgb_modelers

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
    data_processor = processors.PanelDataProcessor(config, spark_df, shuffle_parts = 50)
    data_processor.build_processed_data()
    data_processor.data.show()
    
    modeler_class = lgb_modelers.LGBSurvivalModeler    
    modeler = modeler_class(config=config, data=data_processor.data)
    modeler.build_model(n_intervals=modeler.n_intervals)


if __name__ == '__main__':
    main()
