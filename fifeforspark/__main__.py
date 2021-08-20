from fifeforspark import processors, utils, lgb_modelers
from time import time

def parse_config() -> dict:
    """Parse configuration parameters specified in the command line."""
    parser = utils.FIFEArgParser()
    args = parser.parse_args()
    config = {}
    config.update({k: v for k, v in vars(args).items() if v is not None})
    return config

def main():
    pdp_start = time()
    config = parse_config()
    spark_df = utils.create_example_data2(n_persons = 100, n_periods = 12)
    data_processor = processors.PanelDataProcessor(config, spark_df, shuffle_parts = 20)
    data_processor.build_processed_data()
    data_processor.data.show()
    pdp_end = time()
    
    model_train_start = time()
    modeler_class = lgb_modelers.LGBSurvivalModeler    
    modeler = modeler_class(config=config, data=data_processor.data)
    modeler.build_model()
    model_train_end = time()
    
    model_forecast_start = time()
    print(modeler.forecast())
    model_forecast_end = time()
    
    modeler.n_intervals = modeler.set_n_intervals()
    modeler.build_model(n_intervals=modeler.n_intervals)
    
    model_evaluate_start = time()
    print(modeler.evaluate())
    model_evaluate_end = time()
    
    print('PDP time:', round(pdp_end - pdp_start, 2), 'seconds.')
    print('Model training time:', round(model_train_end - model_train_start,2), 'seconds')
    print('Model forecast time:', round(model_forecast_end - model_forecast_start,2), 'seconds')
    print('Model evaluate time:', round(model_evaluate_end - model_evaluate_start,2), 'seconds')

if __name__ == '__main__':
    main()
