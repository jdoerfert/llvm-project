# input-gen utilities

## Generating ComPileLoop from ComPile
``` shell
python3 generate_com_pile_loop.py --dataset path/to/ComPile/ --output-dataset ./ComPileLoop --output-dataset-json ./ComPileLoopJson --end 100
```

USR1 can be sent to get a status report, USR2 can be sent to abort and write out
the current pending database file.

The script also generates JSON files containing a summary of each generated
parquet file. This information can be used to continue generation of the
database from where the process was interrupted. 

## Demo of how to process ComPileLoop

``` shell
python3 process_com_pile_loop_demo.py --dataset path/to/ComPileLoop/
```

## Generating inputs for a module

``` shell
python3 input_gen_module.py --module input_module.ll {--entry-all,--entry-function=foo,--entry-marked}
```

## Generating ComPileLoop+Inputs from ComPileLoop

``` shell
python3 generate_com_pile_loop_inputs.py --dataset path/to/ComPileLoop/ --output-dataset ./ComPileLoopInputs/ --output-dataset-json ./ComPileLoopInputsJson
```

The `process_com_pile_loop_demo.py` script can also be used with ComPileLoop+Inputs.
