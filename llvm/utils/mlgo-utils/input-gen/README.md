# input-gen utilities

## Generating ComPileLoop from ComPile
``` shell
python3 generate_compile_loop.py --dataset path/to/ComPile/ --output-dataset ./ComPileLoop --end 100
```

USR1 can be sent to get a status report, USR2 can be sent to abort and write out
the current pending database file.

The script also generates JSON files containing a summary of each generated
parquet file. This information can be used to continue generation of the
database from where the process was interrupted. However, the JSON files
interfere with reading the resulting databse so they must be moved out of the
directory. (TODO we should generate them in a separate dir)

## Demo of how to process ComPileLoop

``` shell
python3 generate_compile_loop.py --dataset path/to/ComPileLoop/
```

## Generating inputs for a module

``` shell
python3 input_gen_module.py --module input_module.ll {--entry-all,--entry-function=foo,--entry-marked}
```

## Generating ComPileLoop+Inputs from ComPileLoop

``` shell
python3 input_gen_module.py --module --dataset path/to/ComPileLoop/
```
