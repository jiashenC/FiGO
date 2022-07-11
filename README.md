# FiGO

# Installation 
```bash
conda env create -f environment.yml
```

## Example Run
The easiest way to evaluate the idea of FiGO with other systems is to use cached results (avoid extra inference time).
```bash
# run q1 end-to-end execution with all system implementations
python experiment/end_to_end.py --cache-path $PWD/demo-res --query q1
```