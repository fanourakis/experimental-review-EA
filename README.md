# Knowledge Graph Embedding Methods for Entity Alignment: An Experimental Review

The source code of the survey:

N Fanourakis, Vasilis Efthymiou, Dimitris Kotzinos, Vassilis Christophides:
Knowledge graph embedding methods for entity alignment: experimental review. Data Min. Knowl. Discov. (2023)
[paper](https://doi.org/10.1007/s10618-023-00941-9)

## Dataset License

Due to licensing we are not allowed to distribute the datasets bbc-db, imdb-tmdb, tmdb-tvdb, imdb-tvdb, restaurants.
To run the experiments, please download [bbc-db](https://www.csd.uoc.gr/~vefthym/minoanER/datasets.html), [imdb, tmdb, tvdb](https://github.com/ScaDS/MovieGraphBenchmark) and [restaurants](http://oaei.ontologymatching.org/2010/im/).

## Getting Started

You should :
1. Install Anaconda 4.11.0
2. Download wiki-news-300d-1M.vec.zip from https://fasttext.cc/docs/en/english-vectors.html
3. Unzip wiki-news-300d-1M.vec.zip
4. Copy wiki-news-300d-1M.vec to OpenEA/datasets

### AttrE Instructions
```bash
cd AttrE
conda env create --file install/AttrE.yml --name AttrE_env
conda activate AttrE_env
cd "dataset_name" (e.g., restaurants)
python KBA.py
```

### RREA Instructions
```bash
cd RREA_versions
conda env create --file install/RREA.yml --name RREA_env
conda activate RREA_env
python RREA.py
```

### OpenEA Instructions
```bash
cd OpenEA
conda env create --file install/openea.yml --name OpenEA_env
conda activate OpenEA_env
pip install -e .
cd run
python main_from_args.py "predefined_arguments" "dataset_name" "split"
```
For example, for MTransE on tmdb-tvdb for the first split, please execute the following:
```bash
python main_from_args.py ./args/mtranse_args_15K.json D_W_15K_V1 721_5fold/1/
```

### PARIS Instructions:
```bash
cd PARIS
conda env create --file install/entity_match.yml -n entity_match
conda activate entity_match
cd src/experiments
mkdir results
python3 -u ../run_experiment.py \
        --method PARIS\
        --root_dataset "root of dataset folder"\
        --dataset "dataset_name"\
        --dataset_division 721_5fold\
        --out_folder ./results\
        --use_func > test.log
```
### BERT_INT Instructions:
```bash
cd BERT_INT
conda env create --file install/bert_int.yml -n bert_int
python basic_bert_unit/main.py
bash interaction_model/run.sh
```

### Analysis Instructions
```bash
cd analysis
conda env create --file install/analysis.yml --name analysis_env
conda activate analysis_env
```
For Nemenyi Diagrams
```bash
python nemenyi_diagrams.py
```
For Correlations
```bash
python correlation_analysis.py
```
For Time to Accuracy
```bash
python time_to_accuracy.py
```
For Trade Offs
```bash
python trade_offs.py
```

### Statistics Instructions
```bash
please download descriptions_pickles.zip from https://www.dropbox.com/sh/7r2y1x8wx9y921e/AAAh3DabBguCzxk8OlZnL-Mza?dl=0
uzip descriptions_pickles.zip
copy to ./statistics/
cd statistics
conda env create --file install/statistics.yml --name statistics_env
conda activate statistics_env
python statistics.py
```
